"""Autonomous neural coefficients, including the manuscript's C2 cutoff."""

import jax
import jax.numpy as jnp
import numpy as np

from .initial import add_initial_params


def cutoff(x):
    # The inner ball is exactly constant. Avoid differentiating sqrt at zero
    # even in the inactive branch of a vectorized conditional.
    radius = jnp.sqrt(jnp.maximum(jnp.vdot(x, x), jnp.asarray(1e-12, x.dtype)))
    s = jnp.clip(radius - 1.0, 0.0, 1.0)
    # Same quintic, factored to avoid cancellation and negative roundoff near 2.
    transition = (1.0 - s)**3 * (1.0 + 3.0 * s + 6.0 * s**2)
    return jnp.where(radius <= 1.0, 1.0, jnp.where(radius >= 2.0, 0.0, transition))


def _init_network(key, dim, width, depth, outputs, dtype, zero_output_weights, output_bias=0.0):
    keys = jax.random.split(key, depth + 1)
    layers = []
    inputs = dim
    for i in range(depth):
        bound = jnp.sqrt(6.0 / (inputs + width))
        w = jax.random.uniform(keys[i], (width, inputs), dtype=dtype,
                               minval=-bound, maxval=bound)
        layers.append({"w": w, "b": jnp.zeros(width, dtype)})
        inputs = width
    if zero_output_weights:
        w = jnp.zeros((outputs, width), dtype)
    else:
        # Both networks use this small output scale when initialized randomly.
        w = 0.01 * jax.random.normal(keys[-1], (outputs, width), dtype) / jnp.sqrt(width)
    layers.append({"w": w, "b": jnp.full((outputs,), output_bias, dtype)})
    return tuple(layers)


def init_params(key, cfg):
    """Draw raw random parameters; data-dependent calibration is a separate step."""
    cfg.validate()
    if cfg.dtype == "float64" and not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 before initializing float64 parameters")
    dtype = jnp.dtype(cfg.dtype)
    drift_key, sigma_key = jax.random.split(key)
    params = {
        "drift": _init_network(drift_key, cfg.dim, cfg.width, cfg.depth, cfg.dim, dtype,
                               zero_output_weights=False),
        "diffusion": (() if cfg.diffusion_mode != "neural" else
                      _init_network(sigma_key, cfg.dim, cfg.diffusion_width, cfg.diffusion_depth, 1, dtype,
                                    zero_output_weights=cfg.initial_sigma is not None,
                                    output_bias=(0.0 if cfg.initial_sigma is None
                                                 else cfg.initial_raw_diffusion_output))),
    }
    return add_initial_params(params, cfg)


def calibrate_drift_initialization(params, normalized_train):
    """Set each hidden drift layer's pre-SiLU RMS to one on fixed training points.

    Use at most the first 1,024 normalized training points, without a random
    draw. Recompute features after each rescaling. The final drift layer and
    all diffusion parameters are preserved. This runs only at initialization.
    """
    return _calibrate_hidden_initialization(params, normalized_train, "drift")


def calibrate_diffusion_initialization(params, normalized_train):
    """Calibrate hidden diffusion layers while preserving the final layer.

    Use the same fixed training batch and RMS target as drift calibration.
    Random output weights give a spatially varying initial field after
    calibration. A prescribed initial_sigma uses zero output weights and
    keeps that constant interior amplitude unchanged.
    """
    return _calibrate_hidden_initialization(params, normalized_train, "diffusion")


def _calibrate_hidden_initialization(params, normalized_train, branch):
    points = np.asarray(normalized_train)
    dim = params[branch][0]["w"].shape[1]
    if (points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != dim
            or np.iscomplexobj(points) or not np.isfinite(points).all()):
        raise ValueError(f"{branch} initialization requires finite normalized training points of shape (N, dim)")
    value = jnp.asarray(points[:1024], dtype=params[branch][0]["w"].dtype)
    calibrated = []
    report = {"method": "data_rms", "samples": int(value.shape[0]),
              "sample_selection": "first training points", "target_preactivation_rms": 1.0,
              "layers": []}
    for index, layer in enumerate(params[branch][:-1], start=1):
        if np.any(np.asarray(layer["b"]) != 0):
            raise ValueError(f"{branch} initialization requires zero hidden biases")
        pre = value @ layer["w"].T + layer["b"]
        rms = jnp.sqrt(jnp.mean(pre**2))
        rms_before = float(rms)
        if not np.isfinite(rms_before) or rms_before <= 0:
            raise ValueError(f"{branch} initialization has zero or nonfinite RMS in hidden layer {index}")
        weights = layer["w"] / rms
        pre = value @ weights.T + layer["b"]
        rms_after = float(jnp.sqrt(jnp.mean(pre**2)))
        if not np.isfinite(np.asarray(weights)).all() or not np.isfinite(rms_after):
            raise ValueError(f"Nonfinite calibrated {branch} initialization in hidden layer {index}")
        calibrated.append({"w": weights, "b": layer["b"]})
        report["layers"].append({"layer": index, "rms_before": rms_before,
                                 "rms_after": rms_after, "weight_multiplier": 1.0 / rms_before})
        value = jax.nn.silu(pre)
    calibrated.append(params[branch][-1])
    return {**params, branch: tuple(calibrated)}, report


def network(layers, x):
    for layer in layers[:-1]:
        x = jax.nn.silu(layer["w"] @ x + layer["b"])
    return layers[-1]["w"] @ x + layers[-1]["b"]


def _bounded_drift_output(raw, cap):
    """Smooth radial bound cap * raw / sqrt(cap**2 + |raw|**2).

    Scaling avoids squaring huge raw outputs. The expression is independent
    of the auxiliary scale, so holding it fixed in autodiff preserves all
    derivatives while avoiding artificial derivatives of max/abs.
    """
    cap = jnp.asarray(cap, dtype=raw.dtype)
    scale = jax.lax.stop_gradient(jnp.maximum(cap, jnp.max(jnp.abs(raw))))
    scaled = raw / scale
    return cap * scaled / jnp.sqrt((cap / scale)**2 + jnp.vdot(scaled, scaled))


def _radial_direction(x):
    """Return x / max(|x|, 1), without overflow or a singularity at zero.

    The tail multiplier is zero throughout |x| <= 1, so this regularization
    preserves the full drift and its first two derivatives. The auxiliary
    scale cancels algebraically and is fixed in autodiff.
    """
    scale = jax.lax.stop_gradient(jnp.maximum(1.0, jnp.max(jnp.abs(x))))
    scaled = x / scale
    norm = jnp.sqrt(jnp.maximum(jnp.vdot(scaled, scaled), (1.0 / scale)**2))
    return scaled / norm


def drift(params, x, cfg):
    chi = cutoff(x)
    # Outside the neural region its value is immaterial. A finite input avoids
    # overflowing hidden activations on distant states in constant_norm mode.
    network_x = (jnp.where(chi > 0, x, jnp.zeros_like(x))
                 if cfg.drift_tail == "constant_norm" else x)
    raw = network(params["drift"], network_x)
    neural = _bounded_drift_output(raw, cfg.drift_cap) if cfg.field_bounds == "smooth" else raw
    if cfg.drift_tail == "constant_norm":
        return chi * neural - (1.0 - chi) * cfg.drift_cap * _radial_direction(x)
    return chi * neural - cfg.tail_attraction * (1.0 - chi) * x


def _diffusion_from_drift(f, cfg):
    """C2 decrease from sigma_cap at zero drift to sigma_min at the threshold.

    The cubic positive part of 1 - |F|^2 / threshold^2 has two continuous
    derivatives where it reaches zero. Limiting each component before
    squaring avoids overflow outside this ball, where the result is already
    constant; it does not change the function or its first two derivatives.
    """
    # New runs use the same cap as F. Retain explicit thresholds from older runs.
    transition = jnp.asarray(cfg.drift_cap if cfg.sigma_drift_scale is None
                             else cfg.sigma_drift_scale, dtype=f.dtype)
    scaled_f = jnp.clip(f, -transition, transition) / transition
    remainder = jnp.maximum(1.0 - jnp.vdot(scaled_f, scaled_f), 0.0)
    return cfg.sigma_min + (cfg.sigma_cap - cfg.sigma_min) * remainder**3


def diffusion(params, x, cfg):
    if cfg.diffusion_mode == "constant":
        return jnp.asarray(cfg.constant_sigma, dtype=x.dtype)
    if cfg.diffusion_mode == "drift_norm":
        return _diffusion_from_drift(drift(params, x, cfg), cfg)
    raw = network(params["diffusion"], x)[0]
    if cfg.field_bounds == "smooth":
        amplitude = (cfg.sigma_cap - cfg.sigma_min) * jax.nn.sigmoid(raw)
    else:
        amplitude = jax.nn.softplus(raw)
    return cfg.sigma_min + cutoff(x) * amplitude


def fields(params, x, cfg):
    f = drift(params, x, cfg)
    sig = (_diffusion_from_drift(f, cfg) if cfg.diffusion_mode == "drift_norm"
           else diffusion(params, x, cfg))
    return f, sig


def parameter_count(params):
    return sum(leaf.size for leaf in jax.tree.leaves(params))

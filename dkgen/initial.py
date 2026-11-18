"""Initial Gaussian parameters and their likelihood-ratio response scores."""

import math

import jax.numpy as jnp
import numpy as np


def make_initial_params(cfg):
    """Initialize N(mean, s0**2 I) without consuming a random key.

    ``s0`` is the per-coordinate standard deviation. The older fixed-law
    setting specifies its RMS radius, which is sqrt(dim) times ``s0``.
    """
    cfg.validate()
    dtype = jnp.dtype(cfg.dtype)
    with np.errstate(over="ignore"):
        scale = np.asarray(max(cfg.initial_rms_radius / math.sqrt(cfg.dim),
                               cfg.initial_s0_min), dtype=np.dtype(cfg.dtype))
    if not np.isfinite(scale):
        raise ValueError("Initial s0 cannot be represented as finite in the configured dtype")
    # Rounding a user-specified floor down would violate the same bound used
    # by the optimizer and checkpoint validation.
    if float(scale) < cfg.initial_s0_min:
        with np.errstate(over="ignore"):
            scale = np.nextafter(scale, np.asarray(np.inf, dtype=scale.dtype))
        if not np.isfinite(scale):
            raise ValueError("Initial s0 floor cannot be represented as finite in the configured dtype")
    return {
        "mean": jnp.zeros(cfg.dim, dtype=dtype),
        "s0": jnp.asarray(scale, dtype=dtype),
    }


def add_initial_params(params, cfg):
    """Add a trainable initial Gaussian when enabled, preserving saved values."""
    if cfg.train_initial and "initial" not in params:
        return {**params, "initial": make_initial_params(cfg)}
    return params


def initial_distribution(params, cfg, dtype=None):
    """Return mean and per-coordinate scale, including the legacy fixed law.

    The parameter branch controls the sampled law even when its optimization
    is disabled. Missing branches retain the older fixed initial distribution.
    """
    dtype = jnp.dtype(cfg.dtype) if dtype is None else dtype
    if params is not None and "initial" in params:
        return (jnp.asarray(params["initial"]["mean"], dtype=dtype),
                jnp.asarray(params["initial"]["s0"], dtype=dtype))
    return (jnp.zeros(cfg.dim, dtype=dtype),
            jnp.asarray(cfg.initial_rms_radius / math.sqrt(cfg.dim), dtype=dtype))


def initial_parameter_scores(x, params, cfg):
    """Parameter scores of the initial Gaussian at fixed sampled positions.

    These are derivatives of log h0(x), not derivatives through x. They enter
    the accumulated path response once at time zero.
    """
    mean, s0 = initial_distribution(params, cfg, x.dtype)
    centered = x - mean
    scaled = centered / s0
    return {
        "mean": centered / s0**2,
        "s0": (jnp.sum(scaled**2, axis=-1) - cfg.dim) / s0,
    }

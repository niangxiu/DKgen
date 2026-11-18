"""Adam, moving-average parameters, and portable state for the score baseline."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .checkpoint import _atomic_write, _validate_selection_score
from .score_learning_rate import learning_rate_at_step


SCHEMA_VERSION = 1
MODEL_KIND = "score_vp"


def init_optimizer(params):
    """Initialize Adam's step count and first and second moments."""
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    # Match the compiled step's output dtype to avoid recompiling update two.
    return {"step": jnp.asarray(0, dtype=jnp.int32), "m": zeros, "v": zeros}


def _matching_trees(params, other):
    if jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(other):
        raise ValueError("Score parameter, gradient, and moment trees must match")
    for value, counterpart in zip(jax.tree_util.tree_leaves(params),
                                  jax.tree_util.tree_leaves(other)):
        if (value.shape != counterpart.shape or value.dtype != counterpart.dtype
                or not jnp.issubdtype(value.dtype, jnp.floating)):
            raise ValueError("Score parameter, gradient, and moment shapes and dtypes must match")


def adam_update(params, gradient, state, cfg):
    """Apply global gradient clipping, followed by bias-corrected Adam.

    This numerical operation can be JIT compiled. The caller checks the returned
    diagnostics and parameters for finiteness before accepting the update.
    """
    for value, name in ((cfg.learning_rate, "learning_rate"),
                        (cfg.adam_epsilon, "adam_epsilon")):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for value, name in ((cfg.adam_beta1, "adam_beta1"),
                        (cfg.adam_beta2, "adam_beta2")):
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{name} must be in [0, 1)")
    cap = cfg.max_gradient_norm
    if not math.isfinite(cap) or cap < 0:
        raise ValueError("max_gradient_norm must be finite and nonnegative")
    if set(state) != {"step", "m", "v"}:
        raise ValueError("Adam state must contain step, m, and v")
    for other in (gradient, state["m"], state["v"]):
        _matching_trees(params, other)
    leaves = jax.tree_util.tree_leaves(gradient)
    if not leaves:
        raise ValueError("Score parameters cannot be empty")
    # Normalize before squaring to avoid overflow for large finite gradients.
    largest = jnp.max(jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in leaves]))
    denominator = jnp.where(largest > 0, largest, 1)
    norm = largest * jnp.sqrt(sum(jnp.sum(jnp.square(leaf / denominator))
                                 for leaf in leaves))
    clip_scale = (jnp.where(norm > cap, cap / jnp.where(norm > 0, norm, 1.0), 1.0)
                  if cap > 0 else jnp.ones_like(norm))
    clipped = jax.tree_util.tree_map(lambda grad: grad * clip_scale, gradient)
    step = jnp.asarray(state["step"], dtype=jnp.int32) + 1
    learning_rate = learning_rate_at_step(cfg, step, dtype=leaves[0].dtype)
    m = jax.tree_util.tree_map(
        lambda old, grad: cfg.adam_beta1 * old + (1 - cfg.adam_beta1) * grad,
        state["m"], clipped)
    v = jax.tree_util.tree_map(
        lambda old, grad: cfg.adam_beta2 * old + (1 - cfg.adam_beta2) * jnp.square(grad),
        state["v"], clipped)
    candidate = jax.tree_util.tree_map(
        lambda value, first, second: value - learning_rate
        * (first / (1 - cfg.adam_beta1 ** step))
        / (jnp.sqrt(second / (1 - cfg.adam_beta2 ** step)) + cfg.adam_epsilon),
        params, m, v)
    return candidate, {"step": step, "m": m, "v": v}, {
        "gradient_norm": norm, "gradient_clip_scale": clip_scale,
        "learning_rate": jnp.asarray(learning_rate, dtype=leaves[0].dtype),
    }


def update_ema(ema_params, params, decay):
    """Update the exponential moving average used to generate samples."""
    if not math.isfinite(decay) or not 0 <= decay < 1:
        raise ValueError("EMA decay must be in [0, 1)")
    _matching_trees(params, ema_params)
    if decay == 0:
        return params
    return jax.tree_util.tree_map(
        lambda average, value: decay * average + (1 - decay) * value,
        ema_params, params)


def make_training_step(engine, cfg):
    """Compile the denoising gradient, Adam, EMA and finite checks together.

    The caller supplies one update key and accepts the returned state only if
    both finite flags are true. Parameters and optimizer state remain on device;
    only flags, loss and scalar diagnostics need to be copied to the host.
    """
    def finite(tree):
        return jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(value)) for value in jax.tree_util.tree_leaves(tree)]))

    @jax.jit
    def step(params, optimizer, ema_params, update_key, train):
        loss, gradient = engine.loss_and_gradient(params, update_key, train)
        gradient_finite = finite((loss, gradient))
        candidate, next_optimizer, diagnostics = adam_update(params, gradient, optimizer, cfg)
        next_ema = update_ema(ema_params, candidate, cfg.ema_decay)
        state_finite = finite((candidate, next_optimizer, next_ema, diagnostics))
        return (candidate, next_optimizer, next_ema, loss, diagnostics,
                gradient_finite, state_finite)

    return step


def _integer(value, name):
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer")
    return int(array)


def _layout(config):
    if not isinstance(config, dict):
        raise ValueError("Score checkpoint config must be an object")
    dimensions = {}
    for name in ("dim", "width", "depth", "time_features"):
        if name not in config:
            raise ValueError(f"Score checkpoint config is missing {name}")
        dimensions[name] = _integer(config[name], name)
        lower = 1 if name in ("dim", "width") else 0
        if dimensions[name] < lower:
            raise ValueError(f"Invalid score checkpoint {name}")
    try:
        dtype = np.dtype(config["dtype"])
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid score checkpoint dtype") from error
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Score checkpoint dtype must be float32 or float64")
    sizes = ([dimensions["dim"] + dimensions["time_features"]]
             + [dimensions["width"]] * dimensions["depth"] + [dimensions["dim"]])
    return sizes, dtype


def _network_arrays(params, config, prefix, *, nonnegative=False):
    sizes, dtype = _layout(config)
    if not isinstance(params, (list, tuple)) or len(params) != len(sizes) - 1:
        raise ValueError(f"Wrong layer count in score checkpoint {prefix}")
    arrays = {}
    for index, layer in enumerate(params):
        if not isinstance(layer, dict) or set(layer) != {"w", "b"}:
            raise ValueError(f"Unexpected layer entries in {prefix}.{index}")
        for tensor, shape in (("w", (sizes[index + 1], sizes[index])),
                              ("b", (sizes[index + 1],))):
            name = f"{prefix}.{index}.{tensor}"
            array = np.asarray(layer[tensor])
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"Wrong shape or dtype in {name}")
            if not np.isfinite(array).all():
                raise ValueError(f"Nonfinite value in {name}")
            if nonnegative and np.any(array < 0):
                raise ValueError(f"Negative second moment in {name}")
            arrays[name] = array
    return arrays


def _selection_state(metadata):
    iteration = _integer(metadata["iteration"], "iteration")
    best_iteration = _integer(metadata["best_iteration"], "best_iteration")
    role = metadata["role"]
    if (role not in {"best", "latest"} or not 0 <= best_iteration <= iteration
            or (role == "best" and best_iteration != iteration)):
        raise ValueError("Invalid score checkpoint selection state")
    pending = metadata["pending_validation"]
    if not isinstance(pending, bool) or (role == "best" and pending):
        raise ValueError("Only latest checkpoints may have pending validation")
    score = metadata["best_validation"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("Checkpoint requires a finite validation score")
    _validate_selection_score(metadata["selection_metric"], score)
    if not isinstance(metadata["extra"], dict):
        raise ValueError("Score checkpoint extra must be an object")
    if not isinstance(metadata["data_fingerprint"], str):
        raise ValueError("Score checkpoint data fingerprint must be a string")


def save_score_checkpoint(
    path: Path | str, *, params, ema_params, optimizer, key, iteration: int,
    best_iteration: int, best_validation: float, config, data_hash: str, role: str,
    pending_validation: bool = False, extra: dict | None = None,
    selection_metric: str = "validation_sliced_wasserstein1",
):
    """Atomically save raw/EMA weights, Adam moments, random stream, and metadata."""
    config = asdict(config) if is_dataclass(config) else dict(config)
    iteration = _integer(iteration, "iteration")
    best_iteration = _integer(best_iteration, "best_iteration")
    metadata = {
        "schema_version": SCHEMA_VERSION, "model_kind": MODEL_KIND,
        "config": config, "iteration": iteration, "best_iteration": best_iteration,
        "best_validation": float(best_validation), "role": role,
        "selection_metric": selection_metric, "pending_validation": pending_validation,
        "data_fingerprint": data_hash, "extra": {} if extra is None else extra,
    }
    _selection_state(metadata)
    if not isinstance(optimizer, dict) or set(optimizer) != {"step", "m", "v"}:
        raise ValueError("Adam state must contain step, m, and v")
    step = _integer(optimizer["step"], "optimizer step")
    if step != iteration:
        raise ValueError("Adam step must equal checkpoint iteration")
    arrays = _network_arrays(params, config, "params")
    arrays.update(_network_arrays(ema_params, config, "ema"))
    arrays.update(_network_arrays(optimizer["m"], config, "optimizer.m"))
    arrays.update(_network_arrays(optimizer["v"], config, "optimizer.v", nonnegative=True))
    key = np.asarray(key)
    if key.shape != (2,) or key.dtype != np.dtype("uint32"):
        raise ValueError("Expected a legacy uint32 JAX PRNGKey with shape (2,)")
    arrays["key"] = key
    arrays["optimizer.step"] = np.asarray(step, dtype=np.int64)
    encoded = json.dumps(metadata, sort_keys=True, allow_nan=False).encode("utf-8")
    arrays["metadata_json"] = np.frombuffer(encoded, dtype=np.uint8)
    _atomic_write(Path(path), lambda stream: np.savez_compressed(stream, **arrays))


def load_score_checkpoint(path: Path | str):
    """Load and validate a score checkpoint without executing pickled objects."""
    with np.load(path, allow_pickle=False) as archive:
        encoded = archive["metadata_json"]
        if encoded.dtype != np.uint8 or encoded.ndim != 1:
            raise ValueError("Malformed score checkpoint metadata")
        metadata = json.loads(encoded.tobytes().decode("utf-8"))
        if not isinstance(metadata, dict) or metadata.get("model_kind") != MODEL_KIND:
            raise ValueError("Not a score_vp checkpoint")
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported score checkpoint schema")
        _selection_state(metadata)
        config = metadata["config"]
        sizes, _ = _layout(config)
        # Older score evaluations always included half the query's own peak.
        config.setdefault("density_self_weight", 0.5)
        # Historical score checkpoints trained with uniform time sampling.
        config.setdefault("time_sampling_power", 1.0)
        networks = {}
        arrays = {}
        for prefix in ("params", "ema", "optimizer.m", "optimizer.v"):
            networks[prefix] = tuple(
                {tensor: archive[f"{prefix}.{index}.{tensor}"] for tensor in ("w", "b")}
                for index in range(len(sizes) - 1))
            arrays.update(_network_arrays(networks[prefix], config, prefix,
                                           nonnegative=prefix == "optimizer.v"))
        if set(archive.files) != set(arrays) | {"key", "metadata_json", "optimizer.step"}:
            raise ValueError("Unexpected entries in score checkpoint")
        key = archive["key"]
        if key.shape != (2,) or key.dtype != np.dtype("uint32"):
            raise ValueError("Malformed score checkpoint PRNG key")
        step = _integer(archive["optimizer.step"], "optimizer step")
        if step != metadata["iteration"]:
            raise ValueError("Adam step must equal checkpoint iteration")
    return {"params": networks["params"], "ema_params": networks["ema"],
            "optimizer": {"step": step, "m": networks["optimizer.m"], "v": networks["optimizer.v"]},
            "key": key, "metadata": metadata}

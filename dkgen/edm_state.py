"""Portable checkpoints for the dense EDM benchmark, separate from VP models."""

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

import numpy as np

from .checkpoint import _atomic_write, _validate_selection_score
from .edm_config import EDMConfig


def _integer(value, name):
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu" or int(array) < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(array)


def _metadata(metadata):
    if not isinstance(metadata, dict) or metadata.get("model_kind") != "edm":
        raise ValueError("Not an EDM checkpoint")
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported EDM checkpoint schema")
    cfg = EDMConfig(**metadata["config"]).validate()
    iteration = _integer(metadata["iteration"], "iteration")
    best = _integer(metadata["best_iteration"], "best_iteration")
    role = metadata["role"]
    if role not in ("best", "latest") or best > iteration or (role == "best" and best != iteration):
        raise ValueError("Invalid EDM checkpoint selection state")
    # Historical EDM checkpoints recorded only KL selection and had no pending flag.
    metadata.setdefault("pending_validation", False)
    metadata.setdefault("selection_metric", "validation_averaged_kl")
    pending = metadata["pending_validation"]
    if not isinstance(pending, bool) or (role == "best" and pending):
        raise ValueError("Only latest checkpoints may have pending validation")
    score = metadata["best_validation"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not np.isfinite(score):
        raise ValueError("EDM checkpoint requires a finite validation score")
    _validate_selection_score(metadata["selection_metric"], score)
    if not isinstance(metadata["data_fingerprint"], str) or not isinstance(metadata["extra"], dict):
        raise ValueError("Invalid EDM checkpoint provenance")
    return cfg


def _network_arrays(params, cfg, prefix, *, nonnegative=False):
    from .edm_model import parameter_shapes
    shapes = parameter_shapes(cfg)
    if not isinstance(params, dict) or set(params) != set(shapes):
        raise ValueError(f"Unexpected EDM network tensors: {prefix}")
    arrays = {}
    for name, shape in shapes.items():
        array = np.asarray(params[name])
        if array.shape != shape or array.dtype != np.dtype(cfg.dtype):
            raise ValueError(f"Wrong EDM tensor shape or dtype: {prefix}.{name}")
        if not np.isfinite(array).all() or (nonnegative and np.any(array < 0)):
            raise ValueError(f"Invalid EDM tensor values: {prefix}.{name}")
        arrays[f"{prefix}.{name}"] = array
    return arrays


def save_edm_checkpoint(path, *, params, ema_params, optimizer, key, iteration,
                        best_iteration, best_validation, config, data_hash, role,
                        pending_validation=False, extra=None,
                        selection_metric="validation_averaged_kl"):
    """Save raw/EMA weights, Adam state, scaling, and validation selection state."""
    metadata = {
        "schema_version": 1, "model_kind": "edm",
        "config": asdict(config) if is_dataclass(config) else dict(config),
        "iteration": _integer(iteration, "iteration"),
        "best_iteration": _integer(best_iteration, "best_iteration"),
        "best_validation": best_validation, "role": role,
        "selection_metric": selection_metric, "pending_validation": pending_validation,
        "data_fingerprint": data_hash, "extra": {} if extra is None else extra,
    }
    cfg = _metadata(metadata)
    if not isinstance(optimizer, dict) or set(optimizer) != {"step", "m", "v"}:
        raise ValueError("Adam state must contain step, m, and v")
    step = _integer(optimizer["step"], "optimizer step")
    if step != metadata["iteration"]:
        raise ValueError("Adam step must equal checkpoint iteration")
    arrays = {}
    for prefix, network in (("params", params), ("ema", ema_params),
                            ("optimizer.m", optimizer["m"]), ("optimizer.v", optimizer["v"])):
        arrays.update(_network_arrays(network, cfg, prefix, nonnegative=prefix == "optimizer.v"))
    key = np.asarray(key)
    if key.shape != (2,) or key.dtype != np.dtype("uint32"):
        raise ValueError("Expected a uint32 JAX PRNGKey with shape (2,)")
    arrays["key"] = key
    arrays["optimizer.step"] = np.asarray(step, dtype=np.int64)
    arrays["metadata_json"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True, allow_nan=False).encode("utf-8"), dtype=np.uint8)
    _atomic_write(Path(path), lambda stream: np.savez_compressed(stream, **arrays))


def load_edm_checkpoint(path):
    """Validate and load numeric EDM state without pickle."""
    from .edm_model import parameter_shapes
    with np.load(path, allow_pickle=False) as archive:
        encoded = archive["metadata_json"]
        if encoded.dtype != np.uint8 or encoded.ndim != 1:
            raise ValueError("Malformed EDM checkpoint metadata")
        metadata = json.loads(encoded.tobytes().decode("utf-8"))
        cfg = _metadata(metadata)
        networks, arrays = {}, {}
        for prefix in ("params", "ema", "optimizer.m", "optimizer.v"):
            network = {name: archive[f"{prefix}.{name}"] for name in parameter_shapes(cfg)}
            arrays.update(_network_arrays(network, cfg, prefix, nonnegative=prefix == "optimizer.v"))
            networks[prefix] = network
        if set(archive.files) != set(arrays) | {"metadata_json", "key", "optimizer.step"}:
            raise ValueError("Unexpected entries in EDM checkpoint")
        key = archive["key"]
        if key.shape != (2,) or key.dtype != np.dtype("uint32"):
            raise ValueError("Malformed EDM checkpoint PRNG key")
        step = _integer(archive["optimizer.step"], "optimizer step")
        if step != metadata["iteration"]:
            raise ValueError("Adam step must equal checkpoint iteration")
    return {"params": networks["params"], "ema_params": networks["ema"],
            "optimizer": {"step": step, "m": networks["optimizer.m"], "v": networks["optimizer.v"]},
            "key": key, "metadata": metadata}

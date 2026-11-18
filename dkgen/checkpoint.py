"""Portable checkpoints: numerical NPZ arrays and JSON, never pickle."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .file_io import replace_with_retry
from .schedule import ScheduleSettings, validate_schedule_state


SCHEMA_VERSION = 2
DATA_KEYS = ("train", "val", "test", "train_raw", "val_raw", "test_raw", "R")


def data_fingerprint(data: dict) -> str:
    """Hash saved splits, scale, and optional target metadata, including dtype."""
    digest = hashlib.sha256()
    names = DATA_KEYS + (("mixture_modes",) if "mixture_modes" in data else ())
    for name in names:
        dtype = np.int64 if name == "mixture_modes" else None
        array = np.ascontiguousarray(np.asarray(data[name], dtype=dtype))
        if array.dtype.hasobject:
            raise ValueError(f"Object array is not permitted in data: {name}")
        descriptor = json.dumps([name, array.dtype.str, list(array.shape)])
        digest.update(descriptor.encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path | str, value: dict) -> None:
    """Replace a JSON artifact only after its complete contents are durable."""
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write(path, lambda stream: stream.write(payload.encode("utf-8")))


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _parameter_arrays(params: dict, config: dict, *, moments=False) -> dict[str, np.ndarray]:
    if (not isinstance(params, dict) or not {"drift", "diffusion"}.issubset(params)
            or set(params) - {"drift", "diffusion", "initial"}):
        raise ValueError("Checkpoint parameters must contain drift and diffusion")
    train_initial = config.get("train_initial", False)
    if not isinstance(train_initial, bool):
        raise ValueError("Checkpoint train_initial must be boolean")
    if train_initial and "initial" not in params:
        raise ValueError("Checkpoint is missing trainable initial parameters")
    diffusion_mode = config.get("diffusion_mode", "neural")
    if diffusion_mode not in ("constant", "neural", "drift_norm"):
        raise ValueError("Checkpoint diffusion_mode must be constant, neural, or drift_norm")
    expected_dtype = np.dtype(config["dtype"])
    if expected_dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Checkpoint dtype must be float32 or float64")
    arrays = {}
    for branch, output_dim in (("drift", config["dim"]), ("diffusion", 1)):
        layers = params[branch]
        if branch == "diffusion" and diffusion_mode in ("constant", "drift_norm"):
            if len(layers) != 0:
                raise ValueError(f"{diffusion_mode} diffusion checkpoint must have no diffusion parameters")
            continue
        width = config.get("diffusion_width", config["width"]) if branch == "diffusion" else config["width"]
        depth = config.get("diffusion_depth", config["depth"]) if branch == "diffusion" else config["depth"]
        sizes = [config["dim"]] + [width] * depth + [output_dim]
        if len(layers) != len(sizes) - 1:
            raise ValueError(f"Wrong layer count in checkpoint {branch}")
        for index, layer in enumerate(layers):
            if set(layer) != {"w", "b"}:
                raise ValueError(f"Unexpected layer entries in {branch}.{index}")
            for tensor, shape in (("w", (sizes[index + 1], sizes[index])),
                                  ("b", (sizes[index + 1],))):
                array = np.asarray(layer[tensor])
                if array.shape != shape or array.dtype != expected_dtype:
                    raise ValueError(f"Wrong shape or dtype in {branch}.{index}.{tensor}")
                if not np.isfinite(array).all():
                    raise ValueError(f"Nonfinite parameter in {branch}.{index}.{tensor}")
                arrays[f"{branch}.{index}.{tensor}"] = array
    if "initial" in params:
        initial = params["initial"]
        if not isinstance(initial, dict) or set(initial) != {"mean", "s0"}:
            raise ValueError("Checkpoint initial parameters must contain mean and s0")
        for name, shape in (("mean", (config["dim"],)), ("s0", ())):
            array = np.asarray(initial[name])
            if array.shape != shape or array.dtype != expected_dtype:
                raise ValueError(f"Wrong shape or dtype in initial.{name}")
            if not np.isfinite(array).all():
                raise ValueError(f"Nonfinite parameter in initial.{name}")
            arrays[f"initial.{name}"] = array
        if not moments:
            floor = config.get("initial_s0_min", 0.2)
            if (isinstance(floor, bool) or not isinstance(floor, (int, float))
                    or not math.isfinite(floor) or floor <= 0):
                raise ValueError("Checkpoint initial_s0_min must be finite and positive")
            if float(initial["s0"]) < floor:
                raise ValueError("Checkpoint initial.s0 must be at least initial_s0_min")
    return arrays


def _validate_alpha_state(config: dict, extra: dict) -> None:
    """A saved lagged mean belongs to the same update as the parameters/key."""
    if config.get("alpha_mode") != "sigma" or config.get("alpha_sigma_target") is None:
        return
    if not isinstance(extra, dict) or "previous_sigma_mean" not in extra:
        raise ValueError("Checkpoint is missing previous_sigma_mean for normalized alpha")
    value = extra["previous_sigma_mean"]
    # None explicitly marks the first update with this rule, including opt-in
    # when resuming an older run. Do not silently replace a missing value.
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value) or value <= 0):
        raise ValueError("Checkpoint previous_sigma_mean must be None or finite and positive")


def _validate_response_state(config: dict, extra: dict) -> None:
    """The lagged coefficient mean is part of the accepted optimization state."""
    if config.get("response_weight_mode", "kde") not in ("count_squared", "count_log", "count_only", "three_term", "kde"):
        raise ValueError("Checkpoint response_weight_mode must be count_squared, count_log, count_only, three_term, or kde")
    mode = config.get("response_baseline", "none")
    if mode not in ("none", "previous_mean"):
        raise ValueError("Checkpoint response_baseline must be none or previous_mean")
    if config.get("loss_mode", "forward_kl") == "forward_kl" or mode != "previous_mean":
        return
    if not isinstance(extra, dict) or "previous_response_mean" not in extra:
        raise ValueError("Checkpoint is missing previous_response_mean for the response baseline")
    value = extra["previous_response_mean"]
    # None explicitly bootstraps with zero. A signed mean is valid, including
    # zero; missing state must not silently restart an existing baseline.
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value)):
        raise ValueError("Checkpoint previous_response_mean must be None or a finite scalar")


def _validate_selection_score(selection_metric: str, score: float) -> None:
    if selection_metric not in ("validation_sliced_wasserstein1", "validation_averaged_kl"):
        raise ValueError("Unknown checkpoint selection_metric")
    if not math.isfinite(score):
        raise ValueError("Checkpoint requires a finite validation score")
    # The finite-sample KL estimate may be negative; a distance cannot be.
    if selection_metric == "validation_sliced_wasserstein1" and score < 0:
        raise ValueError("Checkpoint requires a nonnegative validation distance")


def _integer(value, name: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer")
    return int(array)


def _optimizer_arrays(optimizer_state, config: dict, iteration: int, params=None) -> dict:
    """Validate portable Adam state without downcasting its host float64 moments."""
    optimizer = config.get("optimizer", "sgd")
    if optimizer not in ("sgd", "adam"):
        raise ValueError("Checkpoint optimizer must be sgd or adam")
    if optimizer == "sgd":
        if optimizer_state is not None:
            raise ValueError("SGD checkpoints must not contain optimizer state")
        return {}
    if not isinstance(optimizer_state, dict) or set(optimizer_state) != {"step", "m", "v"}:
        raise ValueError("Adam optimizer state must contain step, m, and v")
    step = _integer(optimizer_state["step"], "Adam step")
    if step != iteration:
        raise ValueError("Adam step must equal checkpoint iteration")
    arrays = {"optimizer.step": np.asarray(step, dtype=np.int64)}
    # Float64 moments avoid overflow when finite float32 DK gradients are squared.
    moment_config = dict(config, dtype="float64")
    for moment in ("m", "v"):
        try:
            moment_arrays = _parameter_arrays(optimizer_state[moment], moment_config, moments=True)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid Adam {moment} moments: {error}") from error
        if params is not None and set(moment_arrays) != set(_parameter_arrays(params, config)):
            raise ValueError("Adam moments must match parameter branches")
        for name, array in moment_arrays.items():
            if moment == "v" and np.any(array < 0):
                raise ValueError(f"Negative Adam second moment in {name}")
            arrays[f"optimizer.{moment}.{name}"] = array
    return arrays


def _load_optimizer_state(archive, params: dict, config: dict, iteration: int):
    if config["optimizer"] != "adam":
        return None
    expected = {"optimizer.step"}
    for moment in ("m", "v"):
        for branch, layers in params.items():
            if branch == "initial":
                expected.update(f"optimizer.{moment}.initial.{name}" for name in layers)
                continue
            for index, layer in enumerate(layers):
                expected.update(f"optimizer.{moment}.{branch}.{index}.{name}" for name in layer)
    if not expected.issubset(archive.files):
        raise ValueError("Checkpoint is missing Adam optimizer state")
    state = {"step": _integer(archive["optimizer.step"], "Adam step")}
    for moment in ("m", "v"):
        state[moment] = {
            branch: tuple({name: archive[f"optimizer.{moment}.{branch}.{index}.{name}"]
                           for name in layer}
                          for index, layer in enumerate(layers))
            for branch, layers in params.items() if branch != "initial"
        }
        if "initial" in params:
            state[moment]["initial"] = {
                name: archive[f"optimizer.{moment}.initial.{name}"] for name in params["initial"]}
    return state


def save_checkpoint(
    path: Path | str, *, params: dict, key, iteration: int,
    best_validation: float, config, data_hash: str, role: str,
    best_iteration: int, pending_validation: bool = False, extra: dict | None = None,
    selection_metric: str = "validation_sliced_wasserstein1",
    optimizer_state: dict | None = None,
) -> None:
    """Atomically save the SGD or Adam state and provenance in one NPZ file."""
    config = asdict(config) if is_dataclass(config) else dict(config)
    iteration = _integer(iteration, "iteration")
    best_iteration = _integer(best_iteration, "best_iteration")
    validate_schedule_state(config, iteration)
    _validate_alpha_state(config, extra)
    _validate_response_state(config, extra)
    arrays = _parameter_arrays(params, config)
    arrays.update(_optimizer_arrays(optimizer_state, config, iteration, params))
    key = np.asarray(key)
    if key.shape != (2,) or key.dtype != np.dtype("uint32"):
        raise ValueError("Expected a legacy uint32 JAX PRNGKey with shape (2,)")
    if role not in {"best", "latest"}:
        raise ValueError("Checkpoint role must be best or latest")
    if iteration < 0 or not 0 <= best_iteration <= iteration:
        raise ValueError("Invalid checkpoint iteration")
    if role == "best" and best_iteration != iteration:
        raise ValueError("A best checkpoint must store its selected iteration")
    if not isinstance(pending_validation, bool) or (role == "best" and pending_validation):
        raise ValueError("Only latest checkpoints may have pending validation")
    _validate_selection_score(selection_metric, best_validation)
    metadata = {
        "schema_version": SCHEMA_VERSION, "config": config,
        "iteration": int(iteration), "best_iteration": int(best_iteration),
        "best_validation": float(best_validation), "data_fingerprint": data_hash,
        "role": role, "selection_metric": selection_metric,
        "pending_validation": pending_validation,
        "extra": extra or {},
    }
    encoded = json.dumps(metadata, sort_keys=True, allow_nan=False).encode("utf-8")
    arrays["metadata_json"] = np.frombuffer(encoded, dtype=np.uint8)
    arrays["key"] = key
    _atomic_write(Path(path), lambda stream: np.savez_compressed(stream, **arrays))


def load_checkpoint(path: Path | str) -> dict:
    """Load a checkpoint, reject unsafe arrays, and verify its parameter layout."""
    with np.load(path, allow_pickle=False) as archive:
        encoded = archive["metadata_json"]
        if encoded.dtype != np.uint8 or encoded.ndim != 1:
            raise ValueError("Malformed checkpoint metadata")
        metadata = json.loads(encoded.tobytes().decode("utf-8"))
        if metadata.get("schema_version") not in (1, SCHEMA_VERSION):
            raise ValueError("Unsupported checkpoint schema")
        config = metadata["config"]
        # Earlier runs used a fixed centered Gaussian, including their saved scale.
        config.setdefault("train_initial", False)
        config.setdefault("initial_s0_min", 0.2)
        # Before independent network sizes, both fields used these saved dimensions.
        config.setdefault("diffusion_width", config["width"])
        config.setdefault("diffusion_depth", config["depth"])
        # Pre-Adam checkpoints retain their original SGD updates and learning rate.
        config.setdefault("optimizer", "sgd")
        config.setdefault("adam_beta1", 0.9)
        config.setdefault("adam_beta2", 0.999)
        config.setdefault("adam_epsilon", 1e-8)
        # Existing checkpoints used forward KL with assignment-count weights.
        config.setdefault("loss_mode", "forward_kl")
        # Existing averaged-KL checkpoints retain their two-term KDE response.
        config.setdefault("response_weight_mode", "kde")
        # Existing checkpoints retain their uncentered averaged-KL response.
        config.setdefault("response_baseline", "none")
        config.setdefault("n_neighbor", 32)
        config.setdefault("density_bandwidth", 0.1)
        config.setdefault("density_self_weight", 0.5)
        # The schedule module is the sole source of defaults, including the
        # serialized recipe endpoints added after the original checkpoints.
        # A checkpoint with no schedule tag predates scheduled optimization.
        config.setdefault("schedule", "none")
        for name, value in asdict(ScheduleSettings()).items():
            config.setdefault(name, value)
        # Existing checkpoints always used a trainable diffusion network.
        config.setdefault("diffusion_mode", "neural")
        config.setdefault("constant_sigma", 0.5)
        # These settings do not change the saved diffusion or transfer-rate modes.
        config.setdefault("sigma_drift_scale", 4.0)
        config.setdefault("alpha_sigma_multiplier", 10.0)
        # Existing runs keep their fixed sigma multiplier unless explicitly changed.
        config.setdefault("alpha_sigma_target", None)
        config.setdefault("alpha_cap", 40.0)
        # Before this setting was configurable, every run used RMS radius 0.5.
        config.setdefault("initial_rms_radius", 0.5)
        # Older diffusion networks started with a zero raw output.
        config.setdefault("initial_sigma", config["sigma_min"] + math.log(2.0))
        # Older runs used the same learning rate for both networks.
        config.setdefault("drift_learning_rate_multiplier", 1.0)
        # Older runs did not calibrate hidden drift layers using training data.
        config.setdefault("drift_initialization", "glorot")
        # Diffusion calibration was added after drift calibration.
        config.setdefault("diffusion_initialization", "glorot")
        # Preserve the original coefficient formulas for existing checkpoints.
        config.setdefault("field_bounds", "none")
        config.setdefault("drift_tail", "linear")
        config.setdefault("drift_cap", 2.0)
        config.setdefault("sigma_cap", 1.0)
        # All earlier datasets and region metrics used the eight-mode target.
        config.setdefault("mixture_modes", 8)
        # Earlier runs used the pilot-dependent transfer rate and unclipped SGD.
        config.setdefault("alpha_mode", "adaptive")
        config.setdefault("alpha", 4.0)
        config.setdefault("max_update_norm", 0.0)
        # Existing runs retain their norm cap; KL clipping applies to new runs.
        config.setdefault("update_clip_mode", "norm")
        config.setdefault("kl_clip_fraction", 1.0 / 3.0)
        # Earlier gradient estimates used uncentered nearest-neighbor weights.
        config.setdefault("score_baseline", 0.0)
        # Earlier runs did not penalize either network's weight matrices.
        config.setdefault("drift_weight_decay", 0.0)
        config.setdefault("diffusion_weight_decay", 0.0)
        # Absorb the removed shared multiplier into the saved coefficients.
        # This preserves the effective penalty for checkpoints using that knob.
        if "regularization_multiplier" in config:
            multiplier = config.pop("regularization_multiplier")
            if (isinstance(multiplier, bool) or not isinstance(multiplier, (int, float))
                    or not math.isfinite(multiplier) or multiplier < 0):
                raise ValueError("Invalid saved regularization_multiplier")
            for name in ("drift_weight_decay", "diffusion_weight_decay"):
                decay = config[name]
                if (isinstance(decay, bool) or not isinstance(decay, (int, float))
                        or not math.isfinite(decay) or decay < 0):
                    raise ValueError(f"Invalid saved {name}")
                config[name] = multiplier * decay
                if not math.isfinite(config[name]):
                    raise ValueError(f"Invalid effective {name}")
        # Observational diagnostics may also run when resuming older checkpoints.
        config.setdefault("nu_diagnostic_threshold", 20.0)
        config.setdefault("nu_diagnostic_paths", 100)
        config.setdefault("nu_diagnostic_interval", 50)
        params = {}
        for branch in ("drift", "diffusion"):
            if branch == "diffusion" and config["diffusion_mode"] in ("constant", "drift_norm"):
                params[branch] = ()
                continue
            depth = config["diffusion_depth"] if branch == "diffusion" else config["depth"]
            expected = {f"{branch}.{index}.{tensor}"
                        for index in range(depth + 1) for tensor in ("w", "b")}
            if not expected.issubset(archive.files):
                raise ValueError(f"Checkpoint is missing {branch} parameter tensors")
            params[branch] = tuple({tensor: archive[f"{branch}.{index}.{tensor}"]
                                    for tensor in ("w", "b")}
                                   for index in range(depth + 1))
        initial_names = {"initial.mean", "initial.s0"}
        if initial_names.intersection(archive.files):
            if not initial_names.issubset(archive.files):
                raise ValueError("Checkpoint is missing initial parameter tensors")
            params["initial"] = {name: archive[f"initial.{name}"] for name in ("mean", "s0")}
        arrays = _parameter_arrays(params, config)
        optimizer_state = _load_optimizer_state(archive, params, config, metadata["iteration"])
        arrays.update(_optimizer_arrays(optimizer_state, config, metadata["iteration"], params))
        if set(archive.files) != set(arrays) | {"metadata_json", "key"}:
            raise ValueError("Unexpected entries in checkpoint")
        key = archive["key"]
        if key.shape != (2,) or key.dtype != np.uint32:
            raise ValueError("Malformed checkpoint PRNG key")
    iteration = metadata["iteration"]
    best_iteration = metadata["best_iteration"]
    score = metadata["best_validation"]
    _validate_selection_score(metadata.get("selection_metric"), score)
    # Older schema-1 checkpoints did not record pending validation.
    metadata.setdefault("pending_validation", False)
    pending_validation = metadata["pending_validation"]
    if (not isinstance(iteration, int) or not isinstance(best_iteration, int)
            or not 0 <= best_iteration <= iteration
            or metadata["role"] not in {"best", "latest"}
            or (metadata["role"] == "best" and best_iteration != iteration)
            or not isinstance(pending_validation, bool)
            or (metadata["role"] == "best" and pending_validation)):
        raise ValueError("Invalid checkpoint selection state")
    _validate_alpha_state(config, metadata.get("extra", {}))
    _validate_response_state(config, metadata.get("extra", {}))
    validate_schedule_state(config, iteration)
    return {"params": params, "key": key, "metadata": metadata,
            "optimizer_state": optimizer_state}

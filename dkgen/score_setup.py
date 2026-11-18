"""Shared data and configuration preparation for VP and EDM score models."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path

import numpy as np

from .data import target_metadata, validate_target_metadata
from .checkpoint import data_fingerprint
from .config import Config
from .data import load_data, make_mixture_data
from .edm_config import EDMConfig, matched_config as matched_edm_config
from .score_config import ScoreConfig, matched_config as matched_vp_config
from .schedule import evaluation_bandwidth


_TARGET = "correlated_four_mode_gaussian_mixture"
_DATA_FIELDS = ("dim", "data_seed", "n_train", "n_val", "n_test", "mixture_modes")
_INTERNAL_FIELDS = {"target_parameter_count", "data_scale"}
_CONFIG_TYPES = {"vp": ScoreConfig, "edm": EDMConfig}
_MATCHED_CONFIGS = {"vp": matched_vp_config, "edm": matched_edm_config}


def _json_object(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _numeric_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _rotation_seed(value):
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu" or int(array) < 0:
        raise ValueError("rotation_seed must be a nonnegative scalar integer")
    return int(array)


def _checkpoint_source(args):
    return next((getattr(args, name, None) for name in ("resume", "warm_start", "evaluate")
                 if getattr(args, name, None) is not None), None)


def _checkpoint_metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        encoded = archive["metadata_json"]
        if encoded.dtype != np.uint8 or encoded.ndim != 1:
            raise ValueError("Checkpoint metadata_json must be a byte vector")
        return json.loads(encoded.tobytes().decode("utf-8"))


def _saved_method(saved):
    metadata = saved["metadata"]
    kind = metadata.get("model_kind")
    if kind is not None:
        try:
            return {"score_vp": "vp", "edm": "edm"}[kind]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Unsupported score checkpoint model_kind: {kind!r}") from error
    # Older in-memory callers supplied configuration metadata without its tag.
    return "edm" if "sigma_data" in metadata["config"] else "vp"


def arguments(argv=None, *, default_dim=2, default_method="edm"):
    """Parse explicit settings and infer saved model type before parsing."""
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--method", choices=tuple(_CONFIG_TYPES))
    for name in ("resume", "warm-start", "evaluate"):
        preliminary.add_argument("--" + name, type=Path)
    selected, _ = preliminary.parse_known_args(argv)
    source = _checkpoint_source(selected)
    method = selected.method or default_method
    if source is not None and source.is_file():
        from .score_backend import checkpoint_method
        saved_method = checkpoint_method(source)
        if selected.method is not None and selected.method != saved_method:
            raise ValueError("--method cannot change a saved checkpoint's model")
        method = saved_method
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(_CONFIG_TYPES), default=method,
                        help="New run method (default: edm); saved runs keep their method")
    parser.add_argument("--output", type=Path, help="New folder; default: next results/run-N")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", type=Path, help="Continue a checkpoint in a new run folder")
    mode.add_argument("--warm-start", type=Path,
                      help="Reuse saved weights and data; reset optimizer and update count")
    mode.add_argument("--evaluate", type=Path, help="Evaluate saved weights")
    parser.add_argument("--data", type=Path, help="Saved dataset; otherwise use source run or generate it")
    match = parser.add_mutually_exclusive_group()
    match.add_argument("--match-dk-config", type=Path, help="Match a saved DK configuration")
    match.add_argument("--match-dk-run", type=Path,
                       help="Reuse exact DK data, target, budget, and evaluation settings")
    parser.add_argument("--rotation-seed", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None,
                        help="Set split sizes, generated test count, and training batch size; "
                             "individual settings take precedence")
    defaults = _CONFIG_TYPES[method]()
    for field in fields(defaults):
        if field.name in _INTERNAL_FIELDS:
            continue
        value = getattr(defaults, field.name)
        options = {"default": argparse.SUPPRESS,
                   "help": "Override the selected experiment's configuration"}
        if isinstance(value, bool):
            options["action"] = argparse.BooleanOptionalAction
        else:
            options["type"] = type(value)
        if field.name == "backbone":
            options["choices"] = ("residual", "mlp")
        if field.name == "dtype":
            options["choices"] = ("float32", "float64")
        if field.name == "n_neighbor" and source is None:
            options["choices"] = (0,)
            options["help"] = "0 evaluates KL with all data and generated samples"
        parser.add_argument("--" + field.name.replace("_", "-"), **options)
    args = parser.parse_args(argv)
    args.method = method
    args._method_explicit = selected.method is not None
    args._default_dim = default_dim
    return args


def _dk_configuration(path):
    values = _json_object(path)
    values.setdefault("optimizer", "sgd")
    values.setdefault("update_clip_mode", "norm")
    values.setdefault("train_initial", False)
    values.setdefault("schedule", "none")
    values.setdefault("diffusion_width", values.get("width"))
    values.setdefault("diffusion_depth", values.get("depth"))
    try:
        return Config(**values).validate()
    except TypeError as error:
        raise ValueError("Matching requires a DK configuration") from error


def _saved_evaluation_bandwidth(source, run, cfg):
    records = [run]
    summary = Path(source) / "summary.json"
    if summary.exists():
        records.append(_json_object(summary))
    scheduled = False
    for record in records:
        schedule = record.get("schedule")
        if schedule is None:
            continue
        if not isinstance(schedule, dict):
            raise ValueError("Saved schedule must be an object")
        if "evaluation_density_bandwidth" in schedule:
            return schedule["evaluation_density_bandwidth"]
        scheduled |= schedule.get("name") not in (None, "none")
    if scheduled or cfg.schedule != "none":
        raise ValueError("Scheduled DK run is missing its fixed evaluation bandwidth")
    return cfg.density_bandwidth


def _explicit_overrides(args, method):
    allowed = {field.name for field in fields(_CONFIG_TYPES[method])} - _INTERNAL_FIELDS
    model_fields = ({field.name for field in fields(ScoreConfig)}
                    | {field.name for field in fields(EDMConfig)}) - _INTERNAL_FIELDS
    incompatible = sorted(name for name in model_fields - allowed if hasattr(args, name))
    if incompatible:
        raise ValueError(f"Settings do not apply to {method}: " + ", ".join(incompatible))
    overrides = {name: getattr(args, name) for name in allowed if hasattr(args, name)}
    samples = getattr(args, "samples", None)
    if samples is not None:
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError("samples must be a positive integer")
        for name in ("n_train", "n_val", "n_test", "eval_samples", "batch_size"):
            overrides.setdefault(name, samples)
    return overrides


def _reference_evaluation_bandwidth(args, cfg):
    source = args.match_dk_run
    if source is None and args.match_dk_config is not None:
        source = args.match_dk_config.parent
    if source is not None and (source / "run.json").exists():
        return _saved_evaluation_bandwidth(source, _json_object(source / "run.json"), cfg)
    return evaluation_bandwidth(cfg)


def _configuration(args, saved=None):
    """Resolve new settings or preserve the saved model on continuation."""
    method = getattr(args, "method", "edm")
    if saved is not None:
        saved_method = _saved_method(saved)
        if getattr(args, "_method_explicit", False) and method != saved_method:
            raise ValueError("--method cannot change a saved checkpoint's model")
        method = saved_method
    args.method = method
    overrides = _explicit_overrides(args, method)
    if saved is not None:
        if args.match_dk_config is not None or args.match_dk_run is not None:
            raise ValueError("--match-dk-config and --match-dk-run apply only to a new run")
        cfg = _CONFIG_TYPES[method](**saved["metadata"]["config"]).validate()
        allowed = {"updates", "eval_every", "log_every", "density_bandwidth", "density_self_weight",
                   "sampling_steps", "learning_rate", "learning_rate_schedule", "learning_rate_final",
                   "learning_rate_decay_start", "learning_rate_decay_end"}
        if args.warm_start is not None:
            fixed = set(_DATA_FIELDS) | {"width", "depth", "backbone", "time_features", "time_max_frequency",
                                        "sigma_data", "dtype", "seed"}
            allowed = {field.name for field in fields(cfg)} - fixed - _INTERNAL_FIELDS
        elif args.evaluate is not None:
            allowed = set()
        forbidden = sorted(name for name, value in overrides.items()
                           if value != getattr(cfg, name) and name not in allowed)
        if forbidden:
            raise ValueError("Cannot change saved score configuration: " + ", ".join(forbidden))
        cfg = replace(cfg, **overrides).validate()
        if args.resume is not None and cfg.updates < saved["metadata"]["iteration"]:
            raise ValueError("updates cannot precede the saved checkpoint iteration")
        dk_cfg = None
    else:
        from .train import _default_configuration
        source = args.match_dk_run
        if source is not None:
            dk_cfg = _dk_configuration(source / "config.json")
            mismatches = [name for name in _DATA_FIELDS
                          if name in overrides and overrides[name] != getattr(dk_cfg, name)]
            if mismatches:
                raise ValueError("Saved benchmark dataset mismatch: " + ", ".join(mismatches))
            run = _json_object(source / "run.json")
            bandwidth = _saved_evaluation_bandwidth(source, run, dk_cfg)
        else:
            if args.match_dk_config is not None:
                dk_cfg = _dk_configuration(args.match_dk_config)
            else:
                dim = overrides.get("dim", getattr(args, "_default_dim", 2))
                if "dim" not in overrides and args.data is not None:
                    with np.load(args.data, allow_pickle=False) as data:
                        dim = data["train"].shape[1]
                dk_cfg = _default_configuration(dim)
            changes = {name: value for name, value in overrides.items() if name in _DATA_FIELDS}
            dk_cfg = replace(dk_cfg, **changes).validate()
            bandwidth = _reference_evaluation_bandwidth(args, dk_cfg)
        comparison = {"density_bandwidth": bandwidth, "n_neighbor": 0}
        if method == "vp":
            comparison["time_sampling_power"] = 2.0 if dk_cfg.dim > 2 else 1.0
        comparison.update(overrides)
        if comparison["n_neighbor"] != 0:
            raise ValueError("New score comparisons require n_neighbor=0 (all points)")
        cfg = _MATCHED_CONFIGS[method](dk_cfg, **comparison).validate()
    if cfg.dim < 2 or cfg.dim % 2:
        raise ValueError("Score projection plots require an even dim >= 2")
    if cfg.dim > 2 and cfg.mixture_modes != 4:
        raise ValueError("The dimension-scaling benchmark uses four mixture modes")
    return cfg, overrides, dk_cfg


def _adopt_data_sizes(cfg, data, overrides):
    values = {"n_" + split: len(data[split]) for split in ("train", "val", "test")}
    values["mixture_modes"] = int(data.get("mixture_modes", 8))
    return replace(cfg, **{name: value for name, value in values.items()
                           if name not in overrides}).validate()


def _validate_dataset(data, cfg):
    from .train import _validate_data
    required = {"R", "train", "val", "test", "train_raw", "val_raw", "test_raw"}
    if cfg.dim > 2:
        required.add("mixture_modes")
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Saved benchmark data is missing keys: {sorted(missing)}")
    for name in required:
        if np.asarray(data[name]).dtype.kind not in "fiu":
            raise ValueError(f"Saved benchmark data must contain real numeric values: {name}")
    modes = np.asarray(data.get("mixture_modes", 8))
    if modes.shape != () or modes.dtype.kind not in "iu":
        raise ValueError("Saved benchmark data must declare an integer number of mixture modes")
    if cfg.dim > 2 and int(modes) != 4:
        raise ValueError("Saved benchmark data must declare four mixture modes")
    _validate_data(data, cfg)


def _validate_target(data, target, cfg):
    return validate_target_metadata(data, target, cfg.dim)


def prepare_experiment(args, saved=None):
    """Return configuration, exact DK data, target, and provenance without writing."""
    checkpoint = _checkpoint_source(args)
    if checkpoint is not None and saved is None:
        saved = {"metadata": _checkpoint_metadata(checkpoint)}
    cfg, overrides, dk_cfg = _configuration(args, saved)
    source = (args.match_dk_run.resolve() if args.match_dk_run is not None else
              Path(checkpoint).resolve().parent if checkpoint is not None else None)
    data_path = args.data if args.data is not None else source / "data.npz" if source is not None else None
    rotation_seed = _rotation_seed(1729 if args.rotation_seed is None else args.rotation_seed)
    run = _json_object(source / "run.json") if source is not None and (source / "run.json").exists() else {}
    if data_path is None:
        data = make_mixture_data(cfg.data_seed, cfg.n_train, cfg.n_val, cfg.n_test,
                                 modes=cfg.mixture_modes, dim=cfg.dim, rotation_seed=rotation_seed)
    else:
        data = load_data(data_path)
        if saved is None and args.match_dk_run is None:
            cfg = _adopt_data_sizes(cfg, data, overrides)
    _validate_dataset(data, cfg)
    fingerprint = data_fingerprint(data)
    if saved is not None:
        expected = saved["metadata"].get("data_fingerprint")
        if expected is not None and fingerprint != expected:
            raise ValueError("Saved score data fingerprint does not match checkpoint")
    if args.match_dk_run is not None and run.get("data_fingerprint", fingerprint) != fingerprint:
        raise ValueError("Saved benchmark data fingerprint does not match run.json")
    target = None
    if cfg.dim > 2:
        target_path = Path(data_path).parent / "target.npz" if data_path is not None else None
        if target_path is not None and target_path.exists():
            target = _numeric_archive(target_path)
        elif data_path is not None:
            raise ValueError("Saved dimension-scaling data requires its matching target.npz")
        else:
            target = target_metadata(cfg.dim, rotation_seed, float(data["R"]))
        rotation_seed = _validate_target(data, target, cfg)
        if args.rotation_seed is not None and args.rotation_seed != rotation_seed:
            raise ValueError("Saved benchmark dataset mismatch: rotation_seed")
    reference_bandwidth = (cfg.density_bandwidth if dk_cfg is None else
                           _reference_evaluation_bandwidth(args, dk_cfg))
    provenance = {
        "method": args.method,
        "source_run": str(source) if source is not None else None,
        "source_checkpoint": str(checkpoint) if checkpoint is not None else None,
        "source_data": str(Path(data_path).resolve()) if data_path is not None else None,
        "defaults_source": "dkgen.train._default_configuration",
        "target": _TARGET if cfg.dim > 2 else "gaussian_mixture",
        "rotation_seed": rotation_seed,
        "data_fingerprint": fingerprint,
        "cli_overrides": overrides,
        "density_settings": {
            "reference_evaluation_bandwidth": reference_bandwidth,
            "reference_density_self_weight": (cfg if dk_cfg is None else dk_cfg).density_self_weight,
            "density_bandwidth": cfg.density_bandwidth,
            "density_self_weight": cfg.density_self_weight,
            "n_neighbor": cfg.n_neighbor,
        },
    }
    if args.method == "edm":
        coordinate_std = float(np.sqrt(np.mean(np.var(np.asarray(data["train"], dtype=np.float64), axis=0))))
        if not np.isfinite(coordinate_std) or coordinate_std <= 0:
            raise ValueError("EDM scaling requires finite, nonzero training coordinate variance")
        if saved is None:
            cfg = replace(cfg, data_scale=cfg.sigma_data / coordinate_std).validate()
        provenance["internal_scaling"] = {
            "training_coordinate_std": coordinate_std, "sigma_data": cfg.sigma_data,
            "data_scale": cfg.data_scale, "fit_split": "train", "center_data": False,
            "preserved_from_checkpoint": saved is not None,
            "note": "The model multiplies DK-normalized inputs by data_scale and divides "
                    "generated outputs by data_scale. Saved data, plots, and all metrics "
                    "remain in the original DK-normalized coordinates.",
        }
    json.dumps(provenance, allow_nan=False)
    return cfg, dk_cfg, data, target, provenance

"""Train a DK-SDE mixture example, select on validation, then evaluate on test."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields as dataclass_fields, replace
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np

from .checkpoint import atomic_json, data_fingerprint, load_checkpoint, save_checkpoint
from .config import Config
from .history import save_convergence_plot
from .selection import load_resume_bests
from .schedule import (SCHEDULE_FIELDS, ScheduleSettings, evaluation_bandwidth, schedule_record,
                       scheduled_config, scheduled_fields)
from .source_snapshot import save_source_snapshot


def _default_configuration(dim):
    """Dimension-specific data/model defaults; schedules always come from schedule.py."""
    cfg = Config()
    if dim == 2:
        return cfg
    return replace(
        cfg, dim=dim, paths=1024, pilot_paths=1024, batch_size=1024,
        n_train=1024, n_val=1024, n_test=1024, eval_samples=1024,
        eval_every=50, mixture_modes=4,
        loss_mode="averaged_kl", n_neighbor=0, density_self_weight=0.5,
        diffusion_mode="neural", sigma_cap=2.0, initial_sigma=None,
        field_bounds="smooth", drift_tail="constant_norm",
        alpha_mode="sigma",
        learning_rate=0.001, optimizer="adam", nu_diagnostic_threshold=0.0,
    )


def _arguments(argv=None, *, default_dim=2):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="New run directory (default: results/run-N, largest current N plus one)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", type=Path, help="Continue a best/latest checkpoint in a new run")
    mode.add_argument("--warm-start", type=Path,
                      help="Start a new optimization from saved weights and data; reset optimizer and update count")
    mode.add_argument("--evaluate", type=Path, help="Evaluate a validation-selected best checkpoint")
    parser.add_argument("--data", type=Path,
                        help="Saved dataset; otherwise generated, or found next to the checkpoint")
    dimension_parser = argparse.ArgumentParser(add_help=False)
    dimension_parser.add_argument("--dim", type=int, default=default_dim)
    dimension, _ = dimension_parser.parse_known_args(argv)
    defaults = _default_configuration(dimension.dim)
    parser.set_defaults(_default_dim=default_dim)
    parser.add_argument("--samples", type=int, default=argparse.SUPPRESS,
                        help="Set paths and all data/evaluation sizes; individual settings take precedence")
    parser.add_argument("--rotation-seed", type=int, default=1729,
                        help="Rotation of a new higher-dimensional target; saved targets are reused")
    parser.add_argument("--bandwidth", type=float, default=argparse.SUPPRESS,
                        help="Set the active schedule's initial bandwidth, or the fixed bandwidth with --schedule none")
    parser.add_argument("--alpha-target", dest="alpha_sigma_target", type=float,
                        default=argparse.SUPPRESS, help="Alias for --alpha-sigma-target")
    for field in dataclass_fields(defaults):
        default = getattr(defaults, field.name)
        options = {"type": type(default), "default": argparse.SUPPRESS,
                   "help": f"Configuration value (default: {default})"}
        if field.name in ("width", "depth"):
            options["help"] = f"Drift network hidden {field.name} (default: {default})"
        if field.name in ("diffusion_width", "diffusion_depth"):
            options["help"] = f"Diffusion network hidden {field.name.split('_')[1]} (default: {default})"
        if field.name == "sigma_drift_scale":
            options["type"] = float
            options["help"] = "Legacy diffusion threshold override (default: use drift_cap)"
        if field.name == "alpha_sigma_multiplier":
            options["type"] = float
            options["help"] = f"Fixed sigma multiplier when --alpha-sigma-target is 0 (default: {default})"
        if field.name == "alpha_sigma_target":
            options["type"] = float
            options["help"] = (
                f"Target mean alpha using the previous update's mean sigma (default: {default}); "
                "0 disables normalization. May be changed on resume")
        if field.name == "alpha_cap":
            options["help"] = "Legacy normalized-rule cap, used only when alpha_sigma_multiplier is None"
        if field.name == "initial_sigma":
            options["type"] = float
            options["help"] = (
                "Fixed interior diffusion at initialization; default uses small random "
                "output weights and zero bias instead")
        if field.name == "train_initial":
            options.pop("type")
            options["action"] = argparse.BooleanOptionalAction
            options["help"] = (
                f"Optimize the initial Gaussian mean and scalar standard deviation (default: {default}); "
                "--no-train-initial freezes a saved Gaussian")
        if field.name == "initial_s0_min":
            options["help"] = (
                f"Minimum trainable initial standard deviation in normalized coordinates (default: {default}); "
                "an explicit increase on resume projects s0 and revalidates if the law changes")
        if field.name == "dtype":
            options["choices"] = ("float32", "float64")
        if field.name == "optimizer":
            options["choices"] = ("adam", "sgd")
        if field.name == "update_clip_mode":
            options["choices"] = ("kl", "norm")
            options["help"] = "Per-network update cap: fraction of absolute training KL, or legacy norm cap"
        if field.name == "kl_clip_fraction":
            options["help"] = f"Maximum absolute projected KL change per network / absolute training KL (default: {default})"
        if field.name == "max_update_norm":
            options["help"] = f"Parameter-increment norm cap, used only with --update-clip-mode norm (default: {default})"
        if field.name in ("drift_initialization", "diffusion_initialization"):
            options["choices"] = ("data_rms", "glorot")
        if field.name == "field_bounds":
            options["choices"] = ("smooth", "none")
        if field.name == "drift_tail":
            options["choices"] = ("constant_norm", "linear")
        if field.name == "drift_cap":
            options["help"] = (
                f"Smooth neural drift bound and constant-norm tail magnitude (default: {default}); "
                "may be changed with --warm-start")
        if field.name == "alpha_mode":
            options["choices"] = ("constant", "adaptive", "sigma")
        if field.name == "diffusion_mode":
            options["choices"] = ("constant", "neural", "drift_norm")
        if field.name == "loss_mode":
            options["choices"] = ("averaged_kl", "forward_kl", "forward_kl_kde", "backward_kl")
            options["help"] = ("KDE loss: averaged_kl, forward_kl_kde, or backward_kl; "
                               "legacy forward_kl uses assignment counts")
        if field.name == "response_weight_mode":
            options["choices"] = ("count_squared", "count_log", "count_only", "three_term", "kde")
            options["help"] = (
                "Response coefficient: count_squared uses half the squared normalized assignment count; "
                "count_log uses equal data-assignment count and log-ratio terms; "
                "count_only keeps half the normalized assignment count; "
                "three_term also includes the density ratio; kde uses ratio and log-ratio "
                f"(default: {default}); "
                "may be changed on resume, resetting the previous response mean")
        if field.name == "response_baseline":
            options["choices"] = ("none", "previous_mean")
            options["help"] = (
                "KDE response baseline: subtract the previous accepted update's "
                f"mean coefficient (default: {default}); may be changed on resume")
        if field.name == "n_neighbor":
            options["help"] = (
                f"Gaussian density estimate: 0 uses all points; positive k uses k "
                f"neighbors, excluding self from those k (default: {default})")
        if field.name == "density_bandwidth":
            options["help"] = (
                f"Gaussian density standard deviation in normalized coordinates (default: {default})")
        if field.name == "density_self_weight":
            options["help"] = (
                f"Self-kernel weight in [0, 1] (default: {default}); "
                "0 excludes self and uses population-minus-one normalization. "
                "May be changed on resume, restarting best-KL selection")
        if field.name == "schedule":
            options["choices"] = ("prescribed", "none")
            options["help"] = "prescribed changes four coefficients; none keeps them fixed"
        if field.name == "schedule_updates":
            options["help"] = f"Updates to reach the prescribed schedule's final values, then hold them (default: {default})"
        if field.name.startswith("schedule_") and field.name.endswith(("_start", "_end")):
            options["help"] = f"Prescribed schedule {field.name[len('schedule_'):].replace('_', ' ')} (default: {default})"
        if field.name == "schedule_evaluation_bandwidth":
            options["help"] = f"Fixed validation/test KDE bandwidth with --schedule prescribed (default: {default})"
        if field.name == "neighbors":
            options["help"] = (
                f"Neighbors per datum for forward-KL assignments and plot counts (default: {default})")
        if field.name == "score_baseline":
            options["help"] = f"Forward-KL score baseline; unused by averaged KL (default: {default})"
        if field.name in SCHEDULE_FIELDS:
            options["help"] += "; use --schedule none to set a fixed value"
        parser.add_argument("--" + field.name.replace("_", "-"), **options)
    return parser.parse_args(argv)


def _create_output_directory(requested=None):
    """Use the largest existing run-number prefix, including renamed run folders."""
    if requested is not None:
        output = Path(requested).resolve()
        output.mkdir(parents=True, exist_ok=False)
        return output
    root = Path("results").resolve()
    root.mkdir(parents=True, exist_ok=True)
    numbers = []
    for entry in root.iterdir():
        match = re.match(r"run-(\d+)", entry.name)
        if match:
            numbers.append(int(match.group(1)))
    number = max(numbers, default=0) + 1
    while True:
        output = root / f"run-{number}"
        try:
            output.mkdir()
            return output
        except FileExistsError:
            # Another process may have claimed this number since the scan.
            number += 1


def _configuration(args, saved):
    overrides = {field.name: getattr(args, field.name)
                 for field in dataclass_fields(Config) if hasattr(args, field.name)}
    if hasattr(args, "samples"):
        for name in ("paths", "pilot_paths", "n_train", "n_val", "n_test", "eval_samples"):
            overrides.setdefault(name, args.samples)
    if overrides.get("alpha_sigma_target") == 0:
        overrides["alpha_sigma_target"] = None
    default_dim = (saved["metadata"]["config"]["dim"] if saved is not None
                   else getattr(args, "_default_dim", 2))
    defaults = _default_configuration(overrides.get("dim", default_dim))
    values = asdict(defaults) if saved is None else dict(saved["metadata"]["config"])
    selected_schedule = overrides.get("schedule", values["schedule"])
    if hasattr(args, "bandwidth"):
        bandwidth_field = {"prescribed": "schedule_density_bandwidth_start",
                           "none": "density_bandwidth"}[selected_schedule]
        if bandwidth_field in overrides and overrides[bandwidth_field] != args.bandwidth:
            raise ValueError("--bandwidth disagrees with --" + bandwidth_field.replace("_", "-"))
        overrides[bandwidth_field] = args.bandwidth
    schedule_settings = {field.name for field in dataclass_fields(ScheduleSettings)} - {"schedule"}
    active_settings = schedule_settings if selected_schedule == "prescribed" else set()
    inactive = sorted((set(overrides) & schedule_settings) - active_settings)
    if inactive:
        raise ValueError("Options do not apply to --schedule " + selected_schedule + ": "
                         + ", ".join("--" + name.replace("_", "-") for name in inactive))
    if saved is not None:
        permitted = {"updates", "eval_every", "log_every", "nu_diagnostic_threshold",
                     "nu_diagnostic_paths", "nu_diagnostic_interval",
                     "alpha_sigma_target", "density_self_weight", "response_baseline",
                     "response_weight_mode", "train_initial", "initial_s0_min"} if args.resume or args.warm_start else set()
        if args.warm_start:
            # Reuse the same tensors and dataset, but start new optimization
            # and validation records for the changed field or optimizer.
            permitted |= {"drift_cap", "sigma_min", "sigma_cap", "paths", "batch_size",
                          "density_bandwidth", "density_self_weight", "n_neighbor", "optimizer", "learning_rate", "drift_learning_rate_multiplier",
                          "drift_weight_decay", "diffusion_weight_decay",
                          "update_clip_mode", "kl_clip_fraction", "max_update_norm",
                           "schedule"} | schedule_settings
        elif values.get("schedule", "none") != "none":
            permitted -= set(scheduled_fields(values["schedule"]))
        changed = [name for name, value in overrides.items()
                   if value != values.get(name) and name not in permitted]
        if changed:
            hint = "; use --warm-start to change the schedule" if set(changed) & (schedule_settings | {"schedule"}) else ""
            raise ValueError("Checkpoint configuration mismatch: " + ", ".join(changed) + hint)
    values.update(overrides)
    cfg = Config(**values)
    if cfg.schedule != "none" and (saved is None or args.warm_start):
        cfg = scheduled_config(cfg, 0)
        conflicting = [name for name in scheduled_fields(cfg.schedule)
                       if name in overrides and overrides[name] != getattr(cfg, name)]
        if conflicting:
            raise ValueError("The selected schedule controls " + ", ".join(conflicting)
                             + "; use its --schedule-<field>-start/end options, or --schedule none for fixed values")
        if "updates" not in overrides:
            cfg = replace(cfg, updates=defaults.updates)
    cfg.validate()
    if cfg.dim < 2 or cfg.dim % 2:
        raise ValueError("Mixture projection plots require an even dim >= 2")
    if cfg.dim > 2 and cfg.mixture_modes != 4:
        raise ValueError("The higher-dimensional target has four modes")
    if getattr(args, "rotation_seed", 1729) < 0:
        raise ValueError("rotation_seed must be nonnegative")
    if args.resume and cfg.updates < saved["metadata"]["iteration"]:
        raise ValueError("--updates is the total update count and cannot be less than the resumed iteration")
    return cfg, overrides


def _previous_sigma_mean(saved, cfg):
    """Restore the lagged mean, or bootstrap when first enabling normalization."""
    if saved is None or cfg.alpha_mode != "sigma" or cfg.alpha_sigma_target is None:
        return None
    source_cfg = saved["metadata"]["config"]
    if source_cfg.get("alpha_mode") != "sigma" or source_cfg.get("alpha_sigma_target") is None:
        return None
    return saved["metadata"]["extra"]["previous_sigma_mean"]


def _previous_response_mean(saved, cfg):
    """Restore a mean only for the same coefficient, or bootstrap from zero."""
    if saved is None or cfg.loss_mode == "forward_kl" or cfg.response_baseline != "previous_mean":
        return None
    source_cfg = saved["metadata"]["config"]
    if (source_cfg.get("loss_mode") != cfg.loss_mode
            or source_cfg.get("response_baseline", "none") != "previous_mean"
            or source_cfg.get("response_weight_mode", "kde") != cfg.response_weight_mode):
        return None
    return saved["metadata"]["extra"]["previous_response_mean"]


def _initial_distribution_record(params, cfg):
    """Record the actual Gaussian, including fixed legacy and frozen learned laws."""
    from .initial import initial_distribution

    mean, s0 = initial_distribution(params, cfg)
    return {"mean": np.asarray(mean, dtype=np.float64).tolist(), "s0": float(s0),
            "trainable": bool(cfg.train_initial)}


def _prepare_saved_initial_distribution(saved, cfg):
    """Add a requested initial branch without restarting saved network moments."""
    from .initial import add_initial_params

    source_params = saved["params"]
    before = _initial_distribution_record(source_params, Config(**saved["metadata"]["config"]))
    params = add_initial_params(source_params, cfg)
    optimizer_state = saved.get("optimizer_state")
    if "initial" in params:
        current_s0 = np.asarray(params["initial"]["s0"])
        if float(current_s0) < cfg.initial_s0_min:
            with np.errstate(over="ignore"):
                projected_s0 = np.asarray(cfg.initial_s0_min, dtype=current_s0.dtype)
            if float(projected_s0) < cfg.initial_s0_min:
                projected_s0 = np.nextafter(projected_s0, np.asarray(np.inf, dtype=current_s0.dtype))
            if not np.isfinite(projected_s0):
                raise ValueError("Initial s0 floor cannot be represented in the configured dtype")
            params = dict(params)
            params["initial"] = dict(params["initial"], s0=projected_s0)
        if "initial" not in source_params and optimizer_state is not None:
            optimizer_state = dict(optimizer_state)
            for moment in ("m", "v"):
                optimizer_state[moment] = dict(optimizer_state[moment])
                optimizer_state[moment]["initial"] = {
                    name: np.zeros(np.shape(value), dtype=np.float64)
                    for name, value in params["initial"].items()}
    after = _initial_distribution_record(params, cfg)
    changed = before["mean"] != after["mean"] or before["s0"] != after["s0"]
    return params, optimizer_state, ({"previous": before, "current": after} if changed else None)


def _validate_data(data, cfg):
    if int(data.get("mixture_modes", 8)) != cfg.mixture_modes:
        raise ValueError("Saved dataset mixture_modes does not match the configuration")
    R = np.asarray(data["R"])
    if R.shape != () or not np.isfinite(R) or float(R) <= 0:
        raise ValueError("Dataset scale R must be a positive finite scalar")
    for split, size in (("train", cfg.n_train), ("val", cfg.n_val), ("test", cfg.n_test)):
        for name in (split, split + "_raw"):
            array = np.asarray(data[name])
            if array.shape != (size, cfg.dim) or not np.isfinite(array).all():
                raise ValueError(f"Invalid saved dataset array: {name}")
        if not np.allclose(data[split] * R, data[split + "_raw"], rtol=1e-6, atol=1e-7):
            raise ValueError(f"Saved normalization disagrees with R for {split}")
    expected_R = float(np.linalg.norm(data["train_raw"], axis=1).max())
    if not np.isclose(float(R), expected_R, rtol=1e-6, atol=1e-7):
        raise ValueError("Saved R is not the maximum training-sample radius")


def _environment(jax):
    code_root = Path(__file__).resolve().parents[1]
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(code_root), *args], stderr=subprocess.DEVNULL,
                text=True, timeout=10,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    versions = {}
    for package in ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt",
                    "jax-cuda13-plugin", "jax-cuda13-pjrt", "numpy", "scipy", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": versions, "devices": [str(device) for device in jax.devices()],
        "git_revision": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
        "environment": {name: os.environ[name] for name in (
            "JAX_PLATFORMS", "JAX_ENABLE_X64", "CUDA_VISIBLE_DEVICES", "XLA_FLAGS",
            "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "SLURM_JOB_ID", "SLURM_JOB_PARTITION") if name in os.environ},
    }


def _precision_message(metrics):
    message = ""
    if "knn_precision" in metrics:
        precision = metrics["knn_precision"]
        message = (" | precision=n/a | outside=n/a" if precision is None else
                   f" | precision={precision:.2%} | outside={metrics['knn_outside_fraction']:.2%}")
    for name, label in (("nearest_data_distance_mean", "NN_mean"),
                        ("knn_outside_distance_mean", "outside_mean"),
                        ("knn_outside_distance_p95", "outside_p95")):
        if name in metrics:
            value = metrics[name]
            text = "n/a" if value is None else f"{value:.4g}"
            message += f" | {label}={text}"
    return message


def _screen_message(record):
    event = record["event"]
    step = record.get("iteration", 0)
    if event == "update":
        diagnostics = record["diagnostics"]
        if "drift_update_clip_scale" in diagnostics:
            drift_scale = diagnostics["drift_update_clip_scale"]
            diffusion_scale = diagnostics["diffusion_update_clip_scale"]
            clip_message = (f" | clip_F={drift_scale:.3g} | clip_sigma={diffusion_scale:.3g}"
                            if min(drift_scale, diffusion_scale) < 1.0 else "")
        else:
            clip_scale = diagnostics.get("update_clip_scale", 1.0)
            clip_message = f" | clip={clip_scale:.3g}" if clip_scale < 1.0 else ""
        training_kl = diagnostics.get("training_kl")
        kl_message = f" | KL(train)={training_kl:.6g}" if training_kl is not None else ""
        return (f"Step {step:5d} | {record['update_seconds']:.1f}s{kl_message}"
                f" | grad={diagnostics['gradient_norm']:.3g}"
                f" | alpha={diagnostics['alpha_mean']:.3g}{clip_message}")
    if event in ("validation", "test"):
        metrics = record["metrics"]
        criterion = "KL" if record.get("selection_metric") == "validation_averaged_kl" else "SW1"
        label = (f"Validation {step:5d}" if event == "validation"
                 else f"Test (best {criterion} step {step})")
        best = ""
        if event == "validation":
            if record.get("selected_best_kl"):
                best += " | new best KL"
            elif metrics.get("averaged_kl") is None and record.get("selected_best"):
                best += " | new best SW1"
        kl = metrics.get("averaged_kl")
        kl_message = "n/a" if kl is None else f"{kl:.5f}"
        return (f"{label} | KL={kl_message}{best}{_precision_message(metrics)}"
                f" | SW1={metrics['sliced_wasserstein1']:.5f}")
    if event == "summary":
        best_kl = record.get("best_validation_averaged_kl")
        if best_kl is None:
            return "Best KL | unavailable"
        test_metrics = record.get("test_metrics_best_kl") or {}
        test_kl = test_metrics.get("averaged_kl")
        test_message = "n/a" if test_kl is None else f"{test_kl:.5f}"
        return (f"Best KL | step {record['best_kl_iteration']}"
                f" | validation={best_kl:.5f} | test={test_message}"
                f"{_precision_message(test_metrics)}")
    if event == "failure":
        message = " ".join(str(record["message"]).split())
        return f"Stopped at step {step}: {record['error_type']}: {message}"
    if event == "nu_growth_diagnostic":
        return f"Nu growth | step {step} | {record['paths']} paths | {record['plot']}"
    return f"{event} | step {step}"


def _log(path, record, *, display=True):
    message = json.dumps(record, sort_keys=True, allow_nan=False)
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")
        stream.flush()
    if display:
        print(_screen_message(record), flush=True)


def _update_convergence_plot(log_path):
    try:
        save_convergence_plot(log_path)
    except OSError as exc:
        print(f"Warning: could not update convergence.png: {exc}", file=sys.stderr, flush=True)


def _finite_tree(jax, tree, description):
    for leaf in jax.tree_util.tree_leaves(tree):
        if not np.isfinite(np.asarray(leaf)).all():
            raise FloatingPointError(f"Nonfinite {description}; stopping without applying this update")


def _device_memory(jax):
    result = []
    for device in jax.devices():
        try:
            stats = device.memory_stats()
        except (AttributeError, RuntimeError):
            stats = None
        result.append({"device": str(device), "allocator_stats": stats})
    return result


def _history_plot_slots(inherited_iterations, first_update, last_update, eval_every):
    from .sample_history import history_plot_slots
    return history_plot_slots(inherited_iterations, first_update, last_update, eval_every)


def _snapshot_drift_grid():
    """Use the same normalized square [-2, 2]^2 for drift and diffusion snapshots."""
    gx, gy = np.meshgrid(np.linspace(-2.0, 2.0, 35), np.linspace(-2.0, 2.0, 35))
    return np.stack((gx, gy), axis=-1)


def _nu_diagnostic_due(cfg, gradient_norm, iteration, last_iteration):
    """Trigger on the raw displayed gradient, with spacing between attempts."""
    return (cfg.nu_diagnostic_threshold > 0 and gradient_norm > cfg.nu_diagnostic_threshold
            and (last_iteration is None or iteration - last_iteration >= cfg.nu_diagnostic_interval))


def run(args):
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Output directory already exists: {args.output}")
    source = args.resume or args.warm_start or args.evaluate
    saved = load_checkpoint(source) if source else None
    cfg, overrides = _configuration(args, saved)
    kl_selection_reset = bool(args.resume and cfg.density_self_weight
                              != saved["metadata"]["config"]["density_self_weight"])
    density_estimator_change = ({
        "iteration": saved["metadata"]["iteration"],
        "density_self_weight": {
            "previous": saved["metadata"]["config"]["density_self_weight"],
            "current": cfg.density_self_weight,
        },
    } if kl_selection_reset else None)
    if args.evaluate and saved["metadata"]["role"] != "best":
        raise ValueError("--evaluate requires a best checkpoint selected using validation samples")

    # Set precision before constructing any JAX arrays or importing model/engine.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    jax.config.update("jax_enable_x64", cfg.dtype == "float64")
    import jax.numpy as jnp
    from .data import load_data, make_mixture_data, save_data
    from .engine import Engine
    from .metrics import evaluate_samples
    from .model import (calibrate_diffusion_initialization, calibrate_drift_initialization,
                        fields, init_params, parameter_count)
    from .plots import (diagnostic_grid, projection_frame, projection_grids,
                        save_diagnostics)
    from .optimizer import adam_update, init_optimizer, sgd_update
    from .nu_growth import save_nu_growth
    from .sample_history import (inherit_sample_history, read_sample_history, record_sample_history,
                                 save_sample_history_plot)

    data_path = args.data or (Path(source).resolve().parent / "data.npz" if source else None)
    data = load_data(data_path) if data_path else make_mixture_data(
        cfg.data_seed, cfg.n_train, cfg.n_val, cfg.n_test, modes=cfg.mixture_modes,
        dim=cfg.dim, rotation_seed=args.rotation_seed)
    if args.data and saved is None and "mixture_modes" not in overrides:
        # A saved target takes precedence over the default for newly drawn data.
        cfg = replace(cfg, mixture_modes=int(data.get("mixture_modes", 8)))
    _validate_data(data, cfg)
    target = None
    if cfg.dim > 2:
        from .data import target_metadata, validate_target_metadata
        target_path = Path(data_path).resolve().parent / "target.npz" if data_path else None
        if target_path is not None and target_path.exists():
            with np.load(target_path, allow_pickle=False) as archive:
                target = {name: archive[name] for name in archive.files}
        else:
            rotation_seed = (saved["metadata"].get("extra", {}).get("rotation_seed", args.rotation_seed)
                             if saved else args.rotation_seed)
            target = target_metadata(cfg.dim, rotation_seed, data["R"], modes=cfg.mixture_modes)
        validate_target_metadata(data, target, cfg.dim)
    data_hash = data_fingerprint(data)
    if saved and data_hash != saved["metadata"]["data_fingerprint"]:
        raise ValueError("Saved dataset fingerprint does not match the checkpoint")

    prepared_params = prepared_optimizer = initial_distribution_change = None
    if saved is not None:
        prepared_params, prepared_optimizer, initial_distribution_change = _prepare_saved_initial_distribution(saved, cfg)

    best_saved = best_kl_saved = None
    initial_selection_changes = {}
    if initial_distribution_change is not None:
        initial_selection_changes["current"] = initial_distribution_change
    if args.resume and initial_distribution_change is None:
        best_saved, best_kl_saved = load_resume_bests(source, saved, data_hash)
        for name, selected in (("best", best_saved), ("best_kl", best_kl_saved)):
            if selected is not None:
                _, _, change = _prepare_saved_initial_distribution(selected, cfg)
                if change is not None:
                    initial_selection_changes[name] = change
        if initial_selection_changes:
            best_saved = best_kl_saved = None
        if kl_selection_reset:
            best_kl_saved = None
    elif args.evaluate:
        if saved["metadata"]["selection_metric"] == "validation_averaged_kl":
            best_kl_saved = saved
        else:
            best_saved = saved

    normalize_alpha = cfg.alpha_mode == "sigma" and cfg.alpha_sigma_target is not None
    # Changed diffusion bounds invalidate the source's lagged sigma mean.
    reset_lagged_means = args.warm_start or initial_distribution_change is not None
    previous_sigma_mean = _previous_sigma_mean(None if reset_lagged_means else saved, cfg)
    center_response = cfg.loss_mode != "forward_kl" and cfg.response_baseline == "previous_mean"
    previous_response_mean = _previous_response_mean(None if reset_lagged_means else saved, cfg)
    output = _create_output_directory(args.output)
    print(f"Results: {output}", flush=True)
    if cfg.schedule != "none":
        print(f"Schedule: {cfg.schedule}, linear through update {cfg.schedule_updates}, then fixed; "
              f"validation/test KDE bandwidth={evaluation_bandwidth(cfg):g}", flush=True)
    if args.warm_start:
        print(f"Warm start: {source} (step {saved['metadata']['iteration']}); "
              "optimizer and update count reset", flush=True)
    if args.resume and initial_selection_changes:
        print("Initial Gaussian changed in " + ", ".join(initial_selection_changes)
              + "; restarting validation selection before the next update.", flush=True)
    if kl_selection_reset:
        print(f"Density estimator changed: self weight "
              f"{saved['metadata']['config']['density_self_weight']:g} -> {cfg.density_self_weight:g}; "
              f"restarting best-KL selection and validating step {saved['metadata']['iteration']} "
              "before the next update.", flush=True)
    source_snapshot = save_source_snapshot(output)
    log_path = output / "history.jsonl"
    started = time.perf_counter()
    env = _environment(jax)
    provenance = {"source_git_revision": env["git_revision"],
                  "source_git_status": env["git_status"]}
    if kl_selection_reset:
        provenance.update(kl_selection_reset=True, density_estimator_change=density_estimator_change)
    if initial_distribution_change is not None:
        provenance["initial_distribution_change"] = initial_distribution_change
    if initial_selection_changes:
        provenance["initial_selection_changes"] = initial_selection_changes
    if args.warm_start:
        provenance["warm_start_source"] = {
            "checkpoint": str(Path(source).resolve()),
            "iteration": saved["metadata"]["iteration"],
            "config_overrides": overrides,
        }
    save_data(output / "data.npz", data)
    if target is not None:
        np.savez_compressed(output / "target.npz", **target)
        provenance.update(rotation_seed=int(target["rotation_seed"]))
    atomic_json(output / "config.json", asdict(cfg))
    run_metadata = {
        "config": asdict(cfg), "cli_overrides": overrides,
        "defaults": asdict(_default_configuration(cfg.dim)),
        "mode": ("warm_start" if args.warm_start else "evaluate" if args.evaluate
                 else "resume" if args.resume else "train"),
        "source_checkpoint": str(Path(source).resolve()) if source else None,
        "source_iteration": saved["metadata"]["iteration"] if saved else None,
        "optimizer_reinitialized": bool(args.warm_start),
        "kl_selection_reset": kl_selection_reset,
        "density_estimator_change": density_estimator_change,
        "source_snapshot": source_snapshot,
        "data_fingerprint": data_hash, "environment": env,
        "approved_initial_choices": {"drift_tail": cfg.drift_tail,
                                     "tail_attraction": (cfg.tail_attraction if cfg.drift_tail == "linear"
                                                         else None), "optimizer": cfg.optimizer,
                                     "diffusion_mode": cfg.diffusion_mode,
                                     "initial_sigma": (cfg.constant_sigma if cfg.diffusion_mode == "constant"
                                                       else cfg.initial_sigma if cfg.diffusion_mode == "neural"
                                                       else None),
                                     "initial_raw_diffusion_output": cfg.initial_raw_diffusion_output},
        "engineering_defaults_are_tuned": False,
    }
    if cfg.schedule != "none":
        run_metadata["schedule"] = schedule_record(
            cfg, saved["metadata"]["iteration"] if saved and not args.warm_start else 0)
    if target is not None:
        run_metadata["target"] = "correlated_four_mode_gaussian_mixture"
        run_metadata["note"] = "All metrics use the full state dimension; figures are projections."
    engine = Engine(cfg)
    best_score, best_iteration = float("inf"), 0
    best_kl_score = best_kl_iteration = None
    if best_saved is not None:
        best_score = best_saved["metadata"]["best_validation"]
        best_iteration = best_saved["metadata"]["best_iteration"]
    if best_kl_saved is not None:
        best_kl_score = best_kl_saved["metadata"]["best_validation"]
        best_kl_iteration = best_kl_saved["metadata"]["best_iteration"]
    if saved:
        initialization_report = saved["metadata"].get("extra", {}).get("initialization")
        params = jax.tree_util.tree_map(jnp.asarray, prepared_params)
        # Continue the source's random stream, without redrawing parameters.
        key = jnp.asarray(saved["key"])
        if args.warm_start:
            optimizer_state = init_optimizer(params, cfg)
            iteration = 0
            pending_validation = False
        else:
            # Adam moments remain host float64 even for float32 model parameters.
            optimizer_state = prepared_optimizer
            iteration = saved["metadata"]["iteration"]
            pending_validation = (saved["metadata"]["pending_validation"] or kl_selection_reset
                                  or bool(initial_selection_changes)
                                  or bool(args.resume and saved["metadata"].get("extra", {}).get(
                                      "kl_validation_pending", False)))
    else:
        key, init_key = jax.random.split(jax.random.PRNGKey(cfg.seed))
        params = init_params(init_key, cfg)
        initialization_report = {"drift": {"method": "glorot"},
                                 "diffusion": {"method": "glorot"}}
        if cfg.drift_initialization == "data_rms":
            params, initialization_report["drift"] = calibrate_drift_initialization(params, data["train"])
        if cfg.diffusion_mode == "constant":
            initialization_report["diffusion"] = {"method": "constant", "sigma": cfg.constant_sigma}
        elif cfg.diffusion_mode == "drift_norm":
            initialization_report["diffusion"] = {
                "method": "drift_norm", "sigma_min": cfg.sigma_min,
                "sigma_cap": cfg.sigma_cap,
                "sigma_drift_scale": (cfg.drift_cap if cfg.sigma_drift_scale is None
                                      else cfg.sigma_drift_scale)}
        elif cfg.diffusion_initialization == "data_rms":
            params, initialization_report["diffusion"] = calibrate_diffusion_initialization(params, data["train"])
        if cfg.diffusion_mode == "neural":
            random_output = cfg.initial_sigma is None
            initialization_report["diffusion"]["output_layer"] = {
                "weights": "normal" if random_output else "zero",
                "weight_standard_deviation": 0.01 / cfg.diffusion_width**0.5 if random_output else 0.0,
                "bias": 0.0 if random_output else cfg.initial_raw_diffusion_output,
            }
        iteration = 0
        pending_validation = False
        optimizer_state = init_optimizer(params, cfg)
    if initialization_report is not None:
        provenance["initialization"] = initialization_report
    run_metadata["parameter_count"] = parameter_count(params)
    run_metadata["initialization"] = initialization_report
    run_metadata["initialization_applied"] = saved is None
    run_metadata["initial_distribution"] = _initial_distribution_record(params, cfg)
    atomic_json(output / "run.json", run_metadata)
    normalized_train = jnp.asarray(data["train"], dtype=cfg.dtype)
    history_reference = np.asarray(normalized_train)
    R = float(data["R"])
    projection_center, projection_basis = projection_frame(cfg.dim, target)
    projection_options = ({} if cfg.dim == 2 else {
        "projection_center": projection_center, "projection_basis": projection_basis})

    def field_grid(samples=None, reference=None, *, snapshot=False):
        if cfg.dim == 2:
            return _snapshot_drift_grid() if snapshot else diagnostic_grid(samples, reference)
        return projection_grids(
            history_reference if samples is None else samples,
            history_reference if reference is None else reference,
            center=projection_center, basis=projection_basis,
            extent=2.0 if snapshot else None)

    # Evaluation streams are independent of the saved training-key sequence.
    validation_key = jax.random.fold_in(jax.random.PRNGKey(cfg.seed), 17001)
    test_key = jax.random.fold_in(jax.random.PRNGKey(cfg.seed), 17002)
    snapshot_cfg = cfg
    if cfg.schedule == "prescribed":
        snapshot_kernel = jax.jit(jax.vmap(lambda current_params, x, values: fields(
            current_params, x, replace(snapshot_cfg, drift_cap=values[0], sigma_min=values[1])),
            in_axes=(None, 0, None)))

        def snapshot_fields(current_params, x):
            return snapshot_kernel(current_params, x,
                                   jnp.asarray([cfg.drift_cap, cfg.sigma_min], cfg.dtype))
    else:
        snapshot_fields = jax.jit(jax.vmap(
            lambda current_params, x: fields(current_params, x, snapshot_cfg), in_axes=(None, 0)))

    def render_snapshot(step, slot):
        try:
            save_sample_history_plot(output, iteration=step, slot=slot)
        except Exception as exc:
            # A figure must never prevent recovery of the completed update.
            print(f"Warning: could not save sample history plot for update {step}: {exc}",
                  file=sys.stderr, flush=True)

    completed_iteration = iteration
    latest_checkpoint_iteration = None

    def checkpoint(name, checkpoint_params=None, checkpoint_key=None,
                   checkpoint_iteration=None, checkpoint_extra=None, checkpoint_optimizer=None):
        nonlocal latest_checkpoint_iteration
        saved_iteration = iteration if checkpoint_iteration is None else checkpoint_iteration
        saved_cfg = scheduled_config(cfg, saved_iteration)
        extra = dict(provenance)
        if saved_cfg.schedule != "none":
            extra["schedule"] = schedule_record(saved_cfg, saved_iteration)
        if normalize_alpha:
            extra["previous_sigma_mean"] = previous_sigma_mean
        if center_response:
            extra["previous_response_mean"] = previous_response_mean
        if checkpoint_extra is not None:
            extra.update(checkpoint_extra)
        if name == "latest":
            extra.update(best_kl_iteration=best_kl_iteration,
                         best_validation_averaged_kl=best_kl_score)
        kl_selected = name == "best_kl"
        save_checkpoint(
            output / (name + ".npz"), params=params if checkpoint_params is None else checkpoint_params,
            key=key if checkpoint_key is None else checkpoint_key,
            iteration=saved_iteration,
            best_iteration=best_kl_iteration if kl_selected else best_iteration,
            best_validation=best_kl_score if kl_selected else best_score, config=saved_cfg,
            pending_validation=pending_validation if name == "latest" else False,
            data_hash=data_hash, role="latest" if name == "latest" else "best", extra=extra,
            selection_metric="validation_averaged_kl" if kl_selected else "validation_sliced_wasserstein1",
            optimizer_state=optimizer_state if checkpoint_params is None else checkpoint_optimizer,
        )
        if name == "latest":
            latest_checkpoint_iteration = saved_iteration

    def copy_best(name, state):
        selected_params, selected_optimizer, _ = _prepare_saved_initial_distribution(state, cfg)
        extra = {}
        if normalize_alpha:
            extra["previous_sigma_mean"] = _previous_sigma_mean(state, cfg)
        if center_response:
            extra["previous_response_mean"] = _previous_response_mean(state, cfg)
        source_extra = state["metadata"].get("extra", {})
        metrics = source_extra.get("validation_metrics")
        if kl_selection_reset or source_extra.get("kl_validation_pending", False):
            # SW1 remains comparable, but this state's old KL estimate does not.
            # A later resume of this copied winner must validate it again.
            extra.update(kl_validation_pending=True, best_kl_iteration=None,
                         best_validation_averaged_kl=None)
        elif metrics is not None:
            extra["validation_metrics"] = metrics
        checkpoint(name, selected_params, state["key"], state["metadata"]["iteration"], extra,
                   selected_optimizer)

    def evaluate(current_params, sample_key, reference, n_samples):
        generation_started = time.perf_counter()
        samples = engine.sample(current_params, sample_key, n_samples)
        samples = np.asarray(jax.block_until_ready(samples))
        generation_seconds = time.perf_counter() - generation_started
        if not np.isfinite(samples).all():
            raise FloatingPointError("Nonfinite generated samples; evaluation stopped")
        generated_raw = (np.asarray(samples, dtype=np.float64) * R
                         if target is not None else samples * R)
        metrics = evaluate_samples(
            generated_raw, reference, mixture_modes=cfg.mixture_modes,
            n_neighbor=0 if target is not None else cfg.n_neighbor,
            density_bandwidth=evaluation_bandwidth(cfg),
            density_self_weight=cfg.density_self_weight,
            normalization_scale=R, **({"target": target} if target is not None else {}))
        json.dumps(metrics, allow_nan=False)
        return samples, metrics, generation_seconds

    initial_metrics = None
    null_metrics = None

    def update_convergence():
        _update_convergence_plot(log_path)

    training_seconds = 0.0
    update_durations = []
    validation_seconds = 0.0
    nu_diagnostic_seconds = 0.0
    last_nu_diagnostic_iteration = None

    def render_nu_growth(step, current_params, update_key, diagnostics):
        nonlocal nu_diagnostic_seconds
        started = time.perf_counter()
        try:
            growth = diagnostics.get("lambda_hat")
            alpha_options = ({"alpha_sigma_multiplier": diagnostics["alpha_sigma_multiplier"]}
                             if normalize_alpha else {})
            traces = engine.covector_trajectories(
                current_params, update_key, 0.0 if growth is None else growth,
                n_paths=cfg.nu_diagnostic_paths, **alpha_options)
            grid_z = field_grid(snapshot=True)
            drift, _ = snapshot_fields(
                current_params, jnp.asarray(grid_z.reshape(-1, cfg.dim), dtype=cfg.dtype))
            drift = np.asarray(jax.device_get(drift)).reshape(grid_z.shape)
            plot = save_nu_growth(
                output / "nu_growth", iteration=step,
                trajectories_normalized=traces["trajectories_normalized"],
                covectors=traces["covectors"], path_indices=traces["path_indices"],
                dt=cfg.dt, gradient_norm=diagnostics["gradient_norm"],
                update_clip_scale=diagnostics.get("update_clip_scale", 1.0),
                lambda_hat=growth, data_hash=data_hash, reference_normalized=history_reference,
                update_key=np.asarray(update_key), alpha=traces["alpha"],
                drift_grid_normalized=grid_z, drift_normalized=drift, **projection_options)
            _log(log_path, {"event": "nu_growth_diagnostic", "iteration": step,
                            "gradient_norm": diagnostics["gradient_norm"],
                            "paths": len(traces["path_indices"]), "plot": str(plot.relative_to(output)),
                            "diagnostic_seconds": time.perf_counter() - started})
        except Exception as exc:
            # Plotting must not interrupt training or repeatedly retry at every update.
            print(f"Warning: could not save nu growth diagnostic for update {step}: {exc}",
                  file=sys.stderr, flush=True)
        finally:
            nu_diagnostic_seconds += time.perf_counter() - started

    def complete_validation():
        nonlocal best_score, best_iteration, pending_validation, validation_seconds
        nonlocal best_kl_score, best_kl_iteration, initial_metrics
        pending_validation = True
        samples, metrics, duration = evaluate(
            params, jax.random.fold_in(validation_key, iteration), data["val_raw"], cfg.n_val)
        validation_seconds += duration
        if initial_metrics is None:
            initial_metrics = metrics
        score = float(metrics["sliced_wasserstein1"])
        improved = score < best_score
        kl_score = metrics.get("averaged_kl")
        improved_kl = kl_score is not None and (best_kl_score is None or kl_score < best_kl_score)
        if improved:
            best_score, best_iteration = score, iteration
            checkpoint("best", checkpoint_extra={"validation_metrics": metrics})
        if improved_kl:
            best_kl_score, best_kl_iteration = float(kl_score), iteration
            checkpoint("best_kl", checkpoint_extra={"validation_metrics": metrics})
            if target is not None:
                np.savez_compressed(output / "best_validation_samples.npz",
                                    generated_normalized=samples, iteration=iteration)
        pending_validation = False
        try:
            checkpoint("latest")
        except (Exception, KeyboardInterrupt):
            pending_validation = True
            raise
        _log(log_path, {"event": "validation", "iteration": iteration,
                        "metrics": metrics, "generation_seconds": duration,
                        "selected_best": improved, "selected_best_kl": improved_kl,
                        **({"schedule": schedule_record(cfg, iteration)}
                           if cfg.schedule != "none" else {})})
        update_convergence()

    try:
        if target is not None:
            null_metrics = evaluate_samples(
                data["train_raw"], data["val_raw"], target=target, normalization_scale=R,
                density_bandwidth=evaluation_bandwidth(cfg),
                density_self_weight=cfg.density_self_weight, n_neighbor=0)
            atomic_json(output / "target_baseline.json", null_metrics)
        if args.resume:
            inherit_sample_history(Path(source).resolve().parent, output, iteration,
                                   history_reference, data_hash,
                                   sample_source="training", neighbors=cfg.neighbors,
                                   **projection_options)
            if best_saved is not None:
                copy_best("best", best_saved)
            if best_kl_saved is not None:
                copy_best("best_kl", best_kl_saved)
            if best_saved is not None:
                checkpoint("latest")
            if pending_validation or best_saved is None:
                complete_validation()
        elif not args.evaluate:
            complete_validation()

        if not args.evaluate:
            first_update = iteration + 1
            inherited_iterations = (
                read_sample_history(output)["iterations"]
                if (output / "sample_history.npz").exists() else np.empty(0, dtype=np.int64))
            plot_slots = _history_plot_slots(inherited_iterations, first_update,
                                             cfg.updates, cfg.eval_every)
            for step in inherited_iterations:
                if int(step) in plot_slots:
                    render_snapshot(int(step), plot_slots[int(step)])
            for iteration in range(iteration + 1, cfg.updates + 1):
                update_started = time.perf_counter()
                if cfg.schedule != "none":
                    cfg = scheduled_config(cfg, iteration)
                    engine.set_config(cfg)
                next_key, update_key = jax.random.split(key)
                capture = (iteration == first_update or iteration % cfg.eval_every == 0
                           or iteration == cfg.updates)
                snapshot = {} if capture else None
                alpha_options = ({"previous_sigma_mean": previous_sigma_mean}
                                 if normalize_alpha else {})
                response_options = ({"previous_response_mean": previous_response_mean}
                                    if center_response else {})
                gradient, diagnostics = engine.gradient(
                    params, update_key, normalized_train, snapshot=snapshot,
                    **alpha_options, **response_options)
                jax.block_until_ready(gradient)
                _finite_tree(jax, gradient, "gradient")
                clip_options = {}
                if cfg.update_clip_mode == "kl":
                    current_kl = float(diagnostics["training_kl"])
                    if not np.isfinite(current_kl):
                        raise FloatingPointError("Nonfinite training KL; no parameter update was applied")
                    clip_options["kl_budget"] = cfg.kl_clip_fraction * abs(current_kl)
                if cfg.optimizer == "adam":
                    candidate, next_optimizer_state, update_diagnostics = adam_update(
                        params, gradient, optimizer_state, cfg, **clip_options)
                else:
                    candidate, update_diagnostics = sgd_update(params, gradient, cfg, **clip_options)
                    next_optimizer_state = None
                diagnostics.update(update_diagnostics)
                if cfg.schedule != "none":
                    diagnostics["schedule"] = schedule_record(cfg, iteration)
                jax.block_until_ready(candidate)
                _finite_tree(jax, candidate, "parameters")
                if next_optimizer_state is not None:
                    _finite_tree(jax, next_optimizer_state, "optimizer state")
                diagnostics["initial_distribution"] = _initial_distribution_record(candidate, cfg)
                next_sigma_mean = previous_sigma_mean
                if normalize_alpha:
                    next_sigma_mean = float(diagnostics["sigma_mean"])
                    if not np.isfinite(next_sigma_mean) or next_sigma_mean <= 0:
                        raise FloatingPointError("Nonfinite or nonpositive mean diffusion")
                next_response_mean = previous_response_mean
                if center_response:
                    next_response_mean = float(diagnostics["response_coefficient_mean"])
                    if not np.isfinite(next_response_mean):
                        raise FloatingPointError("Nonfinite mean response coefficient")
                json.dumps(diagnostics, allow_nan=False)
                diagnose_nu = _nu_diagnostic_due(
                    cfg, diagnostics["gradient_norm"], iteration, last_nu_diagnostic_iteration)
                diagnostic_params = params if diagnose_nu else None
                duration = time.perf_counter() - update_started
                training_seconds += duration
                update_durations.append(duration)
                if snapshot is not None:
                    # These are the production paths before the parameter update.
                    initial_law = _initial_distribution_record(params, cfg)
                    grid_z = field_grid(snapshot=True)
                    drift, diffusion = snapshot_fields(
                        params, jnp.asarray(grid_z.reshape(-1, cfg.dim), dtype=cfg.dtype))
                    drift = np.asarray(jax.device_get(drift)).reshape(grid_z.shape)
                    diffusion = np.asarray(jax.device_get(diffusion)).reshape(grid_z.shape[:-1])
                    record_sample_history(
                        output, iteration, snapshot["generated_normalized"], history_reference,
                        data_hash=data_hash,
                        trajectories_normalized=snapshot["trajectories_normalized"],
                        selection_counts=snapshot["selection_counts"],
                        drift_grid_normalized=grid_z, drift_normalized=drift,
                        diffusion_normalized=diffusion,
                        initial_mean_normalized=initial_law["mean"], initial_s0=initial_law["s0"],
                        sample_source="training", neighbors=cfg.neighbors, **projection_options)
                params, key, optimizer_state = candidate, next_key, next_optimizer_state
                previous_sigma_mean = next_sigma_mean
                previous_response_mean = next_response_mean
                completed_iteration = iteration
                pending_validation = iteration % cfg.eval_every == 0 or iteration == cfg.updates
                _log(log_path, {"event": "update", "iteration": iteration,
                                "update_seconds": duration, "diagnostics": diagnostics},
                     display=iteration % cfg.log_every == 0)
                if diagnose_nu:
                    last_nu_diagnostic_iteration = iteration
                    render_nu_growth(iteration, diagnostic_params, update_key, diagnostics)
                if iteration in plot_slots:
                    render_snapshot(iteration, plot_slots[iteration])
                if pending_validation:
                    complete_validation()
        else:
            copy_best("best_kl" if best_kl_saved is not None else "best", saved)

        # Test samples are consulted only after all validation-based selection is complete.
        final_cfg = cfg
        test_results = {}
        selected_initial_distributions = {}
        for name, suffix in (("best", ""), ("best_kl", "_kl")):
            path = output / (name + ".npz")
            if not path.exists():
                continue
            selected = load_checkpoint(path)
            selected_params = jax.tree_util.tree_map(jnp.asarray, selected["params"])
            selected_iteration = selected["metadata"]["iteration"]
            selected_initial_distributions[name] = _initial_distribution_record(selected_params, cfg)
            if cfg.schedule != "none":
                # A selected model includes its bounds at selection, not just weights.
                cfg = scheduled_config(final_cfg, selected_iteration)
                engine.set_config(cfg)
            samples, test_metrics, generation_seconds = evaluate(
                selected_params, (test_key if target is not None else jax.random.fold_in(test_key, selected_iteration)),
                data["test_raw"], cfg.eval_samples)
            np.savez_compressed(output / f"test_samples{suffix}.npz", generated_raw=samples * R,
                                generated_normalized=samples)
            if name == "best_kl":
                grid_z = field_grid(samples, data["test"])
                grid = jnp.asarray(grid_z.reshape(-1, cfg.dim), dtype=cfg.dtype)
                drift, sigma = jax.jit(jax.vmap(lambda x: fields(selected_params, x, cfg)))(grid)
                drift, sigma = jax.device_get((drift, sigma))
                save_diagnostics(output / f"diagnostics{suffix}.png", samples, data["test"],
                                 grid_z, np.asarray(drift).reshape(grid_z.shape),
                                 np.asarray(sigma).reshape(grid_z.shape[:-1]), **projection_options)
            test_results[name] = (test_metrics, generation_seconds)
            _log(log_path, {"event": "test", "iteration": selected_iteration, "metrics": test_metrics,
                            "selection_metric": selected["metadata"]["selection_metric"],
                            **({"schedule": schedule_record(cfg, selected_iteration)}
                               if cfg.schedule != "none" else {})})
        cfg = final_cfg
        if cfg.schedule != "none":
            engine.set_config(cfg)
        test_metrics, generation_seconds = test_results.get("best", (None, None))
        kl_test_metrics, kl_generation_seconds = test_results.get("best_kl", (None, None))
        summary = {
            "best_iteration": best_iteration if np.isfinite(best_score) else None,
            "best_validation_sliced_wasserstein1": best_score if np.isfinite(best_score) else None,
            "best_kl_iteration": best_kl_iteration, "best_validation_averaged_kl": best_kl_score,
            "test_metrics": test_metrics, "test_metrics_best_kl": kl_test_metrics,
            "initial_distribution_best_kl": selected_initial_distributions.get("best_kl"),
            "parameter_count": parameter_count(params),
            "optimizer": cfg.optimizer, "learning_rate": cfg.learning_rate,
            "training_update_seconds_this_run": training_seconds,
            "validation_generation_seconds_this_run": validation_seconds,
            "nu_diagnostic_seconds_this_run": nu_diagnostic_seconds,
            "test_generation_seconds": generation_seconds,
            "test_seconds_per_generated_sample": (generation_seconds / cfg.eval_samples
                                                  if generation_seconds is not None else None),
            "test_generation_seconds_best_kl": kl_generation_seconds,
            "test_seconds_per_generated_sample_best_kl": (kl_generation_seconds / cfg.eval_samples
                                                          if kl_generation_seconds is not None else None),
            "wall_seconds_this_run": time.perf_counter() - started,
            "timing_note": "Synchronized; includes JIT compilation on first use. Adaptive-alpha updates include pilots.",
            "device_memory": _device_memory(jax), "data_fingerprint": data_hash,
        }
        if cfg.schedule != "none":
            summary["schedule"] = schedule_record(cfg, iteration)
        if target is not None:
            summary.update(
                dimension=cfg.dim, updates_completed=iteration,
                initial_validation_metrics=initial_metrics, target_vs_target_metrics=null_metrics,
                training_update_seconds=training_seconds,
                median_update_seconds_excluding_first=(
                    float(np.median(update_durations[1:])) if len(update_durations) > 1 else None),
                wall_seconds=time.perf_counter() - started)
        atomic_json(output / "summary.json", summary)
        update_convergence()
        print(_screen_message({"event": "summary", **summary}), flush=True)
        print(f"Saved results to {output}", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        if not args.evaluate and completed_iteration != latest_checkpoint_iteration:
            try:
                # Never label a failed, in-progress update as completed. Keep
                # its last accepted parameters, key, optimizer and lagged means.
                checkpoint("latest", checkpoint_iteration=completed_iteration)
            except Exception as checkpoint_exc:
                print(f"Warning: could not save recovery checkpoint: {checkpoint_exc}",
                      file=sys.stderr, flush=True)
        _log(log_path, {"event": "failure", "iteration": iteration,
                        "error_type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if not args.evaluate and (output / "sample_history.npz").exists():
            try:
                save_sample_history_plot(output)
            except Exception as exc:
                # Plotting must not hide a training error or prevent checkpoint recovery.
                print(f"Warning: could not save sample history plots: {exc}",
                      file=sys.stderr, flush=True)

    return output


def main(argv=None):
    args = _arguments(argv)
    try:
        run(args)
    except (ValueError, FloatingPointError, FileNotFoundError, FileExistsError, PermissionError) as exc:
        print(f"DKgen stopped: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

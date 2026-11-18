"""Train matched EDM or VP score models with one shared 2D/8D run lifecycle."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .checkpoint import atomic_json, data_fingerprint
from .selection import load_resume_bests
from .source_snapshot import save_source_snapshot
from .train import (_create_output_directory, _device_memory, _environment,
                    _screen_message, _update_convergence_plot)


def _arguments(argv=None, *, default_dim=2, default_method="edm"):
    from .score_setup import arguments
    return arguments(argv, default_dim=default_dim, default_method=default_method)


def _configuration(args, saved):
    from .score_setup import _configuration as configure
    return configure(args, saved)


def _adopt_data_sizes(cfg, data, overrides):
    from .score_setup import _adopt_data_sizes as adopt
    return adopt(cfg, data, overrides)


def _log(path, record, *, display=True):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
    if not display:
        return
    if record["event"] == "update":
        diagnostics = record["diagnostics"]
        clipping = diagnostics["gradient_clip_scale"]
        clip_text = f" | clip={clipping:.3g}" if clipping < 1 else ""
        loss_key = "edm_loss" if "edm_loss" in diagnostics else "noise_mse"
        label = "EDM_loss" if loss_key == "edm_loss" else "noise_MSE"
        rate_text = (f" | lr={diagnostics['learning_rate']:.3g}"
                     if "learning_rate" in diagnostics else "")
        print(f"Step {record['iteration']:5d} | {label}={diagnostics[loss_key]:.5g}"
              f" | grad={diagnostics['gradient_norm']:.4g}{clip_text}"
              f"{rate_text} | {record['update_seconds']:.3f}s", flush=True)
    else:
        print(_screen_message(record), flush=True)


def run(args):
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Output directory already exists: {args.output}")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from .score_backend import backend_for, checkpoint_metadata
    from .score_setup import prepare_experiment
    source = args.resume or args.warm_start or args.evaluate
    metadata = checkpoint_metadata(source) if source else None
    method = {"edm": "edm", "score_vp": "vp"}[metadata["model_kind"]] if metadata else args.method
    if getattr(args, "_method_explicit", False) and args.method != method:
        raise ValueError("The requested method differs from the checkpoint's saved method")
    args.method = method
    preview_cfg, _, _ = _configuration(args, {"metadata": metadata} if metadata else None)
    import jax
    jax.config.update("jax_enable_x64", preview_cfg.dtype == "float64")
    backend = backend_for(method)
    saved = backend.load_checkpoint(source) if source else None
    cfg, dk_cfg, data, target, provenance = prepare_experiment(args, saved)
    overrides = provenance.get("cli_overrides", {})
    density_selection_reset = bool(args.resume and any(
        getattr(cfg, name) != saved["metadata"]["config"][name]
        for name in ("density_bandwidth", "density_self_weight")))
    sampler_selection_reset = bool(args.resume and
                                   cfg.sampling_steps != saved["metadata"]["config"]["sampling_steps"])
    kl_selection_reset = density_selection_reset or sampler_selection_reset
    if args.evaluate and saved["metadata"]["role"] != "best":
        raise ValueError("--evaluate requires a best checkpoint selected on validation")

    import jax.numpy as jnp
    from .data import save_data
    from .score_state import init_optimizer, make_training_step
    from .metrics import evaluate_samples
    from .plots import diagnostic_grid, projection_frame, projection_grids
    from .score_plots import save_score_diagnostics
    from .sample_history import (history_plot_slots, inherit_sample_history, read_sample_history,
                                 record_sample_history, save_sample_history_plot)

    data_hash = data_fingerprint(data)
    if saved and data_hash != saved["metadata"]["data_fingerprint"]:
        raise ValueError("Saved dataset fingerprint does not match the checkpoint")

    best_saved = best_kl_saved = None
    if args.resume and not sampler_selection_reset:
        best_saved, best_kl_saved = load_resume_bests(
            source, saved, data_hash, checkpoint_loader=backend.load_checkpoint)
        if kl_selection_reset:
            best_kl_saved = None
    elif args.evaluate:
        if saved["metadata"]["selection_metric"] == "validation_averaged_kl":
            best_kl_saved = saved
        else:
            best_saved = saved

    output = _create_output_directory(args.output)
    print(f"Results: {output}", flush=True)
    print(f"{cfg.dim}D {method.upper()} | {cfg.parameter_count:,} parameters "
          f"(DK target {cfg.target_parameter_count:,}) | "
          f"evaluation bandwidth={cfg.density_bandwidth:g}", flush=True)
    if args.warm_start:
        print(f"Warm start: {source}; optimizer and update count reset", flush=True)
    if density_selection_reset:
        print(f"Density estimator changed: bandwidth={cfg.density_bandwidth:g}, "
              f"self weight={cfg.density_self_weight:g}; "
              "restarting best-KL selection from the resumed state.", flush=True)
    if sampler_selection_reset:
        print(f"Sampling steps changed to {cfg.sampling_steps}; restarting both validation "
              "selections and sample history from the resumed state.", flush=True)
    started = time.perf_counter()
    source_snapshot = save_source_snapshot(output)
    env = _environment(jax)
    save_data(output / "data.npz", data)
    if target is not None:
        np.savez_compressed(output / "target.npz", **target)
    atomic_json(output / "config.json", asdict(cfg))
    atomic_json(output / "run.json", {
        "model_kind": backend.model_kind, "method": method,
        "config": asdict(cfg), "cli_overrides": overrides,
        "mode": ("evaluate" if args.evaluate else "resume" if args.resume
                 else "warm_start" if args.warm_start else "train"),
        "target": ("correlated_four_mode_gaussian_mixture" if target is not None
                   else "two_dimensional_gaussian_mixture"),
        "comparison": provenance, "parameter_count": cfg.parameter_count,
        "target_dk_parameter_count": cfg.target_parameter_count,
        "evaluation_density_bandwidth": cfg.density_bandwidth,
        "kl_selection_reset": kl_selection_reset,
        "sampler_selection_reset": sampler_selection_reset,
        "source_checkpoint": str(Path(source).resolve()) if source else None,
        "source_data": provenance.get("source_data"),
        "matched_dk_config": asdict(dk_cfg) if dk_cfg is not None else None,
        "matched_dk_config_path": str(args.match_dk_config.resolve()) if args.match_dk_config else None,
        "source_snapshot": source_snapshot, "data_fingerprint": data_hash, "environment": env,
        "optimizer": "Adam", **backend.description(cfg),
        "evaluation_weights": "EMA" if cfg.ema_decay else "current",
        "engineering_defaults_are_tuned": False,
    })
    log_path = output / "history.jsonl"
    engine = backend.engine_type(cfg)
    training_step = make_training_step(engine, cfg)
    if saved is None:
        key, init_key = jax.random.split(jax.random.PRNGKey(cfg.seed))
        params = backend.init_params(init_key, cfg)
        ema_params, optimizer = params, init_optimizer(params)
        iteration, pending_validation = 0, False
    else:
        params = jax.tree_util.tree_map(jnp.asarray, saved["params"])
        ema_params = jax.tree_util.tree_map(jnp.asarray, saved["ema_params"])
        key = jnp.asarray(saved["key"])
        if args.warm_start:
            optimizer = init_optimizer(params)
            ema_params = params
            iteration, pending_validation = 0, False
        else:
            optimizer = jax.tree_util.tree_map(jnp.asarray, saved["optimizer"])
            optimizer["step"] = jnp.asarray(optimizer["step"], dtype=jnp.int32)
            iteration = saved["metadata"]["iteration"]
            pending_validation = saved["metadata"].get("pending_validation", False) or kl_selection_reset
    best_score, best_iteration = float("inf"), 0
    best_kl_score = best_kl_iteration = None
    if best_saved is not None:
        best_score = best_saved["metadata"]["best_validation"]
        best_iteration = best_saved["metadata"]["iteration"]
    if best_kl_saved is not None:
        best_kl_score = best_kl_saved["metadata"]["best_validation"]
        best_kl_iteration = best_kl_saved["metadata"]["iteration"]
    normalized_train = jnp.asarray(data["train"], dtype=cfg.dtype)
    R = float(data["R"])
    projection_center, projection_basis = projection_frame(cfg.dim, target)
    projection_options = ({} if cfg.dim == 2 else {
        "projection_center": projection_center, "projection_basis": projection_basis})
    validation_key = jax.random.fold_in(jax.random.PRNGKey(cfg.seed), 17001)
    test_key = jax.random.fold_in(jax.random.PRNGKey(cfg.seed), 17002)
    training_seconds = validation_seconds = 0.0
    update_durations = []
    initial_metrics = null_metrics = None
    first_update = iteration + 1
    inherited_iterations = []
    if args.resume and not sampler_selection_reset:
        inherit_sample_history(Path(source).resolve().parent, output, iteration,
                               data["train"], data_hash, sample_source="score",
                               **projection_options)
        if (output / "sample_history.npz").exists():
            inherited_iterations = read_sample_history(output)["iterations"]
    history_start = iteration if sampler_selection_reset else first_update
    history_slots = history_plot_slots(inherited_iterations, history_start, cfg.updates,
                                       cfg.eval_every)
    completed_iteration = iteration
    latest_dirty = True

    def checkpoint(name, metrics=None):
        nonlocal latest_dirty
        kl_selected = name == "best_kl"
        extra = {"best_kl_iteration": best_kl_iteration,
                 "best_validation_averaged_kl": best_kl_score, "comparison": provenance}
        if target is not None:
            extra["rotation_seed"] = int(target["rotation_seed"])
        if metrics is not None:
            extra["validation_metrics"] = metrics
        backend.save_checkpoint(
            output / (name + ".npz"), params=params, ema_params=ema_params,
            optimizer=optimizer, key=key, iteration=completed_iteration,
            best_iteration=best_kl_iteration if kl_selected else best_iteration,
            best_validation=best_kl_score if kl_selected else best_score,
            config=cfg, data_hash=data_hash, role="latest" if name == "latest" else "best",
            pending_validation=pending_validation if name == "latest" else False,
            selection_metric="validation_averaged_kl" if kl_selected else "validation_sliced_wasserstein1",
            extra=extra)
        if name == "latest":
            latest_dirty = False

    def copy_best(name, state):
        metadata = state["metadata"]
        extra = dict(metadata.get("extra", {}))
        if kl_selection_reset:
            # The inherited SW1 winner is still valid, but its embedded KL
            # metrics and selection pointers used the previous estimator.
            extra.pop("validation_metrics", None)
            extra.update(best_kl_iteration=None, best_validation_averaged_kl=None)
        backend.save_checkpoint(
            output / (name + ".npz"), params=state["params"], ema_params=state["ema_params"],
            optimizer=state["optimizer"], key=state["key"], iteration=metadata["iteration"],
            best_iteration=metadata["iteration"], best_validation=metadata["best_validation"],
            config=cfg, data_hash=data_hash, role="best",
            selection_metric=metadata["selection_metric"], extra=extra)

    def evaluate(evaluation_params, sample_key, reference, count):
        generation_started = time.perf_counter()
        samples = engine.sample(evaluation_params, sample_key, count)
        samples = np.asarray(jax.block_until_ready(samples))
        duration = time.perf_counter() - generation_started
        if not np.isfinite(samples).all():
            raise FloatingPointError("Nonfinite score-model samples; evaluation stopped")
        metrics = evaluate_samples(
            (np.asarray(samples, dtype=np.float64) * R if target is not None else samples * R),
            reference, mixture_modes=cfg.mixture_modes,
            n_neighbor=0 if target is not None else cfg.n_neighbor,
            density_bandwidth=cfg.density_bandwidth,
            density_self_weight=cfg.density_self_weight,
            normalization_scale=R, **({"target": target} if target is not None else {}))
        json.dumps(metrics, allow_nan=False)
        return samples, metrics, duration

    def render_history(*, step=None, slot=None):
        try:
            save_sample_history_plot(output, iteration=step, slot=slot)
        except Exception as exc:
            # Plot failures must not discard a completed update or its checkpoint.
            print(f"Warning: could not save sample history plot: {exc}",
                  file=sys.stderr, flush=True)

    def capture_samples(samples):
        record_sample_history(output, iteration, samples, data["train"],
                              data_hash=data_hash, sample_source="score", **projection_options)
        if iteration in history_slots:
            render_history(step=iteration, slot=history_slots[iteration])

    def update_convergence():
        _update_convergence_plot(log_path)

    def complete_validation():
        nonlocal best_score, best_iteration, best_kl_score, best_kl_iteration
        nonlocal pending_validation, validation_seconds, initial_metrics, latest_dirty
        pending_validation = True
        latest_dirty = True
        validation_started = time.perf_counter()
        samples, metrics, duration = evaluate(
            ema_params, jax.random.fold_in(validation_key, iteration), data["val_raw"], cfg.n_val)
        total_duration = time.perf_counter() - validation_started
        validation_seconds += total_duration
        if initial_metrics is None:
            initial_metrics = metrics
        value = float(metrics["sliced_wasserstein1"])
        improved = value < best_score
        kl_value = metrics.get("averaged_kl")
        improved_kl = kl_value is not None and (best_kl_score is None or kl_value < best_kl_score)
        if improved:
            best_score, best_iteration = value, iteration
            checkpoint("best", metrics)
        if improved_kl:
            best_kl_score, best_kl_iteration = float(kl_value), iteration
            checkpoint("best_kl", metrics)
            np.savez_compressed(output / "best_validation_samples.npz",
                                generated_normalized=samples, iteration=iteration)
        pending_validation = False
        try:
            checkpoint("latest")
        except (Exception, KeyboardInterrupt):
            pending_validation = True
            raise
        _log(log_path, {"event": "validation", "iteration": iteration, "metrics": metrics,
                        "generation_seconds": duration, "validation_seconds": total_duration,
                        "selected_best": improved, "selected_best_kl": improved_kl})
        if iteration >= first_update or sampler_selection_reset:
            capture_samples(samples)
        update_convergence()

    try:
        for step in inherited_iterations:
            if step in history_slots:
                render_history(step=int(step), slot=history_slots[step])
        if target is not None:
            null_metrics = evaluate_samples(
                data["train_raw"], data["val_raw"], target=target, normalization_scale=R,
                n_neighbor=0, density_bandwidth=cfg.density_bandwidth,
                density_self_weight=cfg.density_self_weight)
            atomic_json(output / "target_baseline.json", null_metrics)
        for name, state in (("best", best_saved), ("best_kl", best_kl_saved)):
            if state is not None:
                copy_best(name, state)
        if args.resume and best_saved is not None:
            checkpoint("latest")
        if not args.evaluate and (saved is None or args.warm_start or pending_validation or best_saved is None):
            complete_validation()
        if not args.evaluate:
            for iteration in range(first_update, cfg.updates + 1):
                update_started = time.perf_counter()
                next_key, update_key = jax.random.split(key)
                (candidate, candidate_optimizer, candidate_ema, loss, diagnostics,
                 gradient_finite, state_finite) = training_step(
                    params, optimizer, ema_params, update_key, normalized_train)
                loss, diagnostics, gradient_finite, state_finite = jax.device_get(
                    (loss, diagnostics, gradient_finite, state_finite))
                if not bool(gradient_finite):
                    raise FloatingPointError("Nonfinite score loss/gradient; stopping without applying this update")
                if not bool(state_finite):
                    raise FloatingPointError("Nonfinite score parameters/optimizer; stopping without applying this update")
                diagnostics = {name: float(value) for name, value in diagnostics.items()}
                diagnostics[backend.loss_key] = float(loss)
                duration = time.perf_counter() - update_started
                training_seconds += duration
                update_durations.append(duration)
                params, optimizer, ema_params, key = candidate, candidate_optimizer, candidate_ema, next_key
                completed_iteration = iteration
                pending_validation = iteration % cfg.eval_every == 0 or iteration == cfg.updates
                latest_dirty = True
                _log(log_path, {"event": "update", "iteration": iteration,
                                "update_seconds": duration, "diagnostics": diagnostics},
                     display=iteration % cfg.log_every == 0)
                if pending_validation:
                    complete_validation()
                elif iteration == first_update:
                    samples = np.asarray(engine.sample(
                        ema_params, jax.random.fold_in(validation_key, iteration), cfg.n_val))
                    if not np.isfinite(samples).all():
                        raise FloatingPointError("Nonfinite score-model snapshot samples")
                    capture_samples(samples)

        # Each validation-selected state is evaluated once on the untouched test split.
        test_results = {}
        for name, suffix in (("best", ""), ("best_kl", "_kl")):
            path = output / (name + ".npz")
            if not path.exists():
                continue
            selected = backend.load_checkpoint(path)
            selected_params = jax.tree_util.tree_map(jnp.asarray, selected["ema_params"])
            selected_iteration = selected["metadata"]["iteration"]
            samples, test_metrics, generation_seconds = evaluate(
                selected_params, (test_key if target is not None
                                  else jax.random.fold_in(test_key, selected_iteration)),
                data["test_raw"], cfg.eval_samples)
            np.savez_compressed(output / f"test_samples{suffix}.npz",
                                generated_normalized=samples, generated_raw=samples * R,
                                iteration=selected_iteration)
            if name == "best_kl":
                grid_z = (diagnostic_grid(samples, data["test"]) if cfg.dim == 2 else
                          projection_grids(samples, data["test"], center=projection_center,
                                           basis=projection_basis))
                grid = jnp.asarray(grid_z.reshape(-1, cfg.dim), dtype=cfg.dtype)
                times = backend.diagnostic_levels(cfg)
                field = jax.jit(jax.vmap(
                    lambda x, level: backend.score_field(selected_params, x, level, cfg),
                    in_axes=(0, None)))
                scores = np.stack([np.asarray(field(grid, t)).reshape(grid_z.shape) for t in times])
                save_score_diagnostics(output / f"diagnostics{suffix}.png", samples,
                                       data["test"], grid_z, scores, times,
                                       level_name=backend.diagnostic_level_name,
                                       method_name=method.upper(), **projection_options)
            test_results[name] = (test_metrics, generation_seconds)
            _log(log_path, {"event": "test", "iteration": selected_iteration, "metrics": test_metrics,
                            "selection_metric": selected["metadata"]["selection_metric"]})
        test_metrics, generation_seconds = test_results.get("best", (None, None))
        kl_metrics, kl_seconds = test_results.get("best_kl", (None, None))
        summary = {
            "model_kind": backend.model_kind, "method": method,
            "dimension": cfg.dim, "updates_completed": iteration,
            "parameter_count": cfg.parameter_count,
            "target_dk_parameter_count": cfg.target_parameter_count,
            "parameter_count_relative_difference": cfg.parameter_count / cfg.target_parameter_count - 1,
            "best_iteration": best_iteration if np.isfinite(best_score) else None,
            "best_validation_sliced_wasserstein1": best_score if np.isfinite(best_score) else None,
            "best_kl_iteration": best_kl_iteration, "best_validation_averaged_kl": best_kl_score,
            "test_metrics": test_metrics, "test_metrics_best_kl": kl_metrics,
            "training_update_seconds_this_run": training_seconds,
            "validation_seconds_this_run": validation_seconds,
            "test_generation_seconds": generation_seconds,
            "test_seconds_per_generated_sample": (generation_seconds / cfg.eval_samples
                                                  if generation_seconds is not None else None),
            "test_generation_seconds_best_kl": kl_seconds,
            "test_seconds_per_generated_sample_best_kl": (kl_seconds / cfg.eval_samples
                                                          if kl_seconds is not None else None),
            "sampling_steps": cfg.sampling_steps,
            "network_evaluations_per_generated_sample": backend.network_evaluations(cfg),
            "evaluation_density_bandwidth": cfg.density_bandwidth,
            "initial_validation_metrics": initial_metrics,
            "target_vs_target_metrics": null_metrics, "comparison": provenance,
            "training_update_seconds": training_seconds,
            "median_update_seconds_excluding_first": (
                float(np.median(update_durations[1:])) if len(update_durations) > 1 else None),
            "validation_seconds": validation_seconds,
            "evaluation_weights": "EMA" if cfg.ema_decay else "current",
            "learning_rate_schedule": cfg.learning_rate_schedule,
            "learning_rate_start": cfg.learning_rate,
            "learning_rate_final": (cfg.learning_rate_final
                                    if cfg.learning_rate_schedule != "constant" else cfg.learning_rate),
            "learning_rate_decay_start": cfg.learning_rate_decay_start,
            "learning_rate_decay_end": cfg.learning_rate_decay_end,
            "sampler_selection_reset": sampler_selection_reset,
            "wall_seconds_this_run": time.perf_counter() - started,
            "timing_note": "Synchronized; first use includes JIT compilation. Training samples one noise level per datum.",
            "device_memory": _device_memory(jax), "data_fingerprint": data_hash,
        }
        atomic_json(output / "summary.json", summary)
        update_convergence()
        print(_screen_message({"event": "summary", **summary}), flush=True)
        print(f"Saved results to {output}", flush=True)
    except Exception as exc:
        failure = {"event": "failure", "iteration": iteration,
                   "error_type": type(exc).__name__, "message": str(exc)}
        atomic_json(output / "failure.json", failure)
        _log(log_path, failure)
        raise
    finally:
        if not args.evaluate and latest_dirty:
            try:
                # An interrupted update must not label its last accepted state
                # with the unfinished update's iteration number.
                checkpoint("latest")
            except Exception as exc:
                print(f"Warning: could not save recovery checkpoint: {exc}",
                      file=sys.stderr, flush=True)
        if not args.evaluate and (output / "sample_history.npz").exists():
            render_history()
    return output


def main(argv=None, *, default_dim=2, default_method="edm"):
    try:
        args = _arguments(argv, default_dim=default_dim, default_method=default_method)
        run(args)
    except (ValueError, FloatingPointError, FileNotFoundError, FileExistsError, PermissionError) as exc:
        print(f"Score baseline stopped: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

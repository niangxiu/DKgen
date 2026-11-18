"""Save the actual covector growth along selected production sample paths."""

from __future__ import annotations

import json
import math
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np

from .checkpoint import _atomic_write
from .sample_history import _drift_field
from .plots import validate_projection


def covector_growth(covectors, dt):
    """Return stable norms, log norms, and per-step log growth, with validity.

    A rate requires two finite, nonzero covectors. Zero and nonfinite states
    remain explicit missing values instead of being replaced by an epsilon.
    Norms themselves can overflow float64; their log norms can still be valid.
    """
    values = np.asarray(covectors)
    if (values.dtype.kind not in "fiu" or values.ndim != 3
            or values.shape[0] < 1 or values.shape[1] < 2 or values.shape[2] < 1):
        raise ValueError("Covectors must have shape (paths, at least 2 times, dimension)")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    values = values.astype(np.float64)
    finite = np.isfinite(values).all(axis=-1)
    scale = np.max(np.abs(values), axis=-1)
    positive = finite & (scale > 0)
    scaled = np.zeros_like(values)
    np.divide(values, scale[..., None], out=scaled, where=positive[..., None])
    length = np.sqrt(np.sum(scaled * scaled, axis=-1))
    norms = np.full(scale.shape, np.nan)
    log_norms = np.full(scale.shape, np.nan)
    zero = finite & (scale == 0)
    norms[zero], log_norms[zero] = 0.0, -np.inf
    with np.errstate(over="ignore", under="ignore"):
        norms[positive] = scale[positive] * length[positive]
    log_norms[positive] = np.log(scale[positive]) + np.log(length[positive])
    valid = positive[:, :-1] & positive[:, 1:]
    rates = np.full((values.shape[0], values.shape[1] - 1), np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        np.divide(log_norms[:, 1:] - log_norms[:, :-1], dt, out=rates, where=valid)
    valid &= np.isfinite(rates)
    rates[~valid] = np.nan
    return norms, log_norms, rates, valid


def _path_segments(trajectories):
    """Flatten in path-major order, matching rates[path, step].ravel()."""
    return np.stack((trajectories[:, :-1], trajectories[:, 1:]), axis=2).reshape(-1, 2, 2)


def _color_limit(rates, valid):
    absolute = np.abs(rates[valid])
    if not len(absolute):
        return 1.0
    # Selecting an observed value avoids interpolating near the float64 limit.
    limit = float(np.quantile(absolute, 0.98, method="nearest"))
    if limit == 0:
        limit = float(np.max(absolute))
    return limit if limit > 0 else 1.0


def _optional_drift_field(grid, drift, *, projection_center=None, projection_basis=None):
    if (grid is None) != (drift is None):
        raise ValueError("Drift grid and vectors must be provided together")
    return ((None, None) if grid is None else _drift_field(
        grid, drift, projection_center=projection_center, projection_basis=projection_basis))


def _draw_drift_contours(spatial, grid, drift):
    if grid is None:
        label = "Drift field not recorded"
    else:
        magnitude = np.hypot.reduce(drift.astype(np.float64), axis=-1)
        low, high = float(np.min(magnitude)), float(np.max(magnitude))
        if low == high:
            label = rf"Drift magnitude $\|F\| = {low:.4g}$ (constant)"
        else:
            levels = MaxNLocator(nbins=7).tick_values(low, high)
            levels = levels[(levels > low) & (levels < high)]
            contours = spatial.contour(grid[..., 0], grid[..., 1], magnitude,
                                       levels=levels, colors="0.3", linewidths=0.55,
                                       alpha=0.55, zorder=4)
            spatial.clabel(contours, inline=True, fontsize=7, fmt="%g")
            label = r"Contours: drift magnitude $\|F\|$"
    spatial.text(0.02, 0.02, label, transform=spatial.transAxes, fontsize=7,
                 ha="left", va="bottom", color="0.25", zorder=6,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2})


def _figure(trajectories, norms, rates, valid, *, dt, iteration,
            gradient_norm, update_clip_scale, color_limit, reference, lambda_hat,
            drift_grid_normalized=None, drift_normalized=None,
            projection_center=None, projection_basis=None):
    dimension = trajectories.shape[-1]
    center, basis = validate_projection(dimension, projection_center, projection_basis)
    drift_grid, drift = _optional_drift_field(
        drift_grid_normalized, drift_normalized, projection_center=center, projection_basis=basis)
    all_trajectories = (trajectories - center) @ basis
    all_reference = None if reference is None else (reference - center) @ basis
    if drift_grid is not None and drift_grid.ndim == 3:
        drift_grid, drift = drift_grid[None], drift[None]
    pairs = dimension // 2
    figure = Figure(figsize=(12.5, 5.4 * pairs), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(pairs, 2, squeeze=False, gridspec_kw={"width_ratios": [1.05, 1.0]})
    for pair, (spatial, growth) in enumerate(axes):
        trajectories = all_trajectories[..., 2 * pair:2 * pair + 2]
        reference = None if all_reference is None else all_reference[:, 2 * pair:2 * pair + 2]
        visible_grid = (None if drift_grid is None else
                        ((drift_grid[pair] - center) @ basis)[..., 2 * pair:2 * pair + 2])
        visible_drift = None if drift is None else drift[pair]
        if reference is not None:
            spatial.scatter(reference[:, 0], reference[:, 1], s=5, color="tab:orange",
                            alpha=0.14, linewidths=0, label="Training data", zorder=0)
        segments = _path_segments(trajectories)
        flat_valid, flat_rates = valid.ravel(), rates.ravel()
        if np.any(~flat_valid):
            spatial.add_collection(LineCollection(segments[~flat_valid], colors="0.65",
                                                  linewidths=0.5, alpha=0.12, zorder=1))
        color_norm = Normalize(-color_limit, color_limit, clip=True)
        nonpositive = flat_valid & (flat_rates <= 0)
        if np.any(nonpositive):
            background = LineCollection(segments[nonpositive], cmap="coolwarm",
                                        norm=color_norm, linewidths=0.55, alpha=0.15, zorder=2)
            background.set_array(flat_rates[nonpositive])
            spatial.add_collection(background)
        # Draw positive growth last, in increasing order, to keep the strongest
        # red segments visible. Only display order changes; archive order is intact.
        positive_indices = np.flatnonzero(flat_valid & (flat_rates > 0))
        positive_indices = positive_indices[np.argsort(flat_rates[positive_indices], kind="stable")]
        if len(positive_indices):
            foreground = LineCollection(segments[positive_indices], cmap="coolwarm",
                                        norm=color_norm, linewidths=0.65, alpha=1.0, zorder=3)
            foreground.set_array(flat_rates[positive_indices])
            spatial.add_collection(foreground)
        endpoints = trajectories[:, -1]
        spatial.scatter(endpoints[:, 0], endpoints[:, 1], s=12, facecolors="none",
                        edgecolors="0.25", linewidths=0.5, alpha=0.2, label="Path endpoint", zorder=2.5)
        _draw_drift_contours(spatial, visible_grid, visible_drift)
        # Match the sample-history view and keep it fixed across updates.
        spatial.set(xlim=(-2, 2), ylim=(-2, 2),
                    xticks=(-2, -1, 0, 1, 2), yticks=(-2, -1, 0, 1, 2))
        spatial.set_aspect("equal", adjustable="box")
        spatial.set(xlabel=r"$z_1$" if dimension == 2 else f"Projected coordinate {2 * pair + 1}",
                    ylabel=r"$z_2$" if dimension == 2 else f"Projected coordinate {2 * pair + 2}",
                    title=f"{len(trajectories)} production paths: actual covector growth"
                          + (f"; projection {pair + 1}" if dimension > 2 else ""))
        spatial.legend(loc="upper right", fontsize=8, framealpha=0.85)
        # Keep the numeric color key opaque, independent of segment transparency.
        colorbar = figure.colorbar(ScalarMappable(norm=color_norm, cmap="coolwarm"),
                                  ax=spatial, orientation="horizontal", pad=0.06,
                                  fraction=0.075, extend="both")
        colorbar.set_label(r"$\log(\|\nu_{n+1}\|/\|\nu_n\|)/\Delta t$  (inverse time)", fontsize=10)
        time = np.arange(norms.shape[1], dtype=np.float64) * dt
        visible_norms = np.where(np.isfinite(norms) & (norms > 0), norms, np.nan)
        for path in visible_norms:
            growth.plot(time, path, color="tab:blue", alpha=0.2, linewidth=0.7)
        growth.set_yscale("log")
        if not np.isfinite(visible_norms).any():
            growth.set_ylim(1e-2, 1)
            growth.text(0.5, 0.5, "No finite positive covector norms", transform=growth.transAxes,
                        ha="center", va="center")
        growth.set(xlabel="SDE time", ylabel=r"$\|\nu_n\|$ (log scale)",
                   title="Covector magnitude along the same paths")
        growth.grid(alpha=0.2, which="major")
    saturated = int(np.count_nonzero(valid & (np.abs(rates) > color_limit)))
    masked = int(np.count_nonzero(~valid))
    info = (f"Color range: 98th percentile of |rate|; {saturated:,} saturated segments.\n"
            f"{masked:,} undefined segments are gray (zero/nonfinite norm or rate).\n"
            "Actual covector includes the additive kernel term.")
    omitted = int(np.count_nonzero(~np.isfinite(visible_norms)))
    if omitted:
        info += f"  Norm plot omits {omitted:,} zero/nonfinite points."
    figure.supxlabel(info, fontsize=8, ha="center")
    title = (f"Before update {iteration} | raw gradient norm = {gradient_norm:.5g}"
             f" | update norm ratio = {update_clip_scale:.5g}")
    if lambda_hat is not None:
        title += f" | pilot growth = {lambda_hat:.4g}"
    figure.suptitle(title, fontsize=11)
    return figure


def save_nu_growth(
    output_directory, *, iteration, trajectories_normalized, covectors,
    path_indices, dt, gradient_norm, update_clip_scale, lambda_hat=None,
    data_hash, reference_normalized=None, update_key=None, alpha=None,
    drift_grid_normalized=None, drift_normalized=None,
    projection_center=None, projection_basis=None,
):
    """Write paired PNG/NPZ diagnostics without advancing the training RNG.

    Positions, covectors, and any drift field must use the same pre-update model.
    Colors use a symmetric percentile range; the archive retains uncapped rates.
    """
    trajectories = np.asarray(trajectories_normalized)
    values = np.asarray(covectors)
    if (trajectories.dtype.kind not in "fiu" or trajectories.ndim != 3
            or trajectories.shape[0] < 1 or trajectories.shape[1] < 2
            or trajectories.shape[2] < 2 or trajectories.shape[2] % 2
            or not np.isfinite(trajectories).all()):
        raise ValueError("Trajectories must have finite shape (paths, at least 2 times, even dimension >= 2)")
    dimension = trajectories.shape[-1]
    center, basis = validate_projection(dimension, projection_center, projection_basis)
    if values.shape != trajectories.shape:
        raise ValueError("Covectors and trajectories must have identical shapes")
    indices = np.asarray(path_indices)
    if (indices.dtype.kind not in "iu" or indices.shape != (len(trajectories),)
            or np.any(indices < 0) or len(np.unique(indices)) != len(indices)):
        raise ValueError("Path indices must be unique nonnegative integers, one per path")
    if (isinstance(iteration, (bool, np.bool_))
            or not isinstance(iteration, (int, np.integer)) or iteration < 0):
        raise ValueError("Iteration must be a nonnegative integer")
    for name, number in (("gradient_norm", gradient_norm),
                         ("update_clip_scale", update_clip_scale)):
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if update_clip_scale > 1:
        raise ValueError("update_clip_scale must not exceed one")
    if lambda_hat is not None and not math.isfinite(lambda_hat):
        raise ValueError("lambda_hat must be finite when provided")
    reference = None if reference_normalized is None else np.asarray(reference_normalized)
    if reference is not None and (reference.dtype.kind not in "fiu" or reference.ndim != 2
                                  or reference.shape[1] != dimension or not np.isfinite(reference).all()):
        raise ValueError("Training reference must have finite shape (data, dimension)")
    drift_grid, drift = _optional_drift_field(
        drift_grid_normalized, drift_normalized, projection_center=center, projection_basis=basis)
    if drift_grid is not None and drift_grid.shape[-1] != dimension:
        raise ValueError("Drift grid and trajectories must have the same dimension")
    norms, log_norms, rates, valid = covector_growth(values, dt)
    limit = _color_limit(rates, valid)
    metadata = {
        "schema_version": 1, "iteration": int(iteration), "state": "before_update",
        "sample_source": "production_training_paths", "data_hash": str(data_hash),
        "dt": float(dt), "gradient_norm": float(gradient_norm),
        "update_clip_scale": float(update_clip_scale),
        "lambda_hat": None if lambda_hat is None else float(lambda_hat),
        "rate_definition": "(log(norm(nu[n+1]))-log(norm(nu[n])))/dt",
        "covector_includes_additive_kernel_term": True,
        "color_percentile": 98, "color_limit": limit,
        "color_saturated_segments": int(np.count_nonzero(valid & (np.abs(rates) > limit))),
        "undefined_segments": int(np.count_nonzero(~valid)),
        "drift_field_recorded": drift is not None,
        "dimension": dimension,
        "growth_uses_full_covector_norm": True,
        "drift_contours_use_full_vector_norm": True,
    }
    arrays = {
        "trajectories_normalized": trajectories, "covectors": values,
        "path_indices": indices, "times": np.arange(values.shape[1]) * float(dt),
        "covector_norms": norms, "log_covector_norms": log_norms,
        "growth_rates": rates, "growth_rate_valid": valid,
        "projection_center": center, "projection_basis": basis,
        "metadata_json": np.frombuffer(json.dumps(metadata, sort_keys=True,
                                                    allow_nan=False).encode("utf-8"), dtype=np.uint8),
    }
    if reference is not None:
        arrays["reference_normalized"] = reference
    if drift is not None:
        arrays["drift_grid_normalized"] = drift_grid
        arrays["drift_normalized"] = drift
    if update_key is not None:
        update_key = np.asarray(update_key)
        if update_key.dtype != np.dtype("uint32") or update_key.shape != (2,):
            raise ValueError("update_key must be a uint32 PRNG key with shape (2,)")
        arrays["update_key"] = update_key
    if alpha is not None:
        alpha = np.asarray(alpha)
        if (alpha.dtype.kind not in "fiu" or alpha.shape != rates.shape
                or not np.isfinite(alpha).all() or np.any(alpha < 0)):
            raise ValueError("alpha must be finite and nonnegative with shape (paths, steps)")
        arrays["alpha"] = alpha
    output_directory = Path(output_directory)
    stem = output_directory / f"update-{iteration:06d}"
    _atomic_write(stem.with_suffix(".npz"), lambda stream: np.savez_compressed(stream, **arrays))
    figure = _figure(trajectories, norms, rates, valid, dt=dt, iteration=iteration,
                     gradient_norm=gradient_norm, update_clip_scale=update_clip_scale,
                     color_limit=limit, reference=reference, lambda_hat=lambda_hat,
                     drift_grid_normalized=drift_grid, drift_normalized=drift,
                     projection_center=center, projection_basis=basis)
    image_path = stem.with_suffix(".png")
    try:
        _atomic_write(image_path, lambda stream: figure.savefig(stream, format="png", dpi=160))
    finally:
        figure.clear()
    return image_path

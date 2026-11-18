"""Sample and time-conditioned score diagnostics in normalized coordinates."""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from .metrics import _points
from .plots import draw_sample_panel, validate_projection, project_grid_values, _plane_labels


def _figure(generated, reference, grid, scores, times, *, level_name="t", method_name="VP",
            projection_center=None, projection_basis=None):
    """Build a sample panel and three score-field panels with shared limits."""
    generated = _points(generated, "generated")
    dimension = generated.shape[1]
    reference = _points(reference, "reference", dimension=dimension)
    center, basis = validate_projection(dimension, projection_center, projection_basis)
    grid = np.asarray(grid)
    projected_grids = project_grid_values(grid, basis, center)
    if min(projected_grids.shape[1:3]) < 2:
        raise ValueError("grid must have at least two rows and columns")
    times = np.asarray(times)
    if level_name not in ("t", "sigma"):
        raise ValueError("level_name must be t or sigma")
    if (times.shape != (3,) or np.iscomplexobj(times)
            or not np.isfinite(times).all() or np.any(times <= 0)
            or (level_name == "t" and np.any(times > 1))):
        raise ValueError("times must contain three positive finite levels (VP times at most 1)")
    scores = np.asarray(scores)
    if scores.shape != (3, *grid.shape):
        raise ValueError("scores must have shape (3, *grid.shape) matching grid")
    projected_scores = np.stack([project_grid_values(field, basis) for field in scores])
    generated, reference = [(points - center) @ basis for points in (generated, reference)]
    planes = dimension // 2
    limits = []
    for plane in range(planes):
        flat_grid = projected_grids[plane].reshape(-1, 2)
        lower, upper = flat_grid.min(axis=0), flat_grid.max(axis=0)
        if np.any(upper <= lower):
            raise ValueError("grid must span a nonzero interval in both coordinates")
        pair = slice(2 * plane, 2 * plane + 2)
        all_samples = np.concatenate((generated[:, pair], reference[:, pair]))
        if np.any(all_samples < lower) or np.any(all_samples > upper):
            raise ValueError("grid must cover all generated and reference samples")
        limits.append((lower, upper))

    figure = Figure(figsize=(10, 9) if dimension == 2 else (20, 4.8 * planes),
                    layout="constrained")
    FigureCanvasAgg(figure)
    axes = (figure.subplots(2, 2).reshape(1, 4) if dimension == 2
            else figure.subplots(planes, 4, squeeze=False))
    # Share arrow lengths across levels and planes to preserve relative sizes.
    magnitude = np.hypot(projected_scores[..., 0], projected_scores[..., 1])
    typical = float(np.percentile(magnitude, 95))
    spacing = min(min((upper[0] - lower[0]) / (projected_grids.shape[2] - 1),
                      (upper[1] - lower[1]) / (projected_grids.shape[1] - 1))
                  for lower, upper in limits)
    arrow_scale = typical / (0.8 * spacing) if typical > 0 else 1.0
    for plane, row in enumerate(axes):
        pair = slice(2 * plane, 2 * plane + 2)
        draw_sample_panel(row[0], generated[:, pair], reference[:, pair])
        flat_grid = projected_grids[plane].reshape(-1, 2)
        flat_scores = projected_scores[:, plane].reshape(3, -1, 2)
        for axis, field, time in zip(row[1:], flat_scores, times):
            axis.quiver(flat_grid[:, 0], flat_grid[:, 1], field[:, 0], field[:, 1],
                        angles="xy", scale_units="xy", scale=arrow_scale, pivot="tail",
                        color="0.2", alpha=0.8)
            axis.set_title(f"Score vectors at {level_name} = {float(time):g}")
        lower, upper = limits[plane]
        xlabel, ylabel, title = _plane_labels(dimension, plane)
        if dimension > 2:
            row[0].set_title(title + ": samples")
        for index, axis in enumerate(row):
            axis.set(xlabel=xlabel, xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
            if index == 0 or (dimension == 2 and index == 2):
                axis.set_ylabel(ylabel)
            axis.set_aspect("equal", adjustable="box")
    title = f"{method_name}: learned scores of the noisy data distribution"
    if dimension > 2:
        title += "\nField slices through the target mean; omitted projected coordinates are zero"
    figure.suptitle(title)
    return figure


def save_score_diagnostics(output, generated, reference, grid, scores, times, *,
                            level_name="t", method_name="VP",
                            projection_center=None, projection_basis=None):
    """Save matched sample displays and three time- or noise-conditioned scores.

    Inputs use normalized coordinates. The grid must cover both sample clouds;
    ``plots.diagnostic_grid`` constructs a suitable uniform evaluation grid.
    Both DK and score plots display at most 5,000 points from each cloud.
    Score vectors are gradients of log noisy-data density, not the DK drift.
    EDM uses level_name="sigma" for noise standard deviations in normalized units.
    """
    # EDM levels are Gaussian noise standard deviations in normalized z units.
    figure = _figure(generated, reference, grid, scores, times,
                      level_name=level_name, method_name=method_name,
                      projection_center=projection_center, projection_basis=projection_basis)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output, dpi=160)
    finally:
        figure.clear()
    return output

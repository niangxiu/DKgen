"""Headless sample and learned-field diagnostics in normalized coordinates."""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .metrics import _points


DIAGNOSTIC_SAMPLE_LIMIT = 5000


def validate_projection(dimension, center=None, basis=None):
    """Validate an orthonormal frame whose columns are projection directions."""
    if (isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer))
            or dimension < 2 or dimension % 2):
        raise ValueError("Projection plots require an even dimension of at least two")
    if center is None and basis is None:
        return np.zeros(dimension), np.eye(dimension)
    if center is None or basis is None:
        raise ValueError("Projection center and basis must be supplied together")
    from .data import _real_array
    center = _real_array(center, "projection center")
    basis = _real_array(basis, "projection basis")
    if center.shape != (dimension,) or basis.shape != (dimension, dimension):
        raise ValueError("Projection center/basis shapes must be (D,) and (D,D)")
    if not np.allclose(basis.T @ basis, np.eye(dimension), rtol=1e-7, atol=1e-7):
        raise ValueError("Projection basis must be orthonormal")
    return center, basis


def projection_frame(dimension, target=None):
    """Use ordinary 2D coordinates, or a fixed target mean plane and complement."""
    center, basis = validate_projection(dimension)
    if dimension == 2:
        return center, basis
    if target is None:
        raise ValueError("Higher-dimensional projection_frame requires target metadata")
    from .data import _validated_parameters
    weights, means, _, _ = _validated_parameters(
        target["weights"], target["means_normalized"], target["covariances_normalized"])
    if means.shape[1] != dimension:
        raise ValueError("Target dimension does not match projection dimension")
    center = weights @ means
    _, _, vectors = np.linalg.svd(
        np.sqrt(weights)[:, None] * (means - center), full_matrices=False)
    directions = []
    for candidate in [*vectors[:2], *np.eye(dimension)]:
        direction = candidate.copy()
        for _ in range(2):
            for previous in directions:
                direction -= np.dot(previous, direction) * previous
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            continue
        direction /= norm
        if direction[np.argmax(np.abs(direction))] < 0:
            direction *= -1
        directions.append(direction)
        if len(directions) == dimension:
            break
    return validate_projection(dimension, center, np.stack(directions, axis=1))


def projection_grids(generated, reference, *, center, basis, points_per_axis=35, extent=None):
    """Full-state grids on paired projection planes through the fixed center.

    Return (D/2, rows, columns, D). Omitted projected coordinates equal zero;
    sample-cloud projections determine the visible range unless extent is given.
    """
    generated = _points(generated, "generated")
    dimension = generated.shape[1]
    reference = _points(reference, "reference", dimension=dimension)
    center, basis = validate_projection(dimension, center, basis)
    if (isinstance(points_per_axis, bool) or not isinstance(points_per_axis, int)
            or points_per_axis < 2):
        raise ValueError("points_per_axis must be an integer of at least two")
    if extent is not None and (isinstance(extent, bool) or not np.isscalar(extent)
                               or not np.isfinite(extent) or extent <= 0):
        raise ValueError("extent must be finite and positive")
    projected = [(points - center) @ basis for points in (generated, reference)]
    grids = []
    for plane in range(dimension // 2):
        pair = slice(2 * plane, 2 * plane + 2)
        if extent is None:
            coordinates = diagnostic_grid(*(points[:, pair] for points in projected),
                                          points_per_axis=points_per_axis)
        else:
            x, y = np.meshgrid(np.linspace(-extent, extent, points_per_axis),
                               np.linspace(-extent, extent, points_per_axis))
            coordinates = np.stack((x, y), axis=-1)
        grids.append(center + coordinates @ basis[:, pair].T)
    return np.stack(grids)


def project_grid_values(values, basis, center=None):
    """Project each full-state grid/vector onto its corresponding coordinate pair."""
    values = np.asarray(values)
    dimension = basis.shape[0]
    if values.ndim == 3 and dimension == 2:
        values = values[None]
    if (values.ndim != 4 or values.shape[0] != dimension // 2
            or values.shape[-1] != dimension or min(values.shape[1:3]) < 1):
        raise ValueError("Field grids must have shape (D/2, rows, columns, D)")
    values = _points(values.reshape(-1, dimension), "field grid").reshape(values.shape)
    if center is not None:
        values = values - center
    return np.stack([values[plane] @ basis[:, 2 * plane:2 * plane + 2]
                     for plane in range(dimension // 2)])


def _plane_labels(dimension, plane):
    if dimension == 2:
        return r"$z_1 = x_1/R$", r"$z_2 = x_2/R$", "Normalized coordinates"
    first = 2 * plane + 1
    return (f"Projected coordinate {first}", f"Projected coordinate {first + 1}",
            "Target mean plane" if plane == 0 else f"Complement plane {first}, {first + 1}")


def save_sample_projections(path, generated, reference, *, iteration, metrics=None,
                            projection_center=None, projection_basis=None, method_name=None):
    """Save sample clouds in fixed paired projections, with common [-2,2] limits."""
    generated = _points(generated, "generated")
    dimension = generated.shape[1]
    reference = _points(reference, "reference", dimension=dimension)
    center, basis = validate_projection(dimension, projection_center, projection_basis)
    generated, reference = [(points - center) @ basis for points in (generated, reference)]
    planes = dimension // 2
    columns = min(planes, 2)
    rows = (planes + columns - 1) // columns
    figure = Figure(figsize=(5.4 * columns, 5 * rows), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False).ravel()
    for plane, axis in enumerate(axes):
        if plane >= planes:
            axis.set_visible(False)
            continue
        pair = slice(2 * plane, 2 * plane + 2)
        draw_sample_panel(axis, generated[:, pair], reference[:, pair])
        xlabel, ylabel, title = _plane_labels(dimension, plane)
        axis.set(xlabel=xlabel, ylabel=ylabel, title=title, xlim=(-2, 2), ylim=(-2, 2))
        axis.set_aspect("equal", adjustable="box")
        outside = [int(np.any(np.abs(points[:, pair]) > 2, axis=1).sum())
                   for points in (generated, reference)]
        if any(outside):
            axis.text(.02, .02, f"Outside view: {outside[0]} generated, {outside[1]} data",
                      transform=axis.transAxes, fontsize=8)
    title = f"{method_name + ', ' if method_name else ''}{dimension}D, update {iteration:,}"
    if metrics is not None and "averaged_kl" in metrics:
        title += f" | Estimated averaged KL = {metrics['averaged_kl']:.4g}"
    figure.suptitle(title)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output, dpi=160)
    finally:
        figure.clear()
    return output


def draw_sample_panel(axis, generated, reference):
    """Use the same displayed counts and scatter style for DK and score plots."""
    for points, color, label in ((reference, "tab:orange", "Reference"),
                                  (generated, "tab:blue", "Generated")):
        display = points
        if len(points) > DIAGNOSTIC_SAMPLE_LIMIT:
            indices = np.linspace(0, len(points) - 1, DIAGNOSTIC_SAMPLE_LIMIT, dtype=int)
            display = points[indices]
        axis.scatter(display[:, 0], display[:, 1], s=1.5, alpha=0.15,
                     color=color, linewidths=0, label=f"{label} (n={len(display):,})")
    axis.set_title("Samples in normalized coordinates")
    legend = axis.legend(loc="upper right", markerscale=4)
    for handle in legend.legend_handles:
        handle.set_alpha(1)


def diagnostic_grid(generated_z, reference_z, points_per_axis=35):
    """Uniform field grid covering both sample clouds and their display margin."""
    generated = _points(generated_z, "generated_z", dimension=2)
    reference = _points(reference_z, "reference_z", dimension=2)
    if not isinstance(points_per_axis, int) or isinstance(points_per_axis, bool) or points_per_axis < 2:
        raise ValueError("points_per_axis must be an integer of at least two")
    points = np.concatenate((generated, reference), axis=0)
    lower, upper = points.min(axis=0), points.max(axis=0)
    padding = 0.05 * np.maximum(upper - lower, 1.0)
    gx, gy = np.meshgrid(np.linspace(lower[0] - padding[0], upper[0] + padding[0], points_per_axis),
                         np.linspace(lower[1] - padding[1], upper[1] + padding[1], points_per_axis))
    return np.stack((gx, gy), axis=-1)


def _diagnostic_figure(generated_z, reference_z, grid_z, drift_z, sigma_values_z, *,
                       projection_center=None, projection_basis=None):
    generated = _points(generated_z, "generated_z")
    dimension = generated.shape[1]
    reference = _points(reference_z, "reference_z", dimension=dimension)
    center, basis = validate_projection(dimension, projection_center, projection_basis)
    grid_array = np.asarray(grid_z)
    drift_array = np.asarray(drift_z)
    if drift_array.shape != grid_array.shape:
        raise ValueError("drift_z and grid_z must have matching shapes")
    grids = project_grid_values(grid_array, basis, center)
    drifts = project_grid_values(drift_array, basis)
    sigma = np.asarray(sigma_values_z)
    if np.iscomplexobj(sigma):
        raise ValueError("sigma_values_z must be real")
    sigma = np.asarray(sigma, dtype=np.float64)
    expected_shape = grids.shape[:-1]
    if sigma.size != np.prod(expected_shape) or not np.isfinite(sigma).all() or np.any(sigma < 0):
        raise ValueError("sigma_values_z must contain one finite nonnegative value per grid point")
    sigma = sigma.reshape(expected_shape)
    generated, reference = [(points - center) @ basis for points in (generated, reference)]
    planes = dimension // 2
    figure = Figure(figsize=(16, 4.8 * planes), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(planes, 3, squeeze=False)
    sigma_low, sigma_high = float(sigma.min()), float(sigma.max())
    drift_scale = None
    if dimension > 2:
        # One scale preserves drift magnitudes across projection planes. Use
        # the maximum, since a mostly zero field makes percentile scales tiny.
        maximum = float(np.hypot(drifts[..., 0], drifts[..., 1]).max())
        spans = np.ptp(grids, axis=(1, 2))
        positive_spans = spans[spans > 0]
        arrow_length = .1 * float(positive_spans.min()) if positive_spans.size else .1
        drift_scale = maximum / arrow_length if maximum > 0 else 1.
    for plane, row in enumerate(axes):
        pair = slice(2 * plane, 2 * plane + 2)
        grid = grids[plane]
        flat_grid, drift = grid.reshape(-1, 2), drifts[plane].reshape(-1, 2)
        samples, data = generated[:, pair], reference[:, pair]
        all_points = np.concatenate((samples, data, flat_grid))
        lower, upper = all_points.min(axis=0), all_points.max(axis=0)
        draw_sample_panel(row[0], samples, data)
        row[1].quiver(flat_grid[:, 0], flat_grid[:, 1], drift[:, 0], drift[:, 1],
                      angles="xy", scale_units="xy", pivot="tail",
                      **({"scale": drift_scale} if dimension > 2 else {}))
        row[1].set_title("Drift in normalized coordinates" if dimension == 2 else "Projected drift on slice")
        local_sigma = sigma[plane]
        if min(grid.shape[:2]) >= 2:
            if sigma_low < sigma_high:
                colors = row[2].contourf(grid[..., 0], grid[..., 1], local_sigma,
                                         levels=np.linspace(sigma_low, sigma_high, 24), cmap="viridis")
                if local_sigma.min() < local_sigma.max():
                    contours = row[2].contour(
                        grid[..., 0], grid[..., 1], local_sigma,
                        levels=np.linspace(sigma_low, sigma_high, 9)[1:-1],
                        colors="white", linewidths=.6, alpha=.8)
                    row[2].clabel(contours, inline=True, fontsize=7, fmt="%.3g")
            else:
                padding = .01 * max(sigma_low, 1.)
                colors = row[2].contourf(
                    grid[..., 0], grid[..., 1], local_sigma,
                    levels=[max(0., sigma_low - padding), sigma_high + padding], cmap="viridis")
        else:
            colors = row[2].scatter(flat_grid[:, 0], flat_grid[:, 1], c=local_sigma.ravel(),
                                    marker="s", s=18, cmap="viridis", linewidths=0,
                                    vmin=sigma_low, vmax=sigma_high)
        colorbar = figure.colorbar(colors, ax=row[2], label="Normalized scalar diffusion amplitude")
        if sigma_low == sigma_high:
            colorbar.set_ticks([sigma_low])
        row[2].set_title("Diffusion in normalized coordinates" if dimension == 2 else "Scalar diffusion on slice")
        xlabel, ylabel, title = _plane_labels(dimension, plane)
        if dimension > 2:
            row[0].set_title(title + ": samples")
        row[0].set_ylabel(ylabel)
        for axis in row:
            axis.set(xlabel=xlabel, xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
            axis.set_aspect("equal", adjustable="box")
    if dimension > 2:
        figure.suptitle("Field slices through the target mean; omitted projected coordinates are zero")
    return figure


def save_diagnostics(output_path, generated_z, reference_z, grid_z,
                     drift_z, sigma_values_z, *, projection_center=None, projection_basis=None):
    """Save samples, projected drift, and scalar diffusion on fixed field slices.

    The 2D API accepts grids and drift of shape (H,W,2). Higher dimensions use
    (D/2,H,W,D), with scalar diffusion (D/2,H,W). Full-state vector fields are
    projected using basis columns, and samples are projected about the center.
    projection_grids builds the field slices through that same center.
    """
    figure = _diagnostic_figure(
        generated_z, reference_z, grid_z, drift_z, sigma_values_z,
        projection_center=projection_center, projection_basis=projection_basis)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output, dpi=160)
    finally:
        figure.clear()
    return output

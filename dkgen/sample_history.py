"""Record and plot generated samples across optimization without loading JAX."""

import argparse
from pathlib import Path
import tempfile

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import colormaps
from matplotlib.colors import FuncNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import numpy as np

from .file_io import replace_with_retry
from .metrics import nearest_neighbor_weights
from .plots import validate_projection


DEFAULT_SNAPSHOT_COUNT = 20


def _points(values, name):
    values = np.asarray(values)
    if (values.dtype.kind not in "fiu" or values.ndim != 2
            or values.shape[1] < 2 or values.shape[1] % 2 or len(values) == 0
            or not np.isfinite(values).all()):
        raise ValueError(f"{name} must contain finite real samples with shape (N, even dimension >= 2)")
    return values


def _iteration(value):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 0):
        raise ValueError("Iteration must be a nonnegative integer")
    return int(value)


def _iterations(values):
    values = np.asarray(values)
    if (values.ndim != 1 or values.dtype.kind not in "iu"
            or np.any(values < 0) or np.any(values[1:] <= values[:-1])):
        raise ValueError("Iterations must be sorted, unique nonnegative integers")
    return values


def _archive_path(path):
    path = Path(path)
    return path / "sample_history.npz" if path.is_dir() else path


def _source_and_neighbors(sample_source, neighbors, n_generated):
    if sample_source not in ("training", "validation", "score"):
        raise ValueError("Sample source must be training, validation, or score")
    if (isinstance(neighbors, (bool, np.bool_))
            or not isinstance(neighbors, (int, np.integer))
            or not 1 <= neighbors <= n_generated):
        raise ValueError("Neighbors must be an integer between 1 and the sample count")
    return sample_source, int(neighbors)


def _selection_counts(values, shape, n_reference, neighbors):
    counts = np.asarray(values)
    if (counts.dtype.kind not in "iu" or counts.shape != shape
            or np.any(counts < 0) or np.any(counts > n_reference)
            or np.any(counts.sum(axis=-1) != n_reference * neighbors)):
        raise ValueError("Selection counts must be nonnegative integers summing to data count times neighbors")
    return counts


def _count_matches(reference, generated, neighbors):
    weights = nearest_neighbor_weights(reference, generated, neighbors)
    return np.rint(weights * (len(reference) * neighbors)).astype(np.int64)


def _trajectories(values, generated):
    values = np.asarray(values)
    if (values.dtype.kind not in "fiu" or values.ndim != 3
            or not 1 <= values.shape[0] <= min(10, len(generated))
            or values.shape[1] < 2 or values.shape[2] != generated.shape[-1]
            or not np.isfinite(values).all()):
        raise ValueError("Trajectories must have finite shape (1 to 10 paths, at least 2 times, dimension)")
    if not np.allclose(values[:, -1], generated[:len(values)], rtol=1e-5, atol=1e-6):
        raise ValueError("Trajectory endpoints must match the first generated samples")
    return values


def _read_trajectories(archive, history):
    fields = ("trajectories_normalized", "trajectory_available")
    present = [field in archive for field in fields]
    if not any(present):
        return
    if not all(present):
        raise ValueError("Sample history needs both trajectories and their availability mask")
    trajectories, available = (archive[field] for field in fields)
    generated = history["generated_normalized"]
    if (available.dtype.kind != "b" or available.shape != (len(generated),)):
        raise ValueError("Trajectory availability must be a boolean mask with one value per update")
    if (trajectories.dtype.kind not in "fiu" or trajectories.ndim != 4
            or trajectories.shape[0] != len(generated)
            or not 1 <= trajectories.shape[1] <= min(10, generated.shape[1])
            or trajectories.shape[2] < 2 or trajectories.shape[3] != generated.shape[-1]
            or not np.isfinite(trajectories).all()):
        raise ValueError("Trajectory history must have finite shape (updates, paths, times, dimension)")
    if not np.allclose(trajectories[available, :, -1],
                       generated[available, :trajectories.shape[1]], rtol=1e-5, atol=1e-6):
        raise ValueError("Recorded trajectory endpoints must match their generated samples")
    history.update(zip(fields, (trajectories, available)))


def _drift_field(grid_values, drift_values, *, projection_center=None, projection_basis=None):
    grid, drift = np.asarray(grid_values), np.asarray(drift_values)
    if (grid.dtype.kind not in "fiu" or grid.ndim not in (3, 4)
            or min(grid.shape[-3:-1]) < 2 or grid.shape[-1] < 2
            or grid.shape[-1] % 2
            or not np.isfinite(grid).all()):
        raise ValueError("Drift grid must have finite shape (rows, columns, 2) or (projections, rows, columns, dimension)")
    dimension = grid.shape[-1]
    if (grid.ndim == 3 and dimension != 2) or (grid.ndim == 4 and grid.shape[0] != dimension // 2):
        raise ValueError("Drift grid requires one slice for each coordinate pair")
    if (drift.dtype.kind not in "fiu" or drift.shape != grid.shape
            or not np.isfinite(drift).all()):
        raise ValueError("Drift vectors must be finite and match the drift grid shape")
    center, basis = validate_projection(dimension, projection_center, projection_basis)
    slices = grid[None] if grid.ndim == 3 else grid
    for pair, full_grid in enumerate(slices):
        coordinates = (np.asarray(full_grid, dtype=np.float64) - center) @ basis
        visible = coordinates[..., 2 * pair:2 * pair + 2]
        x, y = visible[0, :, 0], visible[:, 0, 1]
        dx, dy = np.diff(x), np.diff(y)
        if (not np.allclose(visible[:, :, 0], x[None, :], atol=1e-6)
                or not np.allclose(visible[:, :, 1], y[:, None], atol=1e-6)
                or np.any(dx <= 0) or np.any(dy <= 0)
                or not np.allclose(dx, dx[0], rtol=1e-5, atol=1e-7)
                or not np.allclose(dy, dy[0], rtol=1e-5, atol=1e-7)):
            raise ValueError("Drift grid must be a uniform rectangular grid increasing along each axis")
        omitted = np.delete(coordinates, (2 * pair, 2 * pair + 1), axis=-1)
        if not np.allclose(omitted, 0., atol=1e-6):
            raise ValueError("Drift slices must fix omitted projection coordinates at zero")
    return grid, drift


def _read_drift(archive, history):
    fields = ("drift_grid_normalized", "drift_normalized", "drift_available")
    present = [field in archive for field in fields]
    if not any(present):
        return
    if not all(present):
        raise ValueError("Sample history needs the drift grid, vectors, and availability mask together")
    grids, drifts, available = (archive[field] for field in fields)
    n_updates = len(history["iterations"])
    if available.dtype.kind != "b" or available.shape != (n_updates,):
        raise ValueError("Drift availability must be a boolean mask with one value per update")
    dimension = history["generated_normalized"].shape[-1]
    if (grids.dtype.kind not in "fiu" or grids.ndim not in (4, 5)
            or grids.shape[0] != n_updates or min(grids.shape[-3:-1]) < 2
            or grids.shape[-1] != dimension or not np.isfinite(grids).all()
            or drifts.dtype.kind not in "fiu" or drifts.shape != grids.shape
            or not np.isfinite(drifts).all()):
        raise ValueError("Drift history must have finite matching shapes with one grid per update")
    if ((grids.ndim == 4 and dimension != 2)
            or (grids.ndim == 5 and grids.shape[1] != dimension // 2)):
        raise ValueError("Drift history requires one slice for each coordinate pair")
    for index in np.flatnonzero(available):
        _drift_field(grids[index], drifts[index], projection_center=history["projection_center"],
                     projection_basis=history["projection_basis"])
    history.update(zip(fields, (grids, drifts, available)))


def _diffusion_field(values, grid):
    diffusion = np.asarray(values)
    if (diffusion.dtype.kind not in "fiu" or diffusion.shape != grid.shape[:-1]
            or not np.isfinite(diffusion).all() or np.any(diffusion <= 0)):
        raise ValueError("Diffusion amplitudes must be finite, positive, and match the drift grid")
    return diffusion


def _read_diffusion(archive, history):
    fields = ("diffusion_normalized", "diffusion_available")
    present = [field in archive for field in fields]
    if not any(present):
        return
    if not all(present):
        raise ValueError("Sample history needs both diffusion amplitudes and their availability mask")
    diffusion, available = (archive[field] for field in fields)
    n_updates = len(history["iterations"])
    if available.dtype.kind != "b" or available.shape != (n_updates,):
        raise ValueError("Diffusion availability must be a boolean mask with one value per update")
    grids = history.get("drift_grid_normalized")
    if grids is None:
        raise ValueError("Diffusion history requires the recorded drift grid and vectors")
    if (diffusion.dtype.kind not in "fiu" or diffusion.shape != grids.shape[:-1]
            or not np.isfinite(diffusion).all()):
        raise ValueError("Diffusion history must have finite shape (updates, rows, columns)")
    if np.any(available & ~history["drift_available"]):
        raise ValueError("Recorded diffusion requires an available drift grid and vectors")
    for index in np.flatnonzero(available):
        _diffusion_field(diffusion[index], grids[index])
    history.update(zip(fields, (diffusion, available)))


def _initial_distribution(mean_values, s0_value, dimension):
    mean, s0 = np.asarray(mean_values), np.asarray(s0_value)
    if (mean.dtype.kind not in "fiu" or mean.shape != (dimension,)
            or not np.isfinite(mean).all()):
        raise ValueError("Initial mean must contain finite real values with shape (dimension,)")
    if (s0.dtype.kind not in "fiu" or s0.shape != ()
            or not np.isfinite(s0) or s0 <= 0):
        raise ValueError("Initial s0 must be a finite positive real scalar")
    return mean, s0


def _read_initial_distribution(archive, history):
    fields = ("initial_mean_normalized", "initial_s0", "initial_distribution_available")
    present = [field in archive for field in fields]
    if not any(present):
        return
    if not all(present):
        raise ValueError("Sample history needs the initial mean, s0, and availability mask together")
    means, scales, available = (archive[field] for field in fields)
    n_updates, _, dimension = history["generated_normalized"].shape
    if available.dtype.kind != "b" or available.shape != (n_updates,):
        raise ValueError("Initial distribution availability must be a boolean mask with one value per update")
    if (means.dtype.kind not in "fiu" or means.shape != (n_updates, dimension)
            or not np.isfinite(means).all()):
        raise ValueError("Initial means must have finite real shape (updates, dimension)")
    if (scales.dtype.kind not in "fiu" or scales.shape != (n_updates,)
            or not np.isfinite(scales).all() or np.any(scales[available] <= 0)):
        raise ValueError("Initial scales must be finite and positive for available distributions")
    if np.any(means[~available] != 0) or np.any(scales[~available] != 0):
        raise ValueError("Unavailable initial distributions must use zero placeholders")
    history.update(zip(fields, (means, scales, available)))


def read_sample_history(history_path):
    """Read full-state normalized samples and their fixed display projection."""
    path = _archive_path(history_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No recorded sample history at {path}; past samples cannot be "
            "reconstructed from error logs alone"
        )
    with np.load(path, allow_pickle=False) as archive:
        try:
            steps = _iterations(archive["iterations"])
            reference = _points(archive["reference_normalized"], "reference_normalized")
            generated = archive["generated_normalized"]
            fingerprint = archive["data_hash"]
        except KeyError as exc:
            raise ValueError(f"Missing sample history field: {exc}") from exc
        if (fingerprint.shape != () or fingerprint.dtype.kind != "U"
                or not str(fingerprint)):
            raise ValueError("Sample history needs a nonempty data fingerprint")
        if (generated.dtype.kind not in "fiu" or generated.ndim != 3
                or generated.shape[0] != len(steps) or generated.shape[1] == 0
                or generated.shape[2] != reference.shape[-1] or not np.isfinite(generated).all()):
            raise ValueError("Generated history must have shape (updates, samples, dimension) and be finite")
        history = {"iterations": steps, "reference_normalized": reference,
                   "generated_normalized": generated, "data_hash": str(fingerprint)}
        center, basis = validate_projection(reference.shape[-1], archive.get("projection_center"),
                                             archive.get("projection_basis"))
        history.update(projection_center=center, projection_basis=basis)
        source = archive.get("sample_source", np.asarray("validation"))
        neighbors = archive.get("neighbors", np.asarray(1))
        if source.shape != () or source.dtype.kind != "U" or neighbors.shape != ():
            raise ValueError("Sample source and neighbors must be scalars")
        source, neighbors = _source_and_neighbors(str(source), neighbors.item(), generated.shape[1])
        history.update(sample_source=source, neighbors=neighbors)
        if "selection_counts" in archive:
            history["selection_counts"] = _selection_counts(
                archive["selection_counts"], generated.shape[:2], len(reference), neighbors)
        elif source == "training":
            raise ValueError("Training history must include the actual selection counts")
        _read_trajectories(archive, history)
        _read_drift(archive, history)
        _read_diffusion(archive, history)
        _read_initial_distribution(archive, history)
        return history


def _check_reference(history, reference, data_hash):
    if not isinstance(data_hash, str) or not data_hash:
        raise ValueError("data_hash must be a nonempty string")
    if history["data_hash"] != data_hash:
        raise ValueError("Sample history data fingerprint does not match the current dataset")
    if not np.array_equal(history["reference_normalized"], reference):
        raise ValueError("Sample history reference samples do not match the current reference data")


def _history_projection(history, dimension, center=None, basis=None):
    if history is not None and center is None and basis is None:
        return history["projection_center"], history["projection_basis"]
    center, basis = validate_projection(dimension, center, basis)
    if history is not None and (
            not np.allclose(history["projection_center"], center, rtol=0, atol=1e-12)
            or not np.allclose(history["projection_basis"], basis, rtol=0, atol=1e-12)):
        raise ValueError("Sample history projection frame must stay fixed")
    return center, basis


def _save_archive(path, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".sample-history-",
                                         suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **history)
        replace_with_retry(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def select_snapshot_indices(iterations, count=DEFAULT_SNAPSHOT_COUNT):
    """Choose distinct snapshots near evenly spaced update numbers, with endpoints."""
    steps = _iterations(iterations)
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("count must be an integer of at least two")
    if len(steps) <= count:
        return np.arange(len(steps))
    targets = np.linspace(float(steps[0]), float(steps[-1]), count)
    selected = [0]
    for position in range(1, count - 1):
        # Reserve enough later entries so sparse histories still give distinct
        # panels and always include the latest update.
        first = selected[-1] + 1
        stop = len(steps) - (count - position - 1)
        selected.append(first + int(np.argmin(np.abs(steps[first:stop] - targets[position]))))
    return np.array(selected + [len(steps) - 1])


def history_plot_slots(inherited_iterations, first_update, last_update, eval_every):
    """Plan the same twenty live snapshot slots for DK and score runs."""
    planned = list(inherited_iterations)
    if first_update <= last_update:
        planned.extend((first_update, last_update))
        first_multiple = ((first_update + eval_every - 1) // eval_every) * eval_every
        planned.extend(range(first_multiple, last_update + 1, eval_every))
    planned = np.asarray(sorted(set(planned)), dtype=np.int64)
    selected = select_snapshot_indices(planned)
    return {int(planned[index]): slot for slot, index in enumerate(selected, start=1)}


def save_sample_history_plot(history_path, output_path=None, *, training_reference=False,
                             iteration=None, slot=None):
    """Save history figures, or one recorded iteration in its planned output slot.

    Full exports share coordinate and color limits. Live exports update only
    the requested slot, using the available history to determine those limits.
    """
    history_path = _archive_path(history_path)
    history = read_sample_history(history_path)
    selected = select_snapshot_indices(history["iterations"])
    if (iteration is None) != (slot is None):
        raise ValueError("A live snapshot requires both iteration and slot")
    if iteration is not None:
        iteration = _iteration(iteration)
        if (isinstance(slot, (bool, np.bool_)) or not isinstance(slot, (int, np.integer))
                or not 1 <= slot <= DEFAULT_SNAPSHOT_COUNT):
            raise ValueError(f"Snapshot slot must be an integer from 1 to {DEFAULT_SNAPSHOT_COUNT}")
        matches = np.flatnonzero(history["iterations"] == iteration)
        if not len(matches):
            raise ValueError(f"No recorded sample snapshot for update {iteration}")
        selected = np.unique(np.append(selected, matches[0]))
        exports = [(int(slot), int(np.flatnonzero(selected == matches[0])[0]))]
    else:
        exports = [(index + 1, index) for index in range(len(selected))]
    if len(selected) == 0:
        return None
    output_path = (Path(output_path) if output_path is not None
                   else history_path.with_name("sample_history"))
    if output_path.suffix.lower() == ".png" or output_path.is_file():
        raise ValueError("Sample history output must be a directory for individual PNGs")
    reference = history["reference_normalized"]
    score_history = history["sample_source"] == "score"
    reference_source = "training" if score_history else history["sample_source"]
    stored_counts = history.get("selection_counts")
    if training_reference and reference_source != "training":
        # Legacy validation paths can be compared with training data, but they
        # must remain labeled as validation paths, not actual training batches.
        from .checkpoint import data_fingerprint
        from .data import load_data
        data = load_data(history_path.with_name("data.npz"))
        if data_fingerprint(data) != history["data_hash"]:
            raise ValueError("Training reference dataset does not match the sample history")
        reference = _points(data["train"], "training reference")
        reference_source, stored_counts = "training", None
    snapshots = history["generated_normalized"][selected]
    neighbors = history["neighbors"]
    counts = (stored_counts[selected] if stored_counts is not None else np.stack([
        _count_matches(reference, generated, neighbors) for generated in snapshots]))
    max_count = max(1, int(counts.max()))
    norm = FuncNorm((np.log1p, np.expm1), vmin=0, vmax=max_count)
    cmap = ListedColormap(colormaps["Blues"](np.linspace(0.3, 1.0, 256)))
    ticks = [0] + [scale * base for scale in (1, 10, 100, 1000, 10000, 100000)
                   for base in (1, 2, 5) if scale * base <= max_count]
    if ticks[-1] != max_count:
        if len(ticks) > 1 and np.log1p(max_count) - np.log1p(ticks[-1]) < 0.075 * np.log1p(max_count):
            ticks.pop()
        ticks.append(max_count)
    n_reference, n_generated = len(reference), snapshots.shape[1]
    dimension = reference.shape[-1]
    center, basis = history["projection_center"], history["projection_basis"]
    reference = (reference - center) @ basis
    projected_snapshots = (snapshots - center) @ basis
    point_sets = [reference, projected_snapshots.reshape(-1, dimension)]
    initial_means = history.get("initial_mean_normalized")
    initial_scales = history.get("initial_s0")
    initial_available = history.get(
        "initial_distribution_available", np.zeros(len(history["iterations"]), dtype=bool))
    if initial_means is not None:
        initial_means = (initial_means - center) @ basis
        initial_selected = selected[initial_available[selected]]
        for sign in (-1, 1):
            point_sets.append(initial_means[initial_selected]
                              + sign * 2 * initial_scales[initial_selected, None])
    trajectories = history.get("trajectories_normalized")
    available = history.get("trajectory_available", np.zeros(len(history["iterations"]), dtype=bool))
    drift_grids = history.get("drift_grid_normalized")
    fixed_view = score_history or drift_grids is not None
    drift_values = history.get("drift_normalized")
    drift_available = history.get("drift_available", np.zeros(len(history["iterations"]), dtype=bool))
    diffusion_values = history.get("diffusion_normalized")
    diffusion_available = history.get(
        "diffusion_available", np.zeros(len(history["iterations"]), dtype=bool))
    diffusion_selected = selected[diffusion_available[selected]]
    if len(diffusion_selected):
        selected_diffusion = diffusion_values[diffusion_selected]
        diffusion_min = float(selected_diffusion.min())
        diffusion_max = float(selected_diffusion.max())
        if diffusion_min == diffusion_max:
            # contourf needs increasing levels, including for constant diffusion.
            padding_sigma = max(0.01 * diffusion_min, 1e-6)
            diffusion_min = max(0.0, diffusion_min - padding_sigma)
            diffusion_max += padding_sigma
        diffusion_levels = np.linspace(diffusion_min, diffusion_max, 13)
        diffusion_cmap = ListedColormap(colormaps["Purples"](np.linspace(0.05, 0.7, 256)))
    if trajectories is not None:
        trajectories = (trajectories - center) @ basis
        point_sets.append(trajectories[selected[available[selected]]].reshape(-1, dimension))
    if drift_grids is not None:
        drift_grids = (drift_grids - center) @ basis
        drift_values = drift_values @ basis
        if drift_grids.ndim == 4:
            drift_grids, drift_values = drift_grids[:, None], drift_values[:, None]
    if diffusion_values is not None and diffusion_values.ndim == 3:
        diffusion_values = diffusion_values[:, None]
    all_points = np.concatenate(point_sets, axis=0)
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    padding = 0.05 * np.maximum(upper - lower, 1.0)
    lower, upper = lower - padding, upper + padding
    reference = reference[::max(1, (len(reference) + 4999) // 5000)]

    output_path.mkdir(parents=True, exist_ok=True)
    for output_slot, index in exports:
        snapshot_index = selected[index]
        pairs = dimension // 2
        columns = min(2, pairs)
        rows = (pairs + columns - 1) // columns
        figure = Figure(figsize=(7 * columns, 7.3 * rows), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(rows, columns, squeeze=False).ravel()
        for unused in axes[pairs:]:
            unused.set_visible(False)
        for pair, axis in enumerate(axes[:pairs]):
            first = 2 * pair
            if diffusion_available[snapshot_index]:
                grid = drift_grids[snapshot_index, pair, ..., first:first + 2]
                diffusion_colors = axis.contourf(
                    grid[:, :, 0], grid[:, :, 1], diffusion_values[snapshot_index, pair],
                    levels=diffusion_levels, cmap=diffusion_cmap, alpha=0.45,
                    zorder=-1)
                diffusion_colorbar = figure.colorbar(
                    diffusion_colors, ax=axis, orientation="vertical", pad=0.02,
                    fraction=0.045, shrink=0.85)
                diffusion_colorbar.set_label(r"Diffusion $\sigma(z)$", fontsize=9)
                diffusion_colorbar.ax.tick_params(labelsize=8)
            elif not score_history:
                axis.text(0.03, 0.15, "Diffusion not recorded", transform=axis.transAxes,
                          fontsize=9, color="0.4")
            if drift_grids is not None:
                if drift_available[snapshot_index]:
                    grid = drift_grids[snapshot_index, pair, ..., first:first + 2].reshape(-1, 2)
                    drift = drift_values[snapshot_index, pair, ..., first:first + 2].reshape(-1, 2)
                    arrows = axis.quiver(grid[:, 0], grid[:, 1], drift[:, 0], drift[:, 1],
                                        angles="xy", scale_units="xy", scale=10,
                                        pivot="tail", color="0.3", alpha=0.3, zorder=0)
                    arrow_label = r"Drift, $|F|=1$" if dimension == 2 else "Projected drift, length 1"
                    key = axis.quiverkey(arrows, 0.07, 0.96, 1, arrow_label,
                                         labelpos="E", coordinates="axes")
                    key.text.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.8))
                else:
                    axis.text(0.03, 0.07, "Drift not recorded", transform=axis.transAxes,
                              fontsize=9, color="0.4")
            generated = projected_snapshots[index, :, first:first + 2]
            if fixed_view:
                outside = int(np.count_nonzero(np.any((generated < -2) | (generated > 2), axis=1)))
                if outside:
                    axis.text(0.03, 0.11, f"{outside:,} generated samples outside view",
                              transform=axis.transAxes, fontsize=9, color="0.4")
            stride = max(1, (len(generated) + 4999) // 5000)
            generated = generated[::stride]
            if available[snapshot_index]:
                for path_index, trajectory in enumerate(trajectories[snapshot_index, ..., first:first + 2]):
                    axis.plot(trajectory[:, 0], trajectory[:, 1], color="#7057a3",
                              alpha=0.15, linewidth=0.6, zorder=1,
                              label="Sample paths" if path_index == 0 else "_nolegend_")
                endpoints = trajectories[snapshot_index, :, -1, first:first + 2]
                axis.scatter(endpoints[:, 0], endpoints[:, 1], s=18, marker="o",
                             facecolors="none", alpha=0.65, edgecolors="#7057a3",
                             linewidths=0.7, zorder=3, label="_nolegend_")
            elif not score_history:
                axis.text(0.03, 0.03, "Paths not recorded", transform=axis.transAxes,
                          fontsize=9, color="0.4")
            axis.scatter(reference[:, first], reference[:, first + 1], s=6, alpha=0.4,
                         color="tab:orange", linewidths=0, zorder=2,
                         label=f"{reference_source.capitalize()} data")
            colors = axis.scatter(generated[:, 0], generated[:, 1], s=8, alpha=0.8,
                                  c=counts[index, ::stride], cmap=cmap, norm=norm,
                                  linewidths=0, zorder=2, label="Generated")
            if initial_available[snapshot_index]:
                initial_mean = initial_means[snapshot_index, first:first + 2]
                initial_scale = initial_scales[snapshot_index]
                for radius in (1, 2):
                    axis.add_patch(Circle(
                        initial_mean, radius * initial_scale, fill=False,
                        edgecolor="#13855c", linestyle="--", linewidth=1.4, zorder=4,
                        label=r"Initial Gaussian ($s_0$, $2s_0$)" if radius == 1 else "_nolegend_"))
                axis.scatter(*initial_mean, marker="+", s=55, color="#13855c",
                             linewidths=1.5, zorder=4, label="Initial mean")
            elif not score_history:
                axis.text(0.03, 0.19, "Initial distribution not recorded",
                          transform=axis.transAxes, fontsize=9, color="0.4")
            color_label = f"{reference_source.capitalize()}-data selections per sample"
            if neighbors > 1:
                color_label += f" (k={neighbors})"
            colorbar = figure.colorbar(colors, ax=axis, orientation="horizontal", pad=0.02,
                                       fraction=0.05, ticks=ticks)
            colorbar.set_label(color_label + "; log color scale", fontsize=9)
            colorbar.ax.tick_params(labelsize=8)
            step = int(history["iterations"][snapshot_index])
            if history["sample_source"] == "training":
                title = f"Training batch for update {step} (before update)"
            elif score_history:
                title = f"Update {step}: score-model generated samples"
            else:
                title = f"Update {step}: validation-generated samples"
            title = (f"{history_path.parent.name}: {title}\n"
                     f"{n_generated:,} generated samples; {n_reference:,} {reference_source} data points")
            if len(generated) < n_generated or len(reference) < n_reference:
                title += f"\nDisplaying {len(generated):,} samples and {len(reference):,} data points"
            if drift_grids is not None or dimension > 2:
                figure.suptitle(title, fontsize=11)
                title = ("Samples, drift, and diffusion in normalized coordinates"
                         if diffusion_available[snapshot_index]
                         else "Samples and drift in normalized coordinates" if drift_grids is not None
                         else "Samples in normalized coordinates")
            if dimension > 2:
                title += f"; projection {pair + 1}"
            axis.set(title=title,
                     xlim=(-2, 2) if fixed_view else (lower[first], upper[first]),
                     ylim=(-2, 2) if fixed_view else (lower[first + 1], upper[first + 1]),
                     xlabel=r"$z_1 = x_1/R$" if dimension == 2 else f"Projected coordinate {first + 1}",
                     ylabel=r"$z_2 = x_2/R$" if dimension == 2 else f"Projected coordinate {first + 2}")
            axis.title.set_fontsize(10)
            axis.set_aspect("equal", adjustable="box")
            if fixed_view:
                axis.set(xticks=(-2, -1, 0, 1, 2), yticks=(-2, -1, 0, 1, 2))
                axis.set_box_aspect(1)
            handles, labels = axis.get_legend_handles_labels()
            if "Sample paths" in labels:
                handles, labels = handles[1:] + handles[:1], labels[1:] + labels[:1]
            legend = axis.legend(handles, labels, loc="upper right", markerscale=2, fontsize=9)
            for handle in legend.legend_handles:
                handle.set_alpha(1)
        destination = output_path / f"snapshot-{output_slot:02d}.png"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output_path, prefix=".sample-history-",
                                             suffix=".png", delete=False) as stream:
                temporary = Path(stream.name)
            figure.savefig(temporary, format="png", dpi=160)
            replace_with_retry(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            figure.clear()
    # Refresh only our fixed output slots; never remove the output directory.
    if iteration is None:
        for unused_slot in range(len(selected) + 1, DEFAULT_SNAPSHOT_COUNT + 1):
            (output_path / f"snapshot-{unused_slot:02d}.png").unlink(missing_ok=True)
    return output_path


def record_sample_history(run_dir, iteration, generated_normalized,
                          reference_normalized, *, data_hash, trajectories_normalized=None,
                          sample_source="validation", neighbors=1, selection_counts=None,
                          drift_grid_normalized=None, drift_normalized=None,
                          diffusion_normalized=None, projection_center=None, projection_basis=None,
                          initial_mean_normalized=None, initial_s0=None):
    """Save samples and optional paths without plotting; return the archive path.

    Duplicate updates replace their snapshots.
    If replacement paths are omitted, old paths survive only while their endpoints
    match the replacement samples. Missing historical paths are never reconstructed.
    Drift grids and vectors must be supplied together. Replacing a snapshot without
    drift removes its old drift, since samples alone do not identify model parameters.
    Optional diffusion amplitudes use the same grid and require the drift fields.
    Omitting diffusion on replacement removes its old values for the same reason.
    Samples, paths, and assignment counts retain every coordinate. For dimensions
    above two, field grids contain one full-state slice per displayed coordinate
    pair, with omitted projection coordinates fixed at zero. The saved projection
    frame affects only rendering and must stay fixed within a history.
    Initial Gaussian means and scalar standard deviations must be supplied together.
    Omitting them on replacement clears the old distribution for that snapshot.
    Source "score" records independently generated score-model samples against
    training data; their assignment counts are descriptive, not update weights.
    """
    iteration = _iteration(iteration)
    generated = _points(generated_normalized, "generated_normalized")
    reference = _points(reference_normalized, "reference_normalized")
    if generated.shape[-1] != reference.shape[-1]:
        raise ValueError("Generated and reference dimensions must match")
    if (initial_mean_normalized is None) != (initial_s0 is None):
        raise ValueError("Initial mean and s0 must be supplied together")
    initial_mean, initial_scale = ((None, None) if initial_mean_normalized is None
                                   else _initial_distribution(
                                       initial_mean_normalized, initial_s0, generated.shape[-1]))
    path = Path(run_dir) / "sample_history.npz"
    history = read_sample_history(path) if path.exists() else None
    center, basis = _history_projection(history, generated.shape[-1], projection_center, projection_basis)
    sample_source, neighbors = _source_and_neighbors(sample_source, neighbors, len(generated))
    if selection_counts is None:
        if sample_source == "training":
            raise ValueError("Training history requires the actual gradient assignment counts")
        selection_counts = _count_matches(reference, generated, neighbors)
    counts = _selection_counts(selection_counts, (len(generated),), len(reference), neighbors)
    trajectories = (None if trajectories_normalized is None
                    else _trajectories(trajectories_normalized, generated))
    if (drift_grid_normalized is None) != (drift_normalized is None):
        raise ValueError("Drift grid and vectors must be supplied together")
    drift_grid, drift = ((None, None) if drift_grid_normalized is None
                         else _drift_field(drift_grid_normalized, drift_normalized,
                                           projection_center=center, projection_basis=basis))
    if drift_grid is not None and drift_grid.shape[-1] != generated.shape[-1]:
        raise ValueError("Drift grid and samples must have the same dimension")
    if diffusion_normalized is not None and drift_grid is None:
        raise ValueError("Recording diffusion requires the drift grid and vectors")
    diffusion = (None if diffusion_normalized is None
                 else _diffusion_field(diffusion_normalized, drift_grid))
    if not isinstance(data_hash, str) or not data_hash:
        raise ValueError("data_hash must be a nonempty string")
    if history is not None:
        _check_reference(history, reference, data_hash)
        if history["sample_source"] != sample_source or history["neighbors"] != neighbors:
            raise ValueError("Sample source and neighbor count must stay fixed within sample history")
        if history["generated_normalized"].shape[1:] != generated.shape:
            raise ValueError("Generated sample count must stay fixed within sample history")
        keep = history["iterations"] != iteration
        steps = np.append(history["iterations"][keep], iteration)
        samples = np.concatenate((history["generated_normalized"][keep], generated[None]), axis=0)
        order = np.argsort(steps)
        old_counts = history.get("selection_counts")
        if old_counts is None:
            old_counts = np.stack([_count_matches(reference, points, neighbors)
                                   for points in history["generated_normalized"]])
        history["selection_counts"] = np.concatenate((old_counts[keep], counts[None]), axis=0)[order]
        old_paths = history.get("trajectories_normalized")
        if old_paths is not None or trajectories is not None:
            if old_paths is None:
                old_paths = np.zeros((len(keep),) + trajectories.shape, dtype=trajectories.dtype)
                history["trajectory_available"] = np.zeros(len(keep), dtype=bool)
            elif trajectories is not None and old_paths.shape[1:] != trajectories.shape:
                raise ValueError("Trajectory count and time grid must stay fixed within sample history")
            new_paths = np.zeros(old_paths.shape[1:], dtype=old_paths.dtype)
            new_available = False
            if trajectories is not None:
                new_paths, new_available = trajectories, True
            elif np.any(~keep):
                duplicate = np.flatnonzero(~keep)[0]
                candidate = old_paths[duplicate]
                if (history["trajectory_available"][duplicate]
                        and np.allclose(candidate[:, -1], generated[:len(candidate)],
                                        rtol=1e-5, atol=1e-6)):
                    new_paths, new_available = candidate, True
            history["trajectories_normalized"] = np.concatenate(
                (old_paths[keep], new_paths[None]), axis=0)[order]
            history["trajectory_available"] = np.append(
                history["trajectory_available"][keep], new_available)[order]
        old_grids = history.get("drift_grid_normalized")
        if old_grids is not None or drift_grid is not None:
            if old_grids is None:
                old_grids = np.zeros((len(keep),) + drift_grid.shape, dtype=drift_grid.dtype)
                old_drifts = np.zeros((len(keep),) + drift.shape, dtype=drift.dtype)
                history["drift_available"] = np.zeros(len(keep), dtype=bool)
            else:
                old_drifts = history["drift_normalized"]
                if drift_grid is not None and old_grids.shape[1:] != drift_grid.shape:
                    raise ValueError("Drift grid shape must stay fixed within sample history")
            new_grid = (np.zeros(old_grids.shape[1:], dtype=old_grids.dtype)
                        if drift_grid is None else drift_grid)
            new_drift = (np.zeros(old_drifts.shape[1:], dtype=old_drifts.dtype)
                         if drift is None else drift)
            history["drift_grid_normalized"] = np.concatenate(
                (old_grids[keep], new_grid[None]), axis=0)[order]
            history["drift_normalized"] = np.concatenate(
                (old_drifts[keep], new_drift[None]), axis=0)[order]
            history["drift_available"] = np.append(
                history["drift_available"][keep], drift is not None)[order]
        old_diffusion = history.get("diffusion_normalized")
        if old_diffusion is not None or diffusion is not None:
            if old_diffusion is None:
                old_diffusion = np.zeros((len(keep),) + diffusion.shape, dtype=diffusion.dtype)
                history["diffusion_available"] = np.zeros(len(keep), dtype=bool)
            elif diffusion is not None and old_diffusion.shape[1:] != diffusion.shape:
                raise ValueError("Diffusion grid shape must stay fixed within sample history")
            new_diffusion = (np.zeros(old_diffusion.shape[1:], dtype=old_diffusion.dtype)
                             if diffusion is None else diffusion)
            history["diffusion_normalized"] = np.concatenate(
                (old_diffusion[keep], new_diffusion[None]), axis=0)[order]
            history["diffusion_available"] = np.append(
                history["diffusion_available"][keep], diffusion is not None)[order]
        old_means = history.get("initial_mean_normalized")
        if old_means is not None or initial_mean is not None:
            if old_means is None:
                old_means = np.zeros((len(keep), generated.shape[-1]), dtype=initial_mean.dtype)
                old_scales = np.zeros(len(keep), dtype=initial_scale.dtype)
                history["initial_distribution_available"] = np.zeros(len(keep), dtype=bool)
            else:
                old_scales = history["initial_s0"]
            new_mean = (np.zeros(generated.shape[-1], dtype=old_means.dtype)
                        if initial_mean is None else initial_mean)
            new_scale = 0 if initial_scale is None else initial_scale
            history["initial_mean_normalized"] = np.concatenate(
                (old_means[keep], new_mean[None]), axis=0)[order]
            history["initial_s0"] = np.append(old_scales[keep], new_scale)[order]
            history["initial_distribution_available"] = np.append(
                history["initial_distribution_available"][keep], initial_mean is not None)[order]
        history["iterations"] = steps[order]
        history["generated_normalized"] = samples[order]
    else:
        history = {"iterations": np.array([iteration], dtype=np.int64),
                   "generated_normalized": generated[None],
                   "reference_normalized": reference, "data_hash": data_hash,
                   "sample_source": sample_source, "neighbors": neighbors,
                   "selection_counts": counts[None],
                   "projection_center": center, "projection_basis": basis}
        if trajectories is not None:
            history["trajectories_normalized"] = trajectories[None]
            history["trajectory_available"] = np.array([True])
        if drift_grid is not None:
            history["drift_grid_normalized"] = drift_grid[None]
            history["drift_normalized"] = drift[None]
            history["drift_available"] = np.array([True])
        if diffusion is not None:
            history["diffusion_normalized"] = diffusion[None]
            history["diffusion_available"] = np.array([True])
        if initial_mean is not None:
            history["initial_mean_normalized"] = initial_mean[None]
            history["initial_s0"] = initial_scale[None]
            history["initial_distribution_available"] = np.array([True])
    _save_archive(path, history)
    return path


def inherit_sample_history(source_run_dir, dest_run_dir, through_iteration,
                           reference_normalized, data_hash, *, sample_source="validation", neighbors=1,
                           projection_center=None, projection_basis=None):
    """Copy snapshots through a resumed checkpoint without plotting; return their count."""
    through_iteration = _iteration(through_iteration)
    reference = _points(reference_normalized, "reference_normalized")
    source = Path(source_run_dir) / "sample_history.npz"
    destination = Path(dest_run_dir) / "sample_history.npz"
    if source.resolve() == destination.resolve():
        raise ValueError("Resumed sample history must be written to a new run directory")
    if destination.exists():
        raise FileExistsError(f"Destination already has sample history: {destination}")
    if not source.exists():
        return 0
    history = read_sample_history(source)
    _history_projection(history, reference.shape[-1], projection_center, projection_basis)
    if sample_source not in ("training", "validation", "score"):
        raise ValueError("Sample source must be training, validation, or score")
    if history["data_hash"] != data_hash:
        raise ValueError("Sample history data fingerprint does not match the current dataset")
    if history["sample_source"] != sample_source:
        # Validation endpoints from older runs are not historical training batches.
        return 0
    _source_and_neighbors(sample_source, neighbors, history["generated_normalized"].shape[1])
    if history["neighbors"] != neighbors:
        raise ValueError("Sample history neighbor count does not match the resumed configuration")
    _check_reference(history, reference, data_hash)
    keep = history["iterations"] <= through_iteration
    if not np.any(keep):
        return 0
    history["iterations"] = history["iterations"][keep]
    history["generated_normalized"] = history["generated_normalized"][keep]
    if "selection_counts" in history:
        history["selection_counts"] = history["selection_counts"][keep]
    if "trajectories_normalized" in history:
        history["trajectories_normalized"] = history["trajectories_normalized"][keep]
        history["trajectory_available"] = history["trajectory_available"][keep]
    if "drift_normalized" in history:
        for field in ("drift_grid_normalized", "drift_normalized", "drift_available"):
            history[field] = history[field][keep]
    if "diffusion_normalized" in history:
        for field in ("diffusion_normalized", "diffusion_available"):
            history[field] = history[field][keep]
    if "initial_mean_normalized" in history:
        for field in ("initial_mean_normalized", "initial_s0", "initial_distribution_available"):
            history[field] = history[field][keep]
    _save_archive(destination, history)
    return int(np.count_nonzero(keep))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, help="Run directory or sample_history.npz")
    parser.add_argument("--output", type=Path, help="Output directory; defaults to sample_history/")
    parser.add_argument("--training-reference", action="store_true",
                        help="Compare legacy validation samples with training data from data.npz")
    args = parser.parse_args(argv)
    try:
        output = save_sample_history_plot(args.history, args.output,
                                         training_reference=args.training_reference)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Could not plot sample history: {exc}\n")
    if output is None:
        print("No recorded sample snapshots yet; no plot written.")
    else:
        print(f"Saved individual sample history plots to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

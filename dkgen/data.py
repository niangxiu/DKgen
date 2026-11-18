"""Gaussian-mixture data in two or more dimensions.

Sample arrays are NumPy float64 arrays in host memory. Normalization
uses the training radius only: ``normalized = raw / R`` with no centering.
The scalar integer mixture_modes records the target used to draw the splits.
"""

from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp


_SPLITS = ("train", "val", "test")


def mixture_parameters(modes=4):
    """Return raw-unit weights, means, and covariances for the requested modes."""
    modes = _positive_integer(modes, "modes")
    j = np.arange(modes, dtype=np.float64)
    weights = (j + 1.0) / (modes * (modes + 1.0) / 2.0)
    theta = 2.0 * np.pi * j / modes
    means = 3.0 * np.column_stack((np.cos(theta), np.sin(theta)))
    angle = theta + np.pi / 6.0
    rotation = np.empty((modes, 2, 2), dtype=np.float64)
    rotation[:, 0, 0] = np.cos(angle)
    rotation[:, 0, 1] = -np.sin(angle)
    rotation[:, 1, 0] = np.sin(angle)
    rotation[:, 1, 1] = np.cos(angle)
    variances = np.column_stack(((0.35 + 0.05 * j) ** 2,
                                 (0.15 + 0.02 * j) ** 2))
    covariances = (rotation * variances[:, None, :]) @ rotation.transpose(0, 2, 1)
    return weights, means, covariances


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def make_mixture_data(seed: int = 0, n_train: int = 10000,
                      n_val: int = 1000, n_test: int = 10000, *, modes: int = 4,
                      dim: int = 2, rotation_seed: int = 1729) -> dict:
    """Draw independent splits and scale them by the largest training norm.

    Separate SeedSequence children make each split reproducible independently
    of the sizes requested for the other splits.  Validation and test points
    are allowed to lie outside the normalized unit ball. Two dimensions retain
    the circular target and its exact drawing convention. Higher dimensions
    use the correlated four-component target, with an independent rotation seed.
    """
    counts = [_positive_integer(n_train, "n_train"),
              _positive_integer(n_val, "n_val"),
              _positive_integer(n_test, "n_test")]
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    dim = _positive_integer(dim, "dim")
    modes = _positive_integer(modes, "modes")
    if dim == 2:
        weights, means, covariances = mixture_parameters(modes)
    else:
        if modes != 4:
            raise ValueError("The correlated higher-dimensional target requires four modes")
        weights, means, covariances = correlated_mixture_parameters(dim, rotation_seed)
    factors = np.linalg.cholesky(covariances)
    streams = np.random.SeedSequence(int(seed)).spawn(3)
    result = {"mixture_modes": np.int64(modes)}
    for name, count, stream in zip(_SPLITS, counts, streams):
        rng = np.random.default_rng(stream)
        components = rng.choice(len(weights), size=count, p=weights)
        standard_normal = rng.standard_normal((count, dim))
        result[f"{name}_raw"] = means[components] + np.einsum(
            "nij,nj->ni", factors[components], standard_normal)
    radius = float(np.linalg.norm(result["train_raw"], axis=1).max())
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("training radius must be finite and positive")
    result["R"] = radius
    for name in _SPLITS:
        result[name] = result[f"{name}_raw"] / radius
    return result


def _validated_data(data):
    required = {"R", *_SPLITS, *(f"{name}_raw" for name in _SPLITS)}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"data is missing required keys: {sorted(missing)}")
    radius_array = np.asarray(data["R"])
    if radius_array.ndim != 0 or np.iscomplexobj(radius_array):
        raise ValueError("R must be a finite positive scalar")
    radius = float(radius_array)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("R must be a finite positive scalar")
    result = {"R": radius}
    # Preserve absence in legacy archives so their saved fingerprints stay valid.
    if "mixture_modes" in data:
        modes = np.asarray(data["mixture_modes"])
        if modes.shape != () or modes.dtype.kind not in "iu":
            raise ValueError("mixture_modes must be a positive scalar integer")
        result["mixture_modes"] = np.int64(_positive_integer(modes.item(), "mixture_modes"))
    dimension = None
    for name in _SPLITS:
        for key in (name, f"{name}_raw"):
            value = np.asarray(data[key])
            if np.iscomplexobj(value):
                raise ValueError(f"{key} must contain real coordinates")
            value = np.asarray(value, dtype=np.float64)
            if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
                raise ValueError(f"{key} must have shape (N, dim), with N > 0 and dim > 0")
            if dimension is None:
                dimension = value.shape[1]
            elif value.shape[1] != dimension:
                raise ValueError(f"{key} must have {dimension} coordinates per point")
            if not np.isfinite(value).all():
                raise ValueError(f"{key} must contain finite coordinates")
            result[key] = value
        if result[name].shape != result[f"{name}_raw"].shape:
            raise ValueError(f"{name} and {name}_raw must have the same shape")
        if not np.allclose(result[name] * radius, result[f"{name}_raw"],
                           rtol=1e-12, atol=1e-12):
            raise ValueError(f"{name} does not match {name}_raw / R")
    training_radius = np.linalg.norm(result["train_raw"], axis=1).max()
    if not np.isclose(radius, training_radius, rtol=1e-12, atol=1e-12):
        raise ValueError("R must equal the largest training-sample norm")
    return result


def save_data(path, data):
    """Save validated splits to a numeric NPZ archive at exactly ``path``."""
    validated = _validated_data(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez_compressed(file, **validated)


def load_data(path) -> dict:
    """Load and validate saved splits without allowing pickled objects."""
    with np.load(Path(path), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    return _validated_data(data)


def _integer(value, name, minimum):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def correlated_mixture_parameters(dim=8, rotation_seed=1729):
    """Return four fixed-weight Gaussian components in ``dim >= 2`` dimensions.

    Component j has weight (j+1)/10. Before rotation, its first two coordinates
    use the existing target and its remaining coordinates have covariance
    ``v_j * rho_j ** abs(a-b)``, where ``v_j`` is half the trace of the existing
    covariance and ``rho_j = 0.2 + 0.1*j``. Extra mean coordinates are zero.
    The same orthogonal matrix rotates every component. Its independent seed
    fixes the target geometry across independent sample draws.

    With dim=2, return the existing four-mode parameters without rotation or
    any numerical changes. No global random state is used or modified.
    """
    dim = _integer(dim, "dim", 2)
    rotation_seed = _integer(rotation_seed, "rotation_seed", 0)
    weights, original_means, original_covariances = mixture_parameters(4)
    if dim == 2:
        return weights, original_means, original_covariances

    means = np.zeros((4, dim), dtype=np.float64)
    means[:, :2] = original_means
    covariances = np.zeros((4, dim, dim), dtype=np.float64)
    covariances[:, :2, :2] = original_covariances
    offsets = np.abs(np.arange(dim - 2)[:, None] - np.arange(dim - 2)[None, :])
    for component in range(4):
        variance = np.trace(original_covariances[component]) / 2.0
        rho = 0.2 + 0.1 * component
        covariances[component, 2:, 2:] = variance * rho ** offsets

    rng = np.random.default_rng(rotation_seed)
    rotation, triangular = np.linalg.qr(rng.standard_normal((dim, dim)))
    # Fix the QR sign convention so a fixed seed specifies one rotation.
    signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
    rotation *= signs[None, :]
    means = means @ rotation.T
    covariances = rotation @ covariances @ rotation.T
    return weights, means, covariances


def target_metadata(dim, rotation_seed, R, *, modes=4):
    """Describe a target in the numeric format used by saved target.npz files.

    Raw parameters determine the target geometry. Normalized parameters use
    the same training-set radius R as every data split and generated sample.
    """
    dim = _integer(dim, "dim", 2)
    rotation_seed = _integer(rotation_seed, "rotation_seed", 0)
    modes = _integer(modes, "modes", 1)
    radius = np.asarray(R)
    if (radius.shape != () or radius.dtype.kind not in "fiu"
            or not np.isfinite(radius) or radius <= 0):
        raise ValueError("R must be a finite positive scalar")
    radius = float(radius)
    if dim == 2:
        weights, means, covariances = mixture_parameters(modes)
    else:
        if modes != 4:
            raise ValueError("The correlated higher-dimensional target requires four modes")
        weights, means, covariances = correlated_mixture_parameters(dim, rotation_seed)
    return {
        "weights": weights, "means_raw": means, "covariances_raw": covariances,
        "means_normalized": means / radius,
        "covariances_normalized": covariances / radius**2,
        "rotation_seed": np.int64(rotation_seed), "R": radius,
    }


def validate_target_metadata(data, target, dim):
    """Validate a saved target against its data and return its rotation seed.

    Validation never regenerates or replaces the saved arrays. The seeded
    target reconstruction checks that its documented geometry agrees with
    the numeric parameters used for diagnostics.
    """
    dim = _integer(dim, "dim", 2)
    required = {"weights", "means_raw", "covariances_raw", "means_normalized",
                "covariances_normalized", "rotation_seed", "R"}
    missing = required.difference(target)
    if missing:
        raise ValueError(f"Saved benchmark target is missing keys: {sorted(missing)}")
    raw = _validated_parameters(target["weights"], target["means_raw"],
                                target["covariances_raw"])
    normalized = _validated_parameters(target["weights"], target["means_normalized"],
                                       target["covariances_normalized"])
    modes = len(raw[0])
    if raw[1].shape[1] != dim or normalized[1].shape[1] != dim:
        raise ValueError("Saved target components must match the state dimension")
    if dim != 2 and modes != 4:
        raise ValueError("Benchmark target must have four components matching the state dimension")
    if "mixture_modes" in data and int(data["mixture_modes"]) != modes:
        raise ValueError("Saved target mode count must match the dataset")
    for split in ("train", "val", "test"):
        if split in data:
            points = np.asarray(data[split])
            if points.ndim != 2 or points.shape[1] != dim:
                raise ValueError("Saved target dimension must match the dataset")
    radius = np.asarray(target["R"])
    data_radius = np.asarray(data.get("R"))
    if (radius.shape != () or radius.dtype.kind not in "fiu"
            or not np.isfinite(radius) or float(radius) <= 0
            or data_radius.shape != () or data_radius.dtype.kind not in "fiu"
            or not np.isfinite(data_radius) or float(data_radius) <= 0
            or not np.isclose(float(radius), float(data_radius), rtol=1e-12, atol=1e-12)):
        raise ValueError("Benchmark target R must match the dataset normalization")
    if (not np.allclose(raw[1] / radius, normalized[1], rtol=1e-12, atol=1e-12)
            or not np.allclose(raw[2] / radius**2, normalized[2], rtol=1e-12, atol=1e-12)):
        raise ValueError("Benchmark target raw and normalized parameters disagree")
    seed_array = np.asarray(target["rotation_seed"])
    if seed_array.shape != () or seed_array.dtype.kind not in "iu" or int(seed_array) < 0:
        raise ValueError("rotation_seed must be a nonnegative scalar integer")
    rotation_seed = int(seed_array)
    expected = (mixture_parameters(modes) if dim == 2 else
                correlated_mixture_parameters(dim, rotation_seed))
    if any(not np.allclose(actual, reference, rtol=1e-12, atol=1e-12)
           for actual, reference in zip(raw[:3], expected)):
        raise ValueError("Saved target differs from the benchmark's seeded correlated mixture")
    return rotation_seed


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numbers")
    return array


def _validated_parameters(weights, means, covariances):
    weights = _real_array(weights, "weights")
    means = _real_array(means, "means")
    covariances = _real_array(covariances, "covariances")
    if weights.ndim != 1 or weights.size == 0 or np.any(weights <= 0.0):
        raise ValueError("weights must be a nonempty vector of positive numbers")
    if not np.isclose(weights.sum(), 1.0, rtol=1e-12, atol=1e-12):
        raise ValueError("weights must sum to one")
    if means.ndim != 2 or means.shape[0] != len(weights) or means.shape[1] == 0:
        raise ValueError("means must have shape (number of components, dimension)")
    dim = means.shape[1]
    if covariances.shape != (len(weights), dim, dim):
        raise ValueError("covariances must have shape (number of components, dimension, dimension)")
    if not np.allclose(covariances, covariances.transpose(0, 2, 1),
                       rtol=1e-12, atol=1e-14):
        raise ValueError("covariances must be symmetric")
    try:
        factors = np.linalg.cholesky(covariances)
    except np.linalg.LinAlgError as error:
        raise ValueError("covariances must be positive definite") from error
    return weights, means, covariances, factors


def sample_mixture(rng, count, weights, means, covariances):
    """Draw ``count`` raw-coordinate points using the supplied NumPy Generator."""
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    count = _integer(count, "count", 1)
    weights, means, _, factors = _validated_parameters(weights, means, covariances)
    components = rng.choice(len(weights), size=count, p=weights)
    noise = rng.standard_normal((count, means.shape[1]))
    return means[components] + np.einsum("nij,nj->ni", factors[components], noise)


def mixture_log_density(points, weights, means, covariances):
    """Return exact mixture log densities at raw points of shape (N, dimension).

    Cholesky factors evaluate Gaussian log densities without matrix inversion;
    log-sum-exp combines components without underflowing their densities.
    """
    weights, means, _, factors = _validated_parameters(weights, means, covariances)
    points = _real_array(points, "points")
    if points.ndim != 2 or points.shape[1] != means.shape[1]:
        raise ValueError("points must have shape (N, dimension) matching the mixture")
    normalizer = 0.5 * means.shape[1] * np.log(2.0 * np.pi)
    component_logs = np.empty((len(weights), len(points)), dtype=np.float64)
    for component, (weight, mean, factor) in enumerate(zip(weights, means, factors)):
        whitened = solve_triangular(factor, (points - mean).T, lower=True)
        component_logs[component] = (
            np.log(weight) - normalizer - np.log(np.diag(factor)).sum()
            - 0.5 * np.sum(whitened * whitened, axis=0))
    return logsumexp(component_logs, axis=0)

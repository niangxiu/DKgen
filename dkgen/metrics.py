"""Sample diagnostics and exact nearest-neighbor terminal weights."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

from .data import _validated_parameters, mixture_log_density
from .data import mixture_parameters
from .density_ratio import averaged_kl_estimate


def _points(value, name, dimension=None):
    points = np.asarray(value)
    if np.iscomplexobj(points):
        raise ValueError(f"{name} must contain real coordinates")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError(f"{name} must have shape (N, d), with N > 0 and d > 0")
    if dimension is not None and points.shape[1] != dimension:
        raise ValueError(f"{name} must have {dimension} coordinates per point")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain finite coordinates")
    return points


def nearest_neighbor_weights(data, endpoints, k=1) -> np.ndarray:
    """Return endpoint masses after equal-weight exact k-nearest assignment.

    Each observed data point distributes mass ``1 / len(data)`` equally over
    its ``k`` nearest endpoints.  Output has length ``len(endpoints)`` and
    total mass one.  cKDTree uses exact Euclidean queries (``eps=0``); any
    selection among endpoints tied at the kth distance is a valid assignment.
    Data and endpoints must use the same coordinate units.
    """
    endpoints = _points(endpoints, "endpoints")
    data = _points(data, "data", dimension=endpoints.shape[1])
    if (isinstance(k, (bool, np.bool_))
            or not isinstance(k, (int, np.integer))
            or not 1 <= k <= len(endpoints)):
        raise ValueError("k must be an integer between 1 and the number of endpoints")
    _, indices = cKDTree(endpoints).query(data, k=int(k), eps=0.0, p=2)
    counts = np.bincount(np.asarray(indices).ravel(), minlength=len(endpoints))
    return counts.astype(np.float64) / (len(data) * int(k))


def knn_precision(generated, reference, *, k=3) -> dict:
    """Measure generated membership in the reference k-neighbor ball union.

    A reference point's closed ball has radius equal to its kth nearest
    *other* reference point. Duplicate observations remain distinct samples.
    Each generated point is accepted if any reference ball contains it,
    including a ball whose center is not its nearest reference point.
    Inputs are converted to float64 and must use the same coordinate units.
    Exact tree queries process one ball at a time, avoiding an N-by-M distance
    matrix. Nearest-data distances are unsquared Euclidean distances to the
    reference samples, not to ball boundaries. Report their overall mean and
    their mean and linearly interpolated 95th percentile among outside points.
    Conditional distances are None when no points are outside or fewer than
    k+1 reference points make precision unavailable; the overall mean remains
    available with any nonempty reference set.
    """
    generated = _points(generated, "generated")
    reference = _points(reference, "reference", dimension=generated.shape[1])
    if (isinstance(k, (bool, np.bool_))
            or not isinstance(k, (int, np.integer)) or k < 1):
        raise ValueError("k must be a positive integer")
    k = int(k)
    reference_tree = cKDTree(reference)
    nearest_distances, _ = reference_tree.query(generated, k=1, eps=0.0, p=2)
    distance_mean = float(np.mean(nearest_distances))
    if len(reference) <= k:
        return {
            "knn_precision": None,
            "knn_outside_fraction": None,
            "knn_outside_count": None,
            "nearest_data_distance_mean": distance_mean,
            "knn_outside_distance_mean": None,
            "knn_outside_distance_p95": None,
            "precision_k": k,
            "knn_precision_unavailable_reason": (
                f"Self-excluded kNN precision needs at least {k + 1} "
                f"reference points; received {len(reference)}"),
        }

    # The (k+1)st distance includes one zero self-distance. This remains the
    # kth OTHER distance when ties or duplicate observations are present.
    _, neighbors = reference_tree.query(reference, k=[k + 1], eps=0.0, p=2)
    offsets = reference - reference[neighbors[:, 0]]
    radii_squared = np.sum(offsets * offsets, axis=1)
    generated_tree = cKDTree(generated)
    accepted = np.zeros(len(generated), dtype=bool)
    for center, radius_squared in zip(reference, radii_squared):
        # Pad only the candidate query by one ULP. The explicit squared
        # comparison below retains the closed boundary without a tolerance.
        radius = np.nextafter(np.sqrt(radius_squared), np.inf)
        candidates = np.asarray(generated_tree.query_ball_point(
            center, radius, eps=0.0, p=2), dtype=np.intp)
        candidates = candidates[~accepted[candidates]]
        if radius_squared == 0.0:
            inside = np.all(generated[candidates] == center, axis=1)
        else:
            offsets = generated[candidates] - center
            inside = np.sum(offsets * offsets, axis=1) <= radius_squared
        accepted[candidates[inside]] = True

    outside_count = int(np.count_nonzero(~accepted))
    outside_fraction = outside_count / len(generated)
    outside_distances = nearest_distances[~accepted]
    return {
        "knn_precision": 1.0 - outside_fraction,
        "knn_outside_fraction": outside_fraction,
        "knn_outside_count": outside_count,
        "nearest_data_distance_mean": distance_mean,
        "knn_outside_distance_mean": (float(np.mean(outside_distances))
                                     if outside_count else None),
        "knn_outside_distance_p95": (float(np.percentile(outside_distances, 95, method="linear"))
                                    if outside_count else None),
        "precision_k": k,
    }


def _moments(points):
    mean = points.mean(axis=0)
    centered = points - mean
    # Covariance of the empirical measure uses divisor N, including when N=1.
    covariance = centered.T @ centered / len(points)
    return mean, covariance


def _regions(points, means):
    squared_distances = np.sum((points[:, None, :] - means[None, :, :]) ** 2, axis=2)
    labels = np.argmin(squared_distances, axis=1)
    return labels, np.bincount(labels, minlength=len(means))


def _region_probabilities(points, means):
    _, counts = _regions(points, means)
    return counts.astype(np.float64) / len(points)


def _sliced_wasserstein1(generated, reference, directions):
    """Average empirical 1D distances along an explicitly supplied direction set."""
    distances = []
    for direction in directions:
        generated_projection = generated @ direction
        reference_projection = reference @ direction
        if len(generated) == len(reference):
            distance = np.mean(np.abs(np.sort(generated_projection)
                                      - np.sort(reference_projection)))
        else:
            distance = wasserstein_distance(generated_projection, reference_projection)
        distances.append(distance)
    return float(np.mean(distances))


def _normalization_scale(value):
    scale = np.asarray(value)
    if (scale.shape != () or scale.dtype.kind not in "fiu"
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError("normalization_scale must be a finite positive scalar")
    return float(scale)


def evaluate_samples(generated_raw, reference_raw, *, mixture_modes=4,
                     n_neighbor=None, density_bandwidth=0.1, normalization_scale=1.0,
                     density_self_weight=0.5, target=None) -> dict:
    """Compare empirical distributions while preserving the example conventions.

    Inputs are always in raw units. With target=None, retain the original 2D
    metrics described below. With a target dictionary from target.npz, divide
    both populations by normalization_scale and report the full-dimensional
    normalized metrics, using its weights, normalized means and covariances.
    That path retains all-point KL and 128 seeded random directions. Use it
    for the higher-dimensional target; it requires n_neighbor=None or 0.

    Mean error is Euclidean, covariance error is Frobenius, and region error
    is L1 over the target's nearest-center regions. Covariances use divisor N.
    Region references are measured from reference_raw, not mixture weights.
    Sliced W1 averages 128 angles j*pi/128, j=0,...,127.  Equal sample counts
    use sorted-projection absolute differences; unequal counts use SciPy's
    exact one-dimensional empirical Wasserstein calculation.
    kNN precision uses k=3 and closed reference balls in float64 coordinates
    divided by normalization_scale; it is unavailable below four references.
    Nearest-data distance summaries use these same normalized coordinates.
    When n_neighbor is supplied (0 uses all points), estimate both KL directions using Gaussian
    densities in coordinates divided by normalization_scale. KL values are in
    nats, can be negative because of estimation error, and are not clipped.
    Zero self weight uses leave-one-out population normalization; positive
    self weights retain the full population denominator.
    """
    if target is not None:
        if (n_neighbor is not None and
                (isinstance(n_neighbor, (bool, np.bool_))
                 or not isinstance(n_neighbor, (int, np.integer)) or n_neighbor != 0)):
            raise ValueError("Full-dimensional target evaluation requires n_neighbor=0 or None")
        generated = _points(generated_raw, "generated_raw")
        reference = _points(reference_raw, "reference_raw", dimension=generated.shape[1])
        scale = _normalization_scale(normalization_scale)
        if "R" in target and not np.isclose(
                _normalization_scale(target["R"]), scale, rtol=1e-12, atol=1e-12):
            raise ValueError("target R must match normalization_scale")
        return _evaluate_target_samples(
            generated / scale, reference / scale,
            weights=target["weights"], means_normalized=target["means_normalized"],
            covariances_normalized=target["covariances_normalized"],
            density_bandwidth=density_bandwidth, density_self_weight=density_self_weight)
    generated = _points(generated_raw, "generated_raw", dimension=2)
    reference = _points(reference_raw, "reference_raw", dimension=2)
    scale = _normalization_scale(normalization_scale)
    generated_normalized = generated / scale
    reference_normalized = reference / scale
    generated_mean, generated_covariance = _moments(generated)
    reference_mean, reference_covariance = _moments(reference)
    _, means, _ = mixture_parameters(mixture_modes)
    generated_probabilities = _region_probabilities(generated, means)
    reference_probabilities = _region_probabilities(reference, means)
    directions = (np.array([np.cos(angle), np.sin(angle)])
                  for angle in np.arange(128, dtype=np.float64) * np.pi / 128.0)
    metrics = {
        "mean_error": float(np.linalg.norm(generated_mean - reference_mean)),
        "covariance_error": float(np.linalg.norm(
            generated_covariance - reference_covariance, ord="fro")),
        "region_probability_error": float(np.abs(
            generated_probabilities - reference_probabilities).sum()),
        "sliced_wasserstein1": _sliced_wasserstein1(generated, reference, directions),
        "generated_count": int(len(generated)),
        "reference_count": int(len(reference)),
        "generated_mean": generated_mean.tolist(),
        "reference_mean": reference_mean.tolist(),
        "generated_covariance": generated_covariance.tolist(),
        "reference_covariance": reference_covariance.tolist(),
        "covariance_divisor": "N",
        "generated_region_probabilities": generated_probabilities.tolist(),
        "reference_region_probabilities": reference_probabilities.tolist(),
        "region_centers": means.tolist(),
        "sliced_wasserstein_directions": 128,
    }
    metrics.update(knn_precision(generated_normalized, reference_normalized))
    metrics["precision_coordinates"] = "normalized"
    if n_neighbor is not None:
        if (isinstance(n_neighbor, (bool, np.bool_))
                or not isinstance(n_neighbor, (int, np.integer)) or n_neighbor < 0):
            raise ValueError("n_neighbor must be a nonnegative integer (0 uses all points)")
        bandwidth = np.asarray(density_bandwidth)
        if (bandwidth.shape != () or bandwidth.dtype.kind not in "fiu"
                or not np.isfinite(bandwidth) or bandwidth <= 0):
            raise ValueError("density_bandwidth must be a finite positive scalar")
        self_weight = np.asarray(density_self_weight)
        if (self_weight.shape != () or self_weight.dtype.kind not in "fiu"
                or not np.isfinite(self_weight) or not 0 <= self_weight <= 1):
            raise ValueError("density_self_weight must be a finite scalar in [0, 1]")
        metrics.update(kl_n_neighbor=int(n_neighbor),
                       kl_density_estimator="all_points" if n_neighbor == 0 else "nearest_neighbors",
                       kl_density_bandwidth=float(density_bandwidth),
                       kl_density_self_weight=float(self_weight),
                       kl_self_density_normalization=(
                           "leave_one_out" if self_weight == 0 else "full_population"),
                       kl_density_coordinates="normalized")
        minimum_count = max(n_neighbor + 1, 2 if self_weight == 0 else 1)
        if min(len(generated), len(reference)) < minimum_count:
            # Keep old/small evaluations usable without silently changing the
            # requested estimator or inventing a KL value.
            metrics.update(
                forward_kl=None, reverse_kl=None, averaged_kl=None,
                averaged_kl_unavailable_reason=(
                    f"Self-excluded KL estimation needs at least {minimum_count} "
                    "points in each set"))
        else:
            metrics.update(averaged_kl_estimate(
                reference_normalized,
                generated_normalized,
                n_neighbor, float(density_bandwidth), self_weight=float(self_weight)))
    return metrics


def _evaluate_target_samples(generated_normalized, reference_normalized, *,
                               weights, means_normalized, covariances_normalized,
                               density_bandwidth=0.1, density_self_weight=0.5):
    """Compare samples and exact target moments in normalized coordinates.

    Both sample populations and mixture parameters must already use the same
    normalized coordinates. KL uses the production all-point Gaussian KDE,
    including its self-weight convention, without clipping negative estimates.
    Sliced W1 averages 128 fixed seeded random unit directions in the full
    dimension. Empirical covariances use divisor N; covariance errors are
    Frobenius norms and mean errors are Euclidean norms.

    Nearest-center regions are geometric assignments, not component labels.
    Their mass error compares against the independent reference. The separately
    named comparison with nominal component weights is only an assignment
    diagnostic: overlapping components can cross region boundaries even under
    the exact target. Conditional region moments likewise compare generated and
    reference samples, not the latent Gaussian components. Conditional means
    need one point in both populations and covariances need at least two.

    Exact target log density gives cross-entropy, not KL: it does not contain
    the generated distribution's entropy. All returned values are JSON-safe.
    """
    weights, means, covariances, _ = _validated_parameters(
        weights, means_normalized, covariances_normalized)
    dimension = means.shape[1]
    generated = _points(generated_normalized, "generated_normalized", dimension)
    reference = _points(reference_normalized, "reference_normalized", dimension)
    # Validate density settings and use precisely the estimator used in training.
    kl = averaged_kl_estimate(reference, generated, 0, density_bandwidth,
                              self_weight=density_self_weight)
    generated_mean, generated_covariance = _moments(generated)
    reference_mean, reference_covariance = _moments(reference)
    target_mean = weights @ means
    centered_means = means - target_mean
    target_covariance = np.einsum("k,kij->ij", weights, covariances)
    target_covariance += np.einsum("k,ki,kj->ij", weights,
                                  centered_means, centered_means)

    generated_labels, generated_counts = _regions(generated, means)
    reference_labels, reference_counts = _regions(reference, means)
    generated_probabilities = generated_counts / len(generated)
    reference_probabilities = reference_counts / len(reference)
    region_mean_errors, region_covariance_errors = [], []
    for region in range(len(weights)):
        generated_region = generated[generated_labels == region]
        reference_region = reference[reference_labels == region]
        if len(generated_region) and len(reference_region):
            generated_region_mean, generated_region_covariance = _moments(generated_region)
            reference_region_mean, reference_region_covariance = _moments(reference_region)
            region_mean_errors.append(float(np.linalg.norm(
                generated_region_mean - reference_region_mean)))
            region_covariance_errors.append(
                float(np.linalg.norm(generated_region_covariance
                                     - reference_region_covariance, ord="fro"))
                if min(len(generated_region), len(reference_region)) >= 2 else None)
        else:
            region_mean_errors.append(None)
            region_covariance_errors.append(None)

    directions = np.random.default_rng(1729).standard_normal((128, dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    generated_log_density = float(np.mean(mixture_log_density(
        generated, weights, means, covariances)))
    reference_log_density = float(np.mean(mixture_log_density(
        reference, weights, means, covariances)))
    metrics = {
        "dimension": dimension,
        "coordinates": "normalized",
        "generated_count": int(len(generated)),
        "reference_count": int(len(reference)),
        "mean_error": float(np.linalg.norm(generated_mean - reference_mean)),
        "covariance_error": float(np.linalg.norm(
            generated_covariance - reference_covariance, ord="fro")),
        "exact_target_mean_error": float(np.linalg.norm(generated_mean - target_mean)),
        "exact_target_covariance_error": float(np.linalg.norm(
            generated_covariance - target_covariance, ord="fro")),
        "reference_exact_target_mean_error": float(np.linalg.norm(reference_mean - target_mean)),
        "reference_exact_target_covariance_error": float(np.linalg.norm(
            reference_covariance - target_covariance, ord="fro")),
        "generated_mean": generated_mean.tolist(),
        "reference_mean": reference_mean.tolist(),
        "exact_target_mean": target_mean.tolist(),
        "generated_covariance": generated_covariance.tolist(),
        "reference_covariance": reference_covariance.tolist(),
        "exact_target_covariance": target_covariance.tolist(),
        "covariance_divisor": "N",
        "region_probability_error": float(np.abs(
            generated_probabilities - reference_probabilities).sum()),
        "region_probability_error_vs_component_weights": float(np.abs(
            generated_probabilities - weights).sum()),
        "reference_region_probability_error_vs_component_weights": float(np.abs(
            reference_probabilities - weights).sum()),
        "generated_region_probabilities": generated_probabilities.tolist(),
        "reference_region_probabilities": reference_probabilities.tolist(),
        "nominal_component_weights": weights.tolist(),
        "generated_region_counts": generated_counts.tolist(),
        "reference_region_counts": reference_counts.tolist(),
        "region_mean_errors": region_mean_errors,
        "region_covariance_errors": region_covariance_errors,
        "region_centers": means.tolist(),
        "region_assignment": "nearest_center",
        "region_assignment_note": (
            "Nearest-center regions are not latent mixture components; component "
            "weights need not equal exact region probabilities."),
        "sliced_wasserstein1": _sliced_wasserstein1(generated, reference, directions),
        "sliced_wasserstein_directions": 128,
        "sliced_wasserstein_direction_seed": 1729,
        "generated_exact_target_log_density_mean": generated_log_density,
        "reference_exact_target_log_density_mean": reference_log_density,
        "generated_target_cross_entropy": -generated_log_density,
        "reference_target_cross_entropy": -reference_log_density,
        "kl_n_neighbor": 0,
        "kl_density_estimator": "all_points",
        "kl_density_bandwidth": float(density_bandwidth),
        "kl_density_self_weight": float(density_self_weight),
        "kl_self_density_normalization": (
            "leave_one_out" if density_self_weight == 0 else "full_population"),
        "kl_density_coordinates": "normalized",
        "precision_coordinates": "normalized",
    }
    metrics.update(kl)
    metrics.update(knn_precision(generated, reference))
    return metrics

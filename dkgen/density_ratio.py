"""Gaussian density ratios for the averaged KL response."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import logsumexp


def _points(value, name):
    value = np.asarray(value)
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real coordinates")
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) == 0:
        raise ValueError(f"{name} must have shape (N, M), with N and M positive")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite coordinates")
    return value


def _self_kernel_parameters(self_weight, n_sample):
    value = np.asarray(self_weight)
    if (value.shape != () or value.dtype.kind not in "fiu"
            or not np.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("self_weight must be a finite scalar in [0, 1]")
    if value == 0:
        if n_sample < 2:
            raise ValueError("self_weight=0 requires at least two samples for leave-one-out density")
        return -np.inf, n_sample - 1
    # Preserve the original full-population normalization for positive weights.
    return np.log(float(value)), n_sample


def _all_points_log_ratio(train, samples, bandwidth, log_self_weight, sample_normalization):
    """Sum every kernel contribution with bounded pairwise-distance storage."""
    n_data, n_sample = len(train), len(samples)
    log_ratio = np.empty(n_sample, dtype=np.float64)
    # Each distance block has at most about one million entries. Process the
    # two populations sequentially, including for 10,000-point evaluations.
    batch_size = max(1, min(128, 1_000_000 // max(n_data, n_sample)))
    log_count_ratio = np.log(float(sample_normalization)) - np.log(float(n_data))
    for start in range(0, n_sample, batch_size):
        stop = min(start + batch_size, n_sample)
        query = samples[start:stop]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            log_kernel = cdist(query, train, metric="sqeuclidean")
            # Divide twice rather than squaring a possibly tiny bandwidth.
            log_kernel /= bandwidth
            log_kernel /= bandwidth
            log_kernel *= -0.5
            data_log_sum = logsumexp(log_kernel, axis=1)
            del log_kernel
            log_kernel = cdist(query, samples, metric="sqeuclidean")
            log_kernel /= bandwidth
            log_kernel /= bandwidth
            log_kernel *= -0.5
            # Replace only the query's own peak; coincident other samples
            # still contribute their full Gaussian peaks.
            log_kernel[np.arange(stop - start), np.arange(start, stop)] = log_self_weight
            log_ratio[start:stop] = (
                log_count_ratio + data_log_sum - logsumexp(log_kernel, axis=1))
    if not np.isfinite(log_ratio).all():
        raise FloatingPointError(
            "Nonfinite Gaussian log density ratio; inspect endpoint coordinates "
            "and density bandwidth")
    return log_ratio


def gaussian_log_density_ratio(train, samples, n_neighbor, bandwidth, *, self_weight=0.5,
                               all_points_evaluator=None):
    """Return log(p_hat / h_hat) at the generated endpoints.

    With ``n_neighbor=0``, the numerator uses every training point and the
    denominator uses every other generated endpoint plus ``self_weight`` times
    the query's own kernel peak (default 0.5). Positive ``n_neighbor`` retains the estimate with
    that many nearest points from each population (other points for the
    denominator), plus the same self contribution. For positive self weights,
    normalizations remain ``1 / len(train)`` and ``1 / len(samples)``.
    With ``self_weight=0``, the self-density uses ``1 / (len(samples) - 1)``;
    the cross-density still uses ``1 / len(train)``. The query is excluded by index
    from neighbor selection; other endpoints at the same location remain
    eligible. ``bandwidth`` is the Gaussian standard deviation in normalized
    coordinates, separate from the SDE diffusion coefficient.

    Only logarithms are computed, so very small density estimates do not
    underflow and large density ratios need not be exponentiated.
    An optional all_points_evaluator performs the same validated all-point
    calculation on another device. Nearest-neighbor estimates use the CPU.
    """
    train = _points(train, "train")
    samples = _points(samples, "samples")
    if train.shape[1] != samples.shape[1]:
        raise ValueError("train and samples must have the same coordinate dimension")
    n_data, n_sample = len(train), len(samples)
    log_self_weight, sample_normalization = _self_kernel_parameters(self_weight, n_sample)
    if (isinstance(n_neighbor, (bool, np.bool_))
            or not isinstance(n_neighbor, (int, np.integer))
            or not 0 <= n_neighbor <= min(n_data, n_sample - 1)):
        raise ValueError(
            "n_neighbor must be 0 (all points), or a positive integer no larger than "
            "min(len(train), len(samples) - 1) for self-exclusion")
    n_neighbor = int(n_neighbor)
    bandwidth_array = np.asarray(bandwidth)
    if (bandwidth_array.shape != () or bandwidth_array.dtype.kind not in "fiu"
            or not np.isfinite(bandwidth_array) or bandwidth_array <= 0):
        raise ValueError("bandwidth must be a finite positive scalar")
    bandwidth = float(bandwidth_array)

    if n_neighbor == 0:
        evaluate = _all_points_log_ratio if all_points_evaluator is None else all_points_evaluator
        return evaluate(
            train, samples, bandwidth, log_self_weight, sample_normalization)

    train_tree, sample_tree = cKDTree(train), cKDTree(samples)
    log_ratio = np.empty(n_sample, dtype=np.float64)
    # Bound temporary nearest-neighbor storage; never construct pairwise
    # distances between every pair of generated samples.
    batch_size = max(1, min(128, 65536 // (n_neighbor + 1)))
    log_count_ratio = np.log(float(sample_normalization)) - np.log(float(n_data))
    for start in range(0, n_sample, batch_size):
        stop = min(start + batch_size, n_sample)
        query = samples[start:stop]
        data_distances, _ = train_tree.query(query, k=n_neighbor)
        data_distances = data_distances.reshape(len(query), n_neighbor)
        sample_distances, sample_indices = sample_tree.query(query, k=n_neighbor + 1)
        # Tied distances need not put the query itself first, or even include
        # it among k+1 duplicates. Remove only the matching original index.
        sample_distances[sample_indices == np.arange(start, stop)[:, None]] = np.inf
        sample_distances = np.partition(
            sample_distances, n_neighbor - 1, axis=1)[:, :n_neighbor]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            data_log_kernel = -0.5 * (data_distances / bandwidth) ** 2
            sample_log_kernel = -0.5 * (sample_distances / bandwidth) ** 2
            # Gaussian normalization cancels. A zero self weight is -inf here,
            # so only the requested other neighbors contribute in that mode.
            sample_log_sum = np.logaddexp(
                log_self_weight, logsumexp(sample_log_kernel, axis=1))
            log_ratio[start:stop] = (
                log_count_ratio + logsumexp(data_log_kernel, axis=1)
                - sample_log_sum)
    if not np.isfinite(log_ratio).all():
        raise FloatingPointError(
            "Nonfinite Gaussian log density ratio; inspect endpoint coordinates "
            "and density bandwidth")
    return log_ratio


def averaged_kl_estimate(reference, samples, n_neighbor, bandwidth, *,
                         sample_log_ratio=None, self_weight=0.5, all_points_evaluator=None):
    """Estimate both KL directions and their average in normalized coordinates.

    The forward direction averages log(p_hat/h_hat) over reference points,
    and the reverse direction averages log(h_hat/p_hat) over generated points.
    With ``n_neighbor=0``, each density uses its entire population, with
    ``self_weight`` for its query's own peak. Positive k uses nearest neighbors.
    Zero self weight uses population-minus-one normalization in each self-density;
    positive self weights retain full-population normalization. Positive k needs
    at least k+1 points in each set; zero self weight always needs at least two.
    Finite-sample estimates may be negative and are not clipped.

    ``sample_log_ratio`` may supply ``log(p_hat / h_hat)`` already evaluated
    at these same samples with the same reference points and settings, including
    ``self_weight``. This
    reuses the endpoint density calculation performed for response weights.
    """
    if sample_log_ratio is None:
        reverse_log_ratio = gaussian_log_density_ratio(
            reference, samples, n_neighbor, bandwidth, self_weight=self_weight,
            all_points_evaluator=all_points_evaluator)
    else:
        reference = _points(reference, "reference")
        samples = _points(samples, "samples")
        _self_kernel_parameters(self_weight, len(samples))
        if (isinstance(n_neighbor, (bool, np.bool_))
                or not isinstance(n_neighbor, (int, np.integer))
                or not 0 <= n_neighbor <= min(len(reference) - 1, len(samples) - 1)):
            raise ValueError(
                "n_neighbor must fit both populations after self-exclusion")
        reverse_log_ratio = np.asarray(sample_log_ratio)
        if (reverse_log_ratio.shape != (len(samples),)
                or reverse_log_ratio.dtype.kind not in "fiu"):
            raise ValueError("sample_log_ratio must be a real vector with one value per sample")
        reverse_log_ratio = np.asarray(reverse_log_ratio, dtype=np.float64)
        if not np.isfinite(reverse_log_ratio).all():
            raise FloatingPointError("Nonfinite sample_log_ratio")
    forward_log_ratio = gaussian_log_density_ratio(
        samples, reference, n_neighbor, bandwidth, self_weight=self_weight,
        all_points_evaluator=all_points_evaluator)
    # Divide before summing so large finite logarithms do not overflow merely
    # while computing their arithmetic mean.
    forward = -float(np.sum(forward_log_ratio / len(forward_log_ratio)))
    reverse = -float(np.sum(reverse_log_ratio / len(reverse_log_ratio)))
    averaged = 0.5 * forward + 0.5 * reverse
    if not np.isfinite([forward, reverse, averaged]).all():
        raise FloatingPointError("Nonfinite averaged-KL density estimate")
    return {"forward_kl": forward, "reverse_kl": reverse, "averaged_kl": averaged}


def averaged_kl_weights(train, samples, n_neighbor, bandwidth, *, self_weight=0.5,
                        weight_mode="kde", assignment_weights=None, all_points_evaluator=None,
                        loss_mode="averaged_kl"):
    """Return terminal response weights and log density ratios at endpoints.

    In ``kde`` mode the weight is ``(p_hat / h_hat + log(p_hat / h_hat)) / (2 * N)``
    for each of the N generated samples. In ``three_term`` mode it is
    ``(p_hat / h_hat + N * assignment_weights + log(p_hat / h_hat)) / (3 * N)``.
    In ``count_log`` mode it is
    ``(N * assignment_weights + log(p_hat / h_hat)) / (2 * N)``; this mode never
    exponentiates the log density ratio.
    ``count_only`` retains just ``assignment_weights / 2``. In ``count_squared``
    the coefficient is ``(N * assignment_weights)**2 / 2``; dividing it by N
    gives returned weights ``N * assignment_weights**2 / 2``. Log density ratios
    are still returned for KL reporting and clipping, but do not enter these weights.
    Each assignment weight is the fraction of data assigned to that endpoint,
    with exactly one nearest endpoint per datum. The engine applies the minus
    sign when contracting these weights with the path responses. Weights can be
    negative and are not clipped, centered, or normalized to sum to one here.
    The engine may subsequently subtract a frozen previous-update baseline.
    See ``gaussian_log_density_ratio`` for the density-estimation conventions.
    Directional KDE losses use ``(p_hat / h_hat) / N`` for ``forward_kl_kde``
    and ``log(p_hat / h_hat) / N`` for ``backward_kl``. Both use the same
    density estimator and the engine's leading minus sign. A constant in the
    backward-KL response is omitted by the zero-mean response identity.
    """
    if loss_mode not in ("averaged_kl", "forward_kl_kde", "backward_kl"):
        raise ValueError("KDE loss_mode must be averaged_kl, forward_kl_kde, or backward_kl")
    if loss_mode != "averaged_kl" and weight_mode != "kde":
        raise ValueError("Directional KDE losses require weight_mode=kde")
    if weight_mode not in ("count_squared", "count_log", "count_only", "kde", "three_term"):
        raise ValueError("weight_mode must be count_squared, count_log, count_only, three_term, or kde")
    log_ratio = gaussian_log_density_ratio(
        train, samples, n_neighbor, bandwidth, self_weight=self_weight,
        all_points_evaluator=all_points_evaluator)
    n_sample = len(log_ratio)
    if loss_mode == "backward_kl":
        return log_ratio / n_sample, log_ratio
    if loss_mode == "forward_kl_kde":
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            response_weights = np.exp(log_ratio - np.log(n_sample))
        if not np.isfinite(response_weights).all():
            raise FloatingPointError("Forward-KL response weights exceed float64 range; "
                                     "inspect endpoint coverage or increase density bandwidth")
        return response_weights, log_ratio
    if weight_mode in ("count_squared", "count_log", "count_only", "three_term"):
        assignment = np.asarray(assignment_weights)
        if (assignment.shape != (n_sample,) or assignment.dtype.kind not in "fiu"
                or not np.isfinite(assignment).all() or np.any(assignment < 0)
                or not np.isclose(np.sum(assignment), 1.0, rtol=1e-7, atol=1e-12)):
            raise ValueError("assignment_weights must be nonnegative endpoint masses summing to one")
        assignment = assignment.astype(np.float64)
    if weight_mode == "count_only":
        return assignment / 2.0, log_ratio
    if weight_mode == "count_squared":
        return (n_sample / 2.0) * np.square(assignment), log_ratio

    # Scale inside the exponential so a representable final response weight
    # does not overflow merely because the unscaled density ratio is larger.
    scale = (3.0 if weight_mode == "three_term" else 2.0) * n_sample
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        response_weights = log_ratio / scale
        if weight_mode == "count_log":
            response_weights = response_weights + assignment / 2.0
        else:
            response_weights = np.exp(log_ratio - np.log(scale)) + response_weights
            if weight_mode == "three_term":
                response_weights += assignment / 3.0
    if not np.isfinite(response_weights).all():
        raise FloatingPointError(
            "Averaged-KL response weights exceed float64 range "
            f"(maximum log density ratio {log_ratio.max():.6g}); "
            "inspect endpoint coverage or increase density bandwidth")
    return response_weights, log_ratio

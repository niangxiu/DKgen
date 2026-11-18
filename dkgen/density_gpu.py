"""Bounded-memory, float64 device kernels for all-point Gaussian densities.

The public density-ratio module validates inputs and chooses this backend only
for an explicitly selected device. The SDE and neural networks retain their own
precision settings.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp


_MAX_QUERY_BLOCK = 1024
_MAX_REFERENCE_BLOCK = 1024
# Bound each coordinate-difference block to 64 MiB in float64, independently
# of the number of data points or generated samples.
_MAX_DIFFERENCE_ENTRIES = 8_388_608


def _block_shape(dim):
    query = _MAX_QUERY_BLOCK
    reference = _MAX_REFERENCE_BLOCK
    while query * reference * dim > _MAX_DIFFERENCE_ENTRIES:
        if query >= reference and query > 1:
            query = max(1, query // 2)
        elif reference > 1:
            reference = max(1, reference // 2)
        else:
            break
    return query, reference


@jax.jit
def _log_ratio_kernel(train, samples, bandwidth, log_self_weight,
                      sample_normalization):
    """Evaluate both densities in one dispatch, merging reference blocks."""
    n_data, n_sample = train.shape[0], samples.shape[0]
    query_limit, reference_limit = _block_shape(samples.shape[1])
    query_size = min(query_limit, n_sample)
    data_size = min(reference_limit, n_data)
    sample_size = min(reference_limit, n_sample)
    query_blocks = (n_sample + query_size - 1) // query_size

    def padded(points, block_size):
        padding = (-points.shape[0]) % block_size
        return jnp.pad(points, ((0, padding), (0, 0)))

    query_points = padded(samples, query_size)
    data_points = padded(train, data_size)
    sample_points = padded(samples, sample_size)

    def population_sum(query, query_start, reference, population_size,
                       reference_size, replace_self):
        reference_blocks = reference.shape[0] // reference_size

        def block_sum(block):
            reference_start = block * reference_size
            points = jax.lax.dynamic_slice_in_dim(
                reference, reference_start, reference_size, axis=0)
            # Direct differences preserve small distances between nearby large
            # coordinates; a norm/dot-product identity would lose precision.
            squared_distance = jnp.sum(
                jnp.square(query[:, None, :] - points[None, :, :]), axis=-1)
            # Dividing twice also avoids forming an underflowed bandwidth^2.
            log_kernel = -0.5 * (squared_distance / bandwidth / bandwidth)
            reference_indices = reference_start + jnp.arange(reference_size)
            if replace_self:
                query_indices = query_start + jnp.arange(query_size)
                log_kernel = jnp.where(
                    query_indices[:, None] == reference_indices[None, :],
                    log_self_weight, log_kernel)
            log_kernel = jnp.where(
                reference_indices[None, :] < population_size,
                log_kernel, -jnp.inf)
            return logsumexp(log_kernel, axis=1)

        if reference_blocks == 1:
            return block_sum(0)

        def accumulate(block, total):
            return jnp.logaddexp(total, block_sum(block))

        return jax.lax.fori_loop(
            0, reference_blocks, accumulate,
            jnp.full((query_size,), -jnp.inf, dtype=jnp.float64))

    count_ratio = jnp.log(sample_normalization) - jnp.log(float(n_data))

    def query_ratio(block):
        query_start = block * query_size
        query = jax.lax.dynamic_slice_in_dim(
            query_points, query_start, query_size, axis=0)
        data_sum = population_sum(
            query, query_start, data_points, n_data, data_size, False)
        sample_sum = population_sum(
            query, query_start, sample_points, n_sample, sample_size, True)
        return count_ratio + data_sum - sample_sum

    if query_blocks == 1:
        return query_ratio(0)[:n_sample]
    return jax.lax.map(query_ratio, jnp.arange(query_blocks)).reshape(-1)[:n_sample]


def all_points_log_ratio(train, samples, bandwidth, log_self_weight,
                         sample_normalization, *, device=None):
    """Return validated all-point log(p/h) values as a host float64 vector.

    ``log_self_weight`` replaces only each query's indexed self contribution;
    distinct endpoints at the same coordinates keep their full kernel peaks.
    The caller supplies L-1 normalization for zero self weight, otherwise L.
    Bandwidth and self weight are dynamic kernel arguments, so a schedule does
    not compile a new executable at every update. Explicit CPU devices are
    supported for numerical regression tests as well as GPU devices.
    """
    with jax.enable_x64(True):
        arguments = tuple(jax.device_put(np.asarray(value, dtype=np.float64), device)
                          for value in (train, samples, bandwidth, log_self_weight,
                                        sample_normalization))
        log_ratio = np.asarray(_log_ratio_kernel(*arguments), dtype=np.float64)
    if not np.isfinite(log_ratio).all():
        raise FloatingPointError(
            "Nonfinite Gaussian log density ratio; inspect endpoint coordinates "
            "and density bandwidth")
    return log_ratio

"""Dense time-conditioned VP score baseline and streaming reverse-SDE sampler.

The network predicts the Gaussian perturbation rather than the score itself.
Noise-prediction squared error is denoising score matching weighted by the
perturbation variance. It is not a terminal-density KL training objective.
"""

import jax
import jax.numpy as jnp
import numpy as np


def parameter_count_formula(dim, width, depth, time_features):
    """Count every weight and bias of the single time-conditioned network."""
    return ((dim + time_features + 1) * width
            + (depth - 1) * (width + 1) * width
            + (width + 1) * dim)


def init_params(key, cfg):
    """Glorot hidden weights, zero biases, and a small random output layer."""
    cfg.validate()
    if cfg.dtype == "float64" and not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 before initializing float64 parameters")
    dtype = jnp.dtype(cfg.dtype)
    keys = jax.random.split(key, cfg.depth + 1)
    layers = []
    inputs = cfg.dim + cfg.time_features
    for index in range(cfg.depth):
        bound = jnp.sqrt(jnp.asarray(6.0 / (inputs + cfg.width), dtype))
        weights = jax.random.uniform(keys[index], (cfg.width, inputs), dtype=dtype,
                                     minval=-bound, maxval=bound)
        layers.append({"w": weights, "b": jnp.zeros(cfg.width, dtype)})
        inputs = cfg.width
    output = (0.01 / jnp.sqrt(jnp.asarray(cfg.width, dtype))
              * jax.random.normal(keys[-1], (cfg.dim, cfg.width), dtype))
    layers.append({"w": output, "b": jnp.zeros(cfg.dim, dtype)})
    return tuple(layers)


def time_embedding(t, cfg):
    """Fixed sine/cosine features with log-spaced angular frequencies."""
    dtype = jnp.dtype(cfg.dtype)
    frequencies = jnp.exp(jnp.linspace(0.0, jnp.log(cfg.time_max_frequency),
                                       cfg.time_features // 2, dtype=dtype))
    angles = jnp.asarray(t, dtype)[..., None] * frequencies
    return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)


def noise_prediction(params, x, t, cfg):
    """Predict the perturbation noise at one location and one time."""
    hidden = jnp.concatenate((x, time_embedding(t, cfg)))
    for layer in params[:-1]:
        hidden = jax.nn.silu(layer["w"] @ hidden + layer["b"])
    return params[-1]["w"] @ hidden + params[-1]["b"]


def beta(t, cfg):
    """Linear VP noise-rate schedule on the normalized time interval [0, 1]."""
    return cfg.beta_min + (cfg.beta_max - cfg.beta_min) * t


def marginal_coefficients(t, cfg):
    """Return a(t), s(t) for x_t = a(t) x_0 + s(t) epsilon."""
    t = jnp.asarray(t, dtype=jnp.dtype(cfg.dtype))
    integrated_beta = cfg.beta_min * t + 0.5 * (cfg.beta_max - cfg.beta_min) * t**2
    mean_scale = jnp.exp(-0.5 * integrated_beta)
    variance = -jnp.expm1(-integrated_beta)
    return mean_scale, jnp.sqrt(variance)


def sample_training_times(key, cfg, batch_size):
    """Sample times with optional extra emphasis near the low-noise endpoint.

    Power 1 retains the original uniform draw and random stream exactly. Larger
    powers use t = epsilon + (1 - epsilon) U**power with U uniform on [0, 1].
    """
    dtype = jnp.dtype(cfg.dtype)
    power = getattr(cfg, "time_sampling_power", 1.0)
    if power == 1:
        return jax.random.uniform(key, (batch_size,), dtype=dtype,
                                  minval=cfg.time_epsilon, maxval=1.0)
    uniform = jax.random.uniform(key, (batch_size,), dtype=dtype)
    return cfg.time_epsilon + (1.0 - cfg.time_epsilon) * uniform**power


def denoising_loss(params, key, train, cfg, batch_size):
    """Mean, over examples, of the sum of coordinate noise-prediction errors.

    Draw data with replacement and times according to time_sampling_power.
    Each example receives an independent standard Gaussian perturbation.
    In score units this is s(t)^2 times denoising score matching. There is no
    importance correction: nonuniform time sampling deliberately changes the
    relative training weight of different noise levels.
    """
    index_key, time_key, noise_key = jax.random.split(key, 3)
    dtype = jnp.dtype(cfg.dtype)
    indices = jax.random.randint(index_key, (batch_size,), 0, train.shape[0])
    clean = jnp.asarray(train, dtype=dtype)[indices]
    times = sample_training_times(time_key, cfg, batch_size)
    noise = jax.random.normal(noise_key, (batch_size, cfg.dim), dtype)
    mean_scale, std = marginal_coefficients(times, cfg)
    noisy = mean_scale[:, None] * clean + std[:, None] * noise
    predicted = jax.vmap(lambda x, t: noise_prediction(params, x, t, cfg))(noisy, times)
    return jnp.mean(jnp.sum((predicted - noise)**2, axis=-1))


def score(params, x, t, cfg):
    """Convert the predicted perturbation to the score at positive time t."""
    _, std = marginal_coefficients(t, cfg)
    return -noise_prediction(params, x, t, cfg) / std


def reverse_drift(params, x, t, cfg):
    """Drift in the increasing reverse clock, while physical t decreases."""
    noise_rate = beta(t, cfg)
    return 0.5 * noise_rate * x + noise_rate * score(params, x, t, cfg)


class ScoreEngine:
    """Jitted denoising gradients and a reverse sampler storing only current states."""

    def __init__(self, cfg):
        cfg.validate()
        self.cfg = cfg
        self.loss_and_gradient = jax.jit(jax.value_and_grad(
            lambda params, key, train: denoising_loss(
                params, key, train, cfg, cfg.batch_size)))
        self._sample_batch = jax.jit(self._make_sample_batch())

    def _make_sample_batch(self):
        cfg = self.cfg
        dtype = jnp.dtype(cfg.dtype)
        step = (1.0 - cfg.time_epsilon) / cfg.sampling_steps

        def sample_batch(params, key, indices):
            # Each path owns its stream, independently of its batch membership.
            path_keys = jax.vmap(lambda index: jax.random.fold_in(key, index))(indices)
            initial_keys = jax.vmap(lambda path_key: jax.random.fold_in(path_key, 0))(path_keys)
            initial = jax.vmap(lambda path_key: jax.random.normal(
                path_key, (cfg.dim,), dtype))(initial_keys)

            def advance(index, current):
                time = jnp.asarray(1.0, dtype) - jnp.asarray(index, dtype) * step
                drift = jax.vmap(lambda x: reverse_drift(params, x, time, cfg))(current)
                noise_keys = jax.vmap(lambda path_key: jax.random.fold_in(
                    path_key, index + 1))(path_keys)
                noise = jax.vmap(lambda noise_key: jax.random.normal(
                    noise_key, (cfg.dim,), dtype))(noise_keys)
                return current + step * drift + jnp.sqrt(beta(time, cfg) * step) * noise

            samples = jax.lax.fori_loop(0, cfg.sampling_steps, advance, initial)
            if cfg.denoise:
                # Posterior-mean estimate of clean x_0 at the final positive time.
                mean_scale, std = marginal_coefficients(cfg.time_epsilon, cfg)
                prediction = jax.vmap(lambda x: noise_prediction(
                    params, x, cfg.time_epsilon, cfg))(samples)
                samples = (samples - std * prediction) / mean_scale
            return samples

        return sample_batch

    def sample(self, params, key, n_samples):
        """Return normalized samples; preserve randomness across batch partitions."""
        if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        if n_samples >= 2**32:
            raise ValueError("n_samples must be smaller than 2**32 for distinct path keys")
        batch_size = min(self.cfg.sampling_batch_size, n_samples)
        chunks = []
        for start in range(0, n_samples, batch_size):
            # Padding keeps every batch the same compiled shape.
            count = min(batch_size, n_samples - start)
            indices = jnp.asarray(np.arange(start, start + batch_size, dtype=np.uint32))
            chunk = np.asarray(self._sample_batch(params, key, indices))
            chunks.append(chunk[:count])
        return np.concatenate(chunks, axis=0)

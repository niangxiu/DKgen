"""EDM denoising objective and Heun sampler for dense vector networks.

The preconditioning, lognormal noise distribution, and probability-flow ODE
follow Karras et al., "Elucidating the Design Space of Diffusion-Based
Generative Models" (2022), https://arxiv.org/abs/2206.00364. The small dense
backbone is an adaptation to vector data, not the paper's image U-Net.
"""

import jax
import jax.numpy as jnp
import numpy as np


def parameter_shapes(cfg):
    """Flat checkpoint layout for the selected backbone."""
    width = cfg.width
    if getattr(cfg, "backbone", "residual") == "mlp":
        shapes = {}
        for index in range(cfg.depth):
            fan_in = cfg.dim + cfg.time_features if index == 0 else width
            shapes[f"hidden_{index}_w"] = (width, fan_in)
            shapes[f"hidden_{index}_b"] = (width,)
        shapes.update(output_w=(cfg.dim, width), output_b=(cfg.dim,))
        return shapes
    shapes = {
        "input_w": (width, cfg.dim), "input_b": (width,),
        "time_w": (width, cfg.time_features), "time_b": (width,),
    }
    for index in range(cfg.depth):
        prefix = f"block_{index}_"
        for name in ("norm1", "norm2"):
            shapes[prefix + name + "_scale"] = (width,)
            shapes[prefix + name + "_bias"] = (width,)
        for name in ("dense1", "time", "dense2"):
            shapes[prefix + name + "_w"] = (width, width)
            shapes[prefix + name + "_b"] = (width,)
    shapes.update(output_w=(cfg.dim, width), output_b=(cfg.dim,))
    return shapes


def init_params(key, cfg):
    """Xavier weights, unit norm scales, and small residual/output projections.

The 1e-5 scale for the final affine maps follows the DDPM++ initialization
choice used with EDM. All biases start at zero. No data-dependent hidden-layer
calibration or trainable noise embedding frequencies are used.
    """
    cfg.validate()
    if cfg.dtype == "float64" and not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 before initializing float64 parameters")
    dtype = jnp.dtype(cfg.dtype)
    shapes = parameter_shapes(cfg)
    keys = jax.random.split(key, len(shapes))
    params = {}
    for (name, shape), subkey in zip(shapes.items(), keys):
        if name.endswith("_scale"):
            params[name] = jnp.ones(shape, dtype)
        elif name.endswith("_w"):
            fan_out, fan_in = shape
            bound = jnp.sqrt(jnp.asarray(6.0 / (fan_in + fan_out), dtype))
            small = name == "output_w" or name.endswith("_dense2_w")
            if small:
                bound = bound * jnp.asarray(1e-5, dtype)
            params[name] = jax.random.uniform(
                subkey, shape, dtype=dtype, minval=-bound, maxval=bound)
        else:
            params[name] = jnp.zeros(shape, dtype)
    return params


def noise_embedding(noise_label, cfg):
    """DDPM-style sine/cosine features of c_noise = log(sigma)/4.

Angular frequencies decrease geometrically from 1 to 1/10000, with both
endpoints included. This embedding is fixed and contributes no parameters.
    """
    dtype = jnp.dtype(cfg.dtype)
    half = cfg.time_features // 2
    powers = jnp.linspace(0.0, 1.0, half, dtype=dtype)
    frequencies = jnp.power(jnp.asarray(10000.0, dtype), -powers)
    angles = jnp.asarray(noise_label, dtype)[..., None] * frequencies
    return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)


def _affine(params, prefix, x):
    return params[prefix + "_w"] @ x + params[prefix + "_b"]


def _layer_norm(params, prefix, x):
    centered = x - jnp.mean(x)
    normalized = centered * jax.lax.rsqrt(
        jnp.mean(centered**2) + jnp.asarray(1e-6, x.dtype))
    return params[prefix + "_scale"] * normalized + params[prefix + "_bias"]


def residual_prediction(params, x, noise_label, cfg):
    """Evaluate the dense residual backbone on one preconditioned vector."""
    hidden = _affine(params, "input", x)
    embedding = jax.nn.silu(_affine(
        params, "time", noise_embedding(noise_label, cfg)))
    for index in range(cfg.depth):
        prefix = f"block_{index}_"
        residual = _affine(params, prefix + "dense1", jax.nn.silu(
            _layer_norm(params, prefix + "norm1", hidden)))
        residual = residual + _affine(params, prefix + "time", embedding)
        residual = _affine(params, prefix + "dense2", jax.nn.silu(
            _layer_norm(params, prefix + "norm2", residual)))
        hidden = (hidden + residual) / jnp.sqrt(jnp.asarray(2.0, hidden.dtype))
    return _affine(params, "output", jax.nn.silu(hidden))


def mlp_prediction(params, x, noise_label, cfg):
    """Concatenate state and fixed noise features, then apply hidden SiLU layers."""
    hidden = jnp.concatenate((x, noise_embedding(noise_label, cfg)))
    for index in range(cfg.depth):
        hidden = jax.nn.silu(_affine(params, f"hidden_{index}", hidden))
    return _affine(params, "output", hidden)


def preconditioning(sigma, cfg):
    """Return EDM's c_skip, c_out, c_in, and c_noise at positive sigma."""
    sigma = jnp.asarray(sigma, dtype=jnp.dtype(cfg.dtype))
    sigma_data = jnp.asarray(cfg.sigma_data, sigma.dtype)
    total_variance = sigma**2 + sigma_data**2
    c_in = jax.lax.rsqrt(total_variance)
    c_skip = sigma_data**2 / total_variance
    c_out = sigma * sigma_data * c_in
    return c_skip, c_out, c_in, jnp.log(sigma) / 4.0


def denoise(params, x, sigma, cfg):
    """Denoise one vector in the internal EDM units, at positive noise sigma."""
    c_skip, c_out, c_in, c_noise = preconditioning(sigma, cfg)
    prediction = (mlp_prediction if getattr(cfg, "backbone", "residual") == "mlp"
                  else residual_prediction)
    return c_skip * x + c_out * prediction(params, c_in * x, c_noise, cfg)


def training_loss(params, key, train, cfg, batch_size):
    """EDM weighted denoising MSE, averaged over examples and coordinates.

Training data remain in benchmark coordinates on disk. The scalar data_scale
converts selected points to the internal EDM units before adding Gaussian
noise. Log(sigma) has mean noise_log_mean and standard deviation noise_log_std;
the training distribution is not truncated to the sampler's noise interval.
    """
    index_key, sigma_key, perturbation_key = jax.random.split(key, 3)
    dtype = jnp.dtype(cfg.dtype)
    indices = jax.random.randint(index_key, (batch_size,), 0, train.shape[0])
    clean = jnp.asarray(train, dtype=dtype)[indices] * cfg.data_scale
    sigma = jnp.exp(cfg.noise_log_mean + cfg.noise_log_std * jax.random.normal(
        sigma_key, (batch_size,), dtype=dtype))
    perturbation = jax.random.normal(
        perturbation_key, (batch_size, cfg.dim), dtype=dtype)
    noisy = clean + sigma[:, None] * perturbation
    prediction = jax.vmap(lambda x, level: denoise(params, x, level, cfg))(noisy, sigma)
    # 1/c_out^2 is exactly the EDM weight (sigma^2 + sigma_data^2)
    # / (sigma * sigma_data)^2. Divide before squaring to avoid a large weight.
    _, c_out, _, _ = preconditioning(sigma, cfg)
    return jnp.mean(((prediction - clean) / c_out[:, None])**2)


def sampling_noise_levels(cfg):
    """EDM rho grid: sampling_steps positive levels and a final zero."""
    dtype = jnp.dtype(cfg.dtype)
    fraction = jnp.linspace(0.0, 1.0, cfg.sampling_steps, dtype=dtype)
    upper = jnp.asarray(cfg.sigma_max ** (1.0 / cfg.rho), dtype)
    lower = jnp.asarray(cfg.sigma_min ** (1.0 / cfg.rho), dtype)
    positive = (upper + fraction * (lower - upper))**cfg.rho
    # Preserve the requested endpoints despite rounding in the power formula.
    positive = positive.at[0].set(cfg.sigma_max).at[-1].set(cfg.sigma_min)
    return jnp.concatenate((positive, jnp.zeros(1, dtype)))


def edm_sample(denoiser, initial, cfg):
    """Integrate dx/dsigma = (x-D(x,sigma))/sigma with deterministic Heun.

``initial`` is an already scaled batch at sigma_max, in internal EDM units.
The callable accepts that entire batch and one scalar sigma. The final step
is Euler to zero, so D is never evaluated at zero. There are 2*N-1 denoiser
evaluations for N=sampling_steps. State and arithmetic use cfg.dtype.
    """
    levels = sampling_noise_levels(cfg)
    initial = jnp.asarray(initial, dtype=jnp.dtype(cfg.dtype))

    def advance(index, current):
        level, next_level = levels[index], levels[index + 1]
        step = next_level - level
        derivative = (current - denoiser(current, level)) / level
        predicted = current + step * derivative
        next_derivative = (predicted - denoiser(predicted, next_level)) / next_level
        return current + 0.5 * step * (derivative + next_derivative)

    current = jax.lax.fori_loop(0, cfg.sampling_steps - 1, advance, initial)
    # Euler from sigma_min to zero simplifies exactly to D(current,sigma_min).
    return denoiser(current, levels[-2])


class EDMEngine:
    """Jitted denoising gradients and deterministic, streaming EDM sampling."""

    def __init__(self, cfg):
        cfg.validate()
        self.cfg = cfg
        self.loss_and_gradient = jax.jit(jax.value_and_grad(
            lambda params, key, train: training_loss(
                params, key, train, cfg, cfg.batch_size)))
        self._sample_batch = jax.jit(self._make_sample_batch())

    def _make_sample_batch(self):
        cfg = self.cfg
        dtype = jnp.dtype(cfg.dtype)

        def sample_batch(params, key, indices):
            # The random stream belongs to each path, independently of batching.
            path_keys = jax.vmap(lambda index: jax.random.fold_in(key, index))(indices)
            initial = cfg.sigma_max * jax.vmap(lambda path_key: jax.random.normal(
                path_key, (cfg.dim,), dtype=dtype))(path_keys)
            denoiser = lambda points, level: jax.vmap(
                lambda point: denoise(params, point, level, cfg))(points)
            return edm_sample(denoiser, initial, cfg) / cfg.data_scale

        return sample_batch

    def sample(self, params, key, n_samples):
        """Return benchmark-coordinate samples, invariant to batch partitions."""
        if (isinstance(n_samples, bool)
                or not isinstance(n_samples, (int, np.integer)) or n_samples < 1):
            raise ValueError("n_samples must be a positive integer")
        if n_samples >= 2**32:
            raise ValueError("n_samples must be smaller than 2**32 for distinct path keys")
        batch_size = min(self.cfg.sampling_batch_size, n_samples)
        chunks = []
        for start in range(0, n_samples, batch_size):
            count = min(batch_size, n_samples - start)
            # Repeat the last valid key for padding; avoid uint32 wraparound.
            indices = np.minimum(np.arange(start, start + batch_size, dtype=np.uint64),
                                 n_samples - 1).astype(np.uint32)
            chunk = np.asarray(self._sample_batch(params, key, jnp.asarray(indices)))
            chunks.append(chunk[:count])
        return np.concatenate(chunks, axis=0)

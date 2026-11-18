"""Local derivatives and expanded DK steps, not the exact inverse-map scheme.

All coefficients include the cutoff. Exact traces batch the coordinate
derivative directions; Hessian-vector products use directional derivatives.
"""

import jax
import jax.numpy as jnp

from .model import drift, diffusion, fields


def exact_divergence(vector_field, x):
    """Compute the exact trace with batched forward-mode derivatives.

    Batching coordinate directions avoids a serial derivative loop in both
    the drift divergence and the diffusion Laplacian (2D and 8D).
    """
    return jnp.trace(jax.jacfwd(vector_field)(x))


def drift_divergence(params, x, cfg):
    return exact_divergence(lambda z: drift(params, z, cfg), x)


def sigma_gradient(params, x, cfg):
    return jax.grad(lambda z: diffusion(params, z, cfg))(x)


def sigma_laplacian(params, x, cfg):
    return exact_divergence(lambda z: sigma_gradient(params, z, cfg), x)


def state_step(params, x, db, cfg):
    f, sig = fields(params, x, cfg)
    return x + cfg.dt * f + sig * db


def transfer_rate(sigma_next, lambda_hat, cfg, alpha_sigma_multiplier=None):
    if cfg.alpha_mode == "constant":
        return jnp.full_like(sigma_next, cfg.alpha)
    if cfg.alpha_mode == "sigma":
        if alpha_sigma_multiplier is not None:
            return alpha_sigma_multiplier * sigma_next
        if cfg.alpha_sigma_multiplier is None:
            # Preserve the normalized rule recorded in older checkpoints.
            return cfg.alpha_cap * (sigma_next / cfg.sigma_cap)
        return cfg.alpha_sigma_multiplier * sigma_next
    return jnp.maximum(0.0, jnp.minimum(2.0 * sigma_next, 2.0 * lambda_hat))


def local_response_scalar(trial_params, x, nu, db, base_grad_sigma, base_lap_sigma, cfg):
    """ell_n(gamma') in the paper; its parameter gradient is the response.

    The caller supplies diffusion derivatives at the base parameter. They and
    the path are detached, even when trial_params and base_params share values.
    """
    x, nu, db, g, lap = jax.tree.map(
        jax.lax.stop_gradient, (x, nu, db, base_grad_sigma, base_lap_sigma))
    f_trial, sig_trial = fields(trial_params, x, cfg)
    return (
        -cfg.dt * (jnp.vdot(nu, f_trial) + drift_divergence(trial_params, x, cfg))
        + sig_trial * (cfg.dt * (jnp.vdot(g, nu) + lap) - jnp.vdot(nu, db))
        + jnp.vdot(sigma_gradient(trial_params, x, cfg), cfg.dt * g - db)
    )


def covector_step(params, x, nu, db, lambda_hat, cfg, g=None, lap=None,
                  alpha_sigma_multiplier=None):
    """Return x_next, nu_next, alpha, and the squared kernel-noise magnitude."""
    if g is None:
        g = sigma_gradient(params, x, cfg)
    if lap is None:
        lap = sigma_laplacian(params, x, cfg)
    sig = diffusion(params, x, cfg)
    x_next = state_step(params, x, db, cfg)
    alpha = transfer_rate(diffusion(params, x_next, cfg), lambda_hat, cfg,
                          alpha_sigma_multiplier=alpha_sigma_multiplier)
    _, pullback = jax.vjp(lambda z: drift(params, z, cfg), x)
    jacobian_transpose_nu = pullback(nu)[0]
    grad_div = jax.grad(lambda z: drift_divergence(params, z, cfg))(x)
    hessian_product = jax.jvp(lambda z: sigma_gradient(params, z, cfg),
                             (x,), (cfg.dt * g - db,))[1]
    kernel_noise = alpha * db / sig
    nu_next = (
        nu + cfg.dt * (g * jnp.vdot(g, nu) - jacobian_transpose_nu
                       - alpha * nu - grad_div + g * lap)
        - g * jnp.vdot(nu, db) + hessian_product - kernel_noise
    )
    return x_next, nu_next, alpha, jnp.vdot(kernel_noise, kernel_noise)


def homogeneous_step(params, x, direction, db, cfg):
    """Undamped homogeneous pilot. No additive covector terms are included."""
    g = sigma_gradient(params, x, cfg)
    _, pullback = jax.vjp(lambda z: drift(params, z, cfg), x)
    next_direction = (direction + cfg.dt * (g * jnp.vdot(g, direction)
                                           - pullback(direction)[0])
                      - g * jnp.vdot(direction, db))
    return state_step(params, x, db, cfg), next_direction

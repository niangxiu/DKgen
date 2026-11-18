"""Learning rates shared by the VP and EDM score optimizers."""

import math
from numbers import Real


def validate_learning_rate_schedule(cfg):
    """Validate the schedule without importing JAX or initializing a device."""
    mode = getattr(cfg, "learning_rate_schedule", "constant")
    if mode not in ("constant", "cosine"):
        raise ValueError("learning_rate_schedule must be constant or cosine")
    initial = cfg.learning_rate
    final = getattr(cfg, "learning_rate_final", 1e-4)
    for name, value in (("learning_rate", initial), ("learning_rate_final", final)):
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    start = getattr(cfg, "learning_rate_decay_start", 0)
    end = getattr(cfg, "learning_rate_decay_end", 20000)
    for name, value in (("learning_rate_decay_start", start),
                        ("learning_rate_decay_end", end)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if end <= start:
        raise ValueError("learning_rate_decay_end must exceed learning_rate_decay_start")
    if mode == "cosine" and final > initial:
        raise ValueError("learning_rate_final must not exceed learning_rate for cosine decay")


def learning_rate_at_step(cfg, step, *, dtype=None):
    """Return the rate for the next global Adam update, including its endpoints.

    A cosine schedule retains the initial rate through decay_start, reaches the
    final rate at decay_end, then holds it. The step remains a traced JAX value,
    so changing the update number does not compile a new optimizer.
    """
    validate_learning_rate_schedule(cfg)
    if getattr(cfg, "learning_rate_schedule", "constant") == "constant":
        # Retain the original scalar and arithmetic in the constant-rate case.
        return cfg.learning_rate
    import jax.numpy as jnp

    step = jnp.asarray(step, dtype=dtype)
    start = cfg.learning_rate_decay_start
    end = cfg.learning_rate_decay_end
    fraction = jnp.clip((step - start) / (end - start), 0.0, 1.0)
    rate = cfg.learning_rate_final + 0.5 * (
        cfg.learning_rate - cfg.learning_rate_final) * (1.0 + jnp.cos(jnp.pi * fraction))
    return jnp.where(step <= start, cfg.learning_rate,
                     jnp.where(step >= end, cfg.learning_rate_final, rate))

"""Batched JAX simulation and streaming forward parameter responses."""

from functools import partial
from dataclasses import fields as dataclass_fields, replace
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from .core import (covector_step, homogeneous_step, local_response_scalar,
                   sigma_gradient, sigma_laplacian, state_step)
from .density_ratio import averaged_kl_estimate, averaged_kl_weights
from .initial import initial_distribution, initial_parameter_scores
from .metrics import nearest_neighbor_weights
from .model import diffusion
from .optimizer import _global_norm


def path_keys(key, indices):
    """Stable path identity, independent of batching and the total path count."""
    return jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.asarray(indices, jnp.uint32))


def initial_states(keys, cfg, params=None):
    standard_normal = jax.vmap(lambda k: jax.random.normal(
        jax.random.fold_in(k, 0), (cfg.dim,), jnp.dtype(cfg.dtype)))(keys)
    if params is not None and "initial" in params:
        mean, s0 = initial_distribution(params, cfg, standard_normal.dtype)
        return mean + s0 * standard_normal
    # Preserve the operation order for old checkpoints and callers.
    return standard_normal / (jnp.sqrt(cfg.dim) / cfg.initial_rms_radius)


def initial_covectors(x, cfg, params=None):
    """Spatial score of the Gaussian initial law at each initial state."""
    if params is not None and "initial" in params:
        mean, s0 = initial_distribution(params, cfg, x.dtype)
        return -(x - mean) / s0**2
    return -(cfg.dim / cfg.initial_rms_radius**2) * x


def brownian_increments(keys, step, cfg):
    return jax.vmap(lambda k: jax.random.normal(jax.random.fold_in(k, step + 1), (cfg.dim,),
                                                jnp.dtype(cfg.dtype)))(keys) * jnp.sqrt(cfg.dt)


def _field_config(cfg, field_values):
    """Use numeric runtime bounds while keeping architecture and shapes static.

    This lightweight dataclass exists only while tracing. The two values stay
    ordinary JAX arguments throughout field evaluation and differentiation, so
    changing a scheduled bound does not compile a new simulation.
    """
    if field_values is None:
        return cfg
    return replace(cfg, drift_cap=field_values[0], sigma_min=field_values[1])


def make_sample_kernel(cfg):
    @jax.jit
    def simulate(params, keys, field_values=None):
        field_cfg = _field_config(cfg, field_values)
        def step(x, n):
            db = brownian_increments(keys, n, cfg)
            return jax.vmap(lambda a, b: state_step(params, a, b, field_cfg))(x, db), None
        return jax.lax.scan(step, initial_states(keys, cfg, params), jnp.arange(cfg.steps))[0]
    return simulate


def make_initial_sigma_sum_kernel(cfg):
    """Sum diffusion on an initial-state sample, excluding padding."""
    @jax.jit
    def initial_sigma_sum(params, keys, active_mask, field_values=None):
        field_cfg = _field_config(cfg, field_values)
        sigma = jax.vmap(lambda x: diffusion(params, x, field_cfg))(initial_states(keys, cfg, params))
        return jnp.sum(jnp.where(active_mask, sigma, 0))
    return initial_sigma_sum


def make_sample_with_paths_kernel(cfg):
    @partial(jax.jit, static_argnames=("n_traces",))
    def simulate(params, keys, n_traces, field_values=None):
        field_cfg = _field_config(cfg, field_values)
        x0 = initial_states(keys, cfg, params)

        def step(x, n):
            db = brownian_increments(keys, n, cfg)
            xn = jax.vmap(lambda a, b: state_step(params, a, b, field_cfg))(x, db)
            # Store only the requested paths, not the whole simulation batch.
            return xn, xn[:n_traces]

        endpoints, traced = jax.lax.scan(step, x0, jnp.arange(cfg.steps))
        trajectories = jnp.concatenate((x0[None, :n_traces], traced), axis=0)
        return endpoints, jnp.swapaxes(trajectories, 0, 1)
    return simulate


def make_pilot_kernel(cfg):
    @jax.jit
    def pilot(params, keys, field_values=None):
        field_cfg = _field_config(cfg, field_values)
        x = initial_states(keys, cfg, params)
        nu = initial_covectors(x, cfg, params)
        direction = nu / jnp.linalg.norm(nu, axis=1, keepdims=True)

        def step(carry, n):
            x, direction, log_growth = carry
            db = brownian_increments(keys, n, cfg)
            xn, propagated = jax.vmap(lambda a, u, b: homogeneous_step(params, a, u, b, field_cfg))(
                x, direction, db)
            norm = jnp.linalg.norm(propagated, axis=1)
            # Renormalization preserves the homogeneous norm ratio exactly in
            # real arithmetic while avoiding exponential overflow/underflow.
            return (xn, propagated / norm[:, None], log_growth + jnp.log(norm)), None

        result, _ = jax.lax.scan(step, (x, direction, jnp.zeros(x.shape[0], x.dtype)),
                                jnp.arange(cfg.steps))
        return result[2] / cfg.T
    return pilot


def make_forward_kernel(cfg):
    """Advance states, covectors, and one response accumulator per path.

    The time scan retains only its current carry. Optional trajectories are
    recorded for plots; the gradient never differentiates through the scan.
    """
    @partial(jax.jit, static_argnames=("n_traces",))
    def forward(params, keys, active_mask, lambda_hat, n_traces=0,
                alpha_sigma_multiplier=None, field_values=None):
        field_cfg = _field_config(cfg, field_values)

        def local(trial, x, nu, db, g, lap):
            return local_response_scalar(trial, x, nu, db, g, lap, field_cfg)

        response_increment = jax.vmap(
            jax.grad(local), in_axes=(None, 0, 0, 0, 0, 0))
        x0 = initial_states(keys, cfg, params)
        nu0 = initial_covectors(x0, cfg, params)
        responses = jax.tree.map(
            lambda a: jnp.zeros((keys.shape[0], *a.shape), a.dtype), params)
        if cfg.train_initial and "initial" in params:
            responses["initial"] = initial_parameter_scores(
                jax.lax.stop_gradient(x0), params, cfg)
        stats = jnp.zeros(4, x0.dtype)

        def step(carry, n):
            x, nu, responses, stats = carry
            x, nu = jax.tree.map(jax.lax.stop_gradient, (x, nu))
            db = brownian_increments(keys, n, cfg)
            g = jax.vmap(lambda a: sigma_gradient(params, a, field_cfg))(x)
            lap = jax.vmap(lambda a: sigma_laplacian(params, a, field_cfg))(x)
            increment = response_increment(params, x, nu, db, g, lap)
            responses = jax.tree.map(jnp.add, responses, increment)
            xn, vn, alpha, kernel_sq = jax.vmap(lambda a, v, b, q, tr: covector_step(
                params, a, v, b, lambda_hat, field_cfg, q, tr,
                alpha_sigma_multiplier=alpha_sigma_multiplier))(x, nu, db, g, lap)
            norm_sq = jnp.sum(vn * vn, axis=1)
            step_sums = jnp.array([jnp.sum(jnp.where(active_mask, alpha, 0)),
                                   jnp.sum(jnp.where(active_mask, kernel_sq, 0)),
                                   jnp.sum(jnp.where(active_mask, norm_sq, 0))])
            stats = stats.at[:3].add(step_sums)
            stats = stats.at[3].set(jnp.maximum(
                stats[3], jnp.max(jnp.where(active_mask, norm_sq, 0))))
            return (xn, vn, responses, stats), xn[:n_traces] if n_traces else None

        result, traced = jax.lax.scan(
            step, (x0, nu0, responses, stats), jnp.arange(cfg.steps))
        if n_traces:
            trajectories = jnp.concatenate((x0[None, :n_traces], traced), axis=0)
            return (*result, jnp.swapaxes(trajectories, 0, 1))
        return result
    return forward


@jax.jit
def _responses_finite(responses):
    """Return one device-side flag without downloading each path's response."""
    return jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(responses)]))


@jax.jit
def _weighted_response_sum(weights, responses):
    # Both KL response formulas use this leading minus sign. Terminal weights
    # are fixed after the forward simulation, before this parameter reduction.
    return jax.tree.map(lambda a: -jnp.tensordot(weights, a, axes=1), responses)


def make_replay_kernel(cfg, *, n_traces=0):
    """Legacy weighted replay for numerical comparisons, unused by training."""
    @jax.jit
    def replay(params, keys, weights, active_mask, lambda_hat,
               alpha_sigma_multiplier=None, field_values=None):
        field_cfg = _field_config(cfg, field_values)
        x = initial_states(keys, cfg, params)
        x0 = x
        nu = initial_covectors(x, cfg, params)
        total_grad = jax.tree.map(jnp.zeros_like, params)
        if cfg.train_initial and "initial" in params:
            scores = initial_parameter_scores(jax.lax.stop_gradient(x0), params, cfg)
            total_grad["initial"] = jax.tree.map(
                lambda a: -jnp.tensordot(jax.lax.stop_gradient(weights), a, axes=1), scores)
        # Sum alpha, squared kernel noise, and squared covectors over active
        # paths/steps; also retain the largest covector norm squared.
        stats = jnp.zeros(4, x.dtype)

        def step(carry, n):
            x, nu, gradient, stats = carry
            x, nu = jax.tree.map(jax.lax.stop_gradient, (x, nu))
            db = brownian_increments(keys, n, cfg)
            g = jax.vmap(lambda a: sigma_gradient(params, a, field_cfg))(x)
            lap = jax.vmap(lambda a: sigma_laplacian(params, a, field_cfg))(x)

            def weighted_local(trial):
                values = jax.vmap(lambda a, v, b, q, tr: local_response_scalar(
                    trial, a, v, b, q, tr, field_cfg))(x, nu, db, g, lap)
                # Minus sign: the objective is negative log density. The
                # frozen weights may include the zero-mean score baseline.
                return -jnp.vdot(jax.lax.stop_gradient(weights), values)

            increment = jax.grad(weighted_local)(params)
            gradient = jax.tree.map(jnp.add, gradient, increment)
            xn, vn, alpha, kernel_sq = jax.vmap(lambda a, v, b, q, tr: covector_step(
                params, a, v, b, lambda_hat, field_cfg, q, tr,
                alpha_sigma_multiplier=alpha_sigma_multiplier))(x, nu, db, g, lap)
            norm_sq = jnp.sum(vn * vn, axis=1)
            step_sums = jnp.array([jnp.sum(jnp.where(active_mask, alpha, 0)),
                                   jnp.sum(jnp.where(active_mask, kernel_sq, 0)),
                                   jnp.sum(jnp.where(active_mask, norm_sq, 0))])
            stats = stats.at[:3].add(step_sums)
            stats = stats.at[3].set(jnp.maximum(stats[3], jnp.max(jnp.where(active_mask, norm_sq, 0))))
            return (xn, vn, gradient, stats), xn[:n_traces] if n_traces else None

        result, traced = jax.lax.scan(
            step, (x, nu, total_grad, stats), jnp.arange(cfg.steps))
        if n_traces:
            trajectories = jnp.concatenate((x0[None, :n_traces], traced), axis=0)
            return (*result, jnp.swapaxes(trajectories, 0, 1))
        return result
    return replay


def make_covector_trajectory_kernel(cfg):
    """Replay the actual covector recurrence, recording selected batch slots."""
    @jax.jit
    def replay(params, keys, lambda_hat, trace_indices, alpha_sigma_multiplier=None,
               field_values=None):
        field_cfg = _field_config(cfg, field_values)
        x0 = initial_states(keys, cfg, params)
        nu0 = initial_covectors(x0, cfg, params)

        def step(carry, n):
            x, nu = jax.tree.map(jax.lax.stop_gradient, carry)
            db = brownian_increments(keys, n, cfg)
            g = jax.vmap(lambda a: sigma_gradient(params, a, field_cfg))(x)
            lap = jax.vmap(lambda a: sigma_laplacian(params, a, field_cfg))(x)
            xn, vn, alpha, _ = jax.vmap(lambda a, v, b, q, tr: covector_step(
                params, a, v, b, lambda_hat, field_cfg, q, tr,
                alpha_sigma_multiplier=alpha_sigma_multiplier))(x, nu, db, g, lap)
            return (xn, vn), (xn[trace_indices], vn[trace_indices], alpha[trace_indices])

        _, (positions, covectors, alpha) = jax.lax.scan(
            step, (x0, nu0), jnp.arange(cfg.steps))
        positions = jnp.concatenate((x0[None, trace_indices], positions), axis=0)
        covectors = jnp.concatenate((nu0[None, trace_indices], covectors), axis=0)
        return tuple(jnp.swapaxes(a, 0, 1) for a in (positions, covectors, alpha))
    return replay


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg.validate()
        if cfg.dtype == "float64" and not jax.config.x64_enabled:
            raise ValueError("Enable jax_enable_x64 before constructing a float64 engine")
        jax.config.update("jax_default_matmul_precision", "highest")
        self._density_kwargs = {}
        if jax.default_backend() == "gpu":
            from .density_gpu import all_points_log_ratio
            self._density_kwargs["all_points_evaluator"] = partial(
                all_points_log_ratio, device=jax.devices()[0])
        self._make_kernels(cfg)

    def _make_kernels(self, cfg):
        self.sample_kernel = make_sample_kernel(cfg)
        self.initial_sigma_sum_kernel = make_initial_sigma_sum_kernel(cfg)
        self.sample_with_paths_kernel = make_sample_with_paths_kernel(cfg)
        self.pilot_kernel = make_pilot_kernel(cfg)
        self.forward_kernel = make_forward_kernel(cfg)
        self.replay_kernel = make_replay_kernel(cfg)
        self.covector_trajectory_kernel = make_covector_trajectory_kernel(cfg)

    def set_config(self, cfg):
        """Update only scheduled coefficients without retracing scheduled kernels.

        A checkpoint's coefficients can also be restored for its evaluation.
        Other settings, including schedule mode, remain fixed for this engine.
        """
        cfg.validate()
        allowed = {"drift_cap", "sigma_min", "alpha_sigma_target", "density_bandwidth"}
        changed = {field.name for field in dataclass_fields(self.cfg)
                   if getattr(self.cfg, field.name) != getattr(cfg, field.name)}
        if changed - allowed:
            raise ValueError("Engine.set_config can change only scheduled coefficients; "
                             "unexpected changes: " + ", ".join(sorted(changed - allowed)))
        if (getattr(cfg, "schedule", "none") != "prescribed"
                and changed & {"drift_cap", "sigma_min"}):
            # Static runs preserve the original kernel signatures. The rare
            # explicit coefficient replacement must rebuild their closures.
            self._make_kernels(cfg)
        self.cfg = cfg

    def _field_kwargs(self):
        if getattr(self.cfg, "schedule", "none") != "prescribed":
            return {}
        return {"field_values": jnp.asarray(
            [self.cfg.drift_cap, self.cfg.sigma_min], dtype=self.cfg.dtype)}

    def _batches(self, key, count):
        for start in range(0, count, self.cfg.batch_size):
            size = min(self.cfg.batch_size, count - start)
            # Repeat an active path for padding. Extra independent paths could
            # fail numerically despite zero weight (0 * NaN is still NaN).
            indices = np.minimum(np.arange(start, start + self.cfg.batch_size), start + size - 1)
            keys = path_keys(key, indices.astype(np.uint32))
            mask = jnp.arange(self.cfg.batch_size) < size
            yield start, size, keys, mask

    def sample(self, params, key, n_paths):
        if not isinstance(n_paths, int) or n_paths < 1:
            raise ValueError("n_paths must be a positive integer")
        parts = []
        for _, size, keys, _ in self._batches(key, n_paths):
            part = self.sample_kernel(params, keys, **self._field_kwargs())
            part.block_until_ready()
            parts.append(part[:size])
        return jnp.concatenate(parts, axis=0)

    def sample_with_paths(self, params, key, n_paths, n_traces=10):
        """Sample endpoints and record the first paths in the same Euler pass.

        Trajectories have shape (min(n_traces, n_paths), steps + 1, dim),
        including the initial states and the corresponding returned endpoints.
        The remaining paths use the ordinary endpoint-only simulation kernel.
        """
        for name, value in (("n_paths", n_paths), ("n_traces", n_traces)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        parts, traces = [], []
        for start, size, keys, _ in self._batches(key, n_paths):
            count = min(size, n_traces - start)
            if count > 0:
                part, trajectory = self.sample_with_paths_kernel(
                    params, keys, count, **self._field_kwargs())
                part.block_until_ready()
                traces.append(trajectory)
            else:
                part = self.sample_kernel(params, keys, **self._field_kwargs())
                part.block_until_ready()
            parts.append(part[:size])
        return jnp.concatenate(parts, axis=0), jnp.concatenate(traces, axis=0)

    def pilot(self, params, key):
        values = []
        for _, size, keys, _ in self._batches(key, self.cfg.pilot_paths):
            values.append(np.asarray(self.pilot_kernel(
                params, keys, **self._field_kwargs()))[:size])
        values = np.concatenate(values).astype(np.float64)
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite homogeneous pilot growth; reduce the time step or inspect parameters")
        estimate = float(values.mean())
        stderr = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        return estimate, stderr

    def covector_trajectories(self, params, update_key, lambda_hat, n_paths=100, *,
                             alpha_sigma_multiplier=None):
        """Regenerate selected production paths for the covector growth plot.

        Pass the parameter values and update key used by ``gradient``, plus its
        pilot estimate (zero for constant alpha), and the runtime multiplier
        for normalized sigma alpha. The original production
        batch shape and padding are retained, without rerunning the pilot or
        evaluating a parameter gradient. Selection uses a separate folded key
        and does not consume the training random stream. Returned path IDs are
        sorted; arrays include the initial state/covector and all time steps.
        Nonfinite covectors are retained for diagnostic failure visualization.
        """
        if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 1:
            raise ValueError("n_paths must be a positive integer")
        cfg = self.cfg
        multiplier_kwargs = {}
        if cfg.alpha_mode == "sigma" and alpha_sigma_multiplier is not None:
            multiplier_kwargs["alpha_sigma_multiplier"] = jnp.asarray(
                alpha_sigma_multiplier, cfg.dtype)
        count = min(n_paths, cfg.paths)
        _, production_key = jax.random.split(update_key)
        selection_key = jax.random.fold_in(update_key, 0x4E554752)
        if count == cfg.paths:
            selected = np.arange(cfg.paths, dtype=np.int64)
        else:
            selected = np.sort(np.asarray(jax.random.choice(
                selection_key, cfg.paths, shape=(count,), replace=False), dtype=np.int64))
        # A fixed capture shape avoids compiling a new kernel for every
        # different number of selected paths within a production batch.
        capture_size = min(count, cfg.batch_size)
        positions, covectors, alphas = [], [], []
        for start, size, keys, _ in self._batches(production_key, cfg.paths):
            local = selected[(selected >= start) & (selected < start + size)] - start
            if not len(local):
                continue
            padded_local = np.full(capture_size, local[0], dtype=np.int32)
            padded_local[:len(local)] = local
            x, nu, alpha = self.covector_trajectory_kernel(
                params, keys, jnp.asarray(lambda_hat, cfg.dtype), jnp.asarray(padded_local),
                **multiplier_kwargs, **self._field_kwargs())
            x.block_until_ready()
            positions.append(np.asarray(x)[:len(local)].copy())
            covectors.append(np.asarray(nu)[:len(local)].copy())
            alphas.append(np.asarray(alpha)[:len(local)].copy())
        return {
            "path_indices": selected,
            "trajectories_normalized": np.concatenate(positions, axis=0),
            "covectors": np.concatenate(covectors, axis=0),
            "alpha": np.concatenate(alphas, axis=0),
        }

    def gradient(self, params, key, normalized_train, *, snapshot=None,
                 previous_sigma_mean=None, previous_response_mean=None):
        """Estimate the gradient with one forward pass over each production path.

        Each path carries its current state, covector, and accumulated local
        parameter response. The selected loss supplies frozen endpoint weights.
        A supplied dictionary receives the actual endpoints, first ten paths,
        and integer data-assignment counts after all gradient checks pass.
        Sigma normalization uses the preceding accepted update's mean diffusion;
        without one, it uses an independent sample of the initial states.
        KL update clipping evaluates its current loss on these same production
        endpoints and training data, without another simulation or random draw.
        In KDE losses, an optional baseline uses the preceding accepted
        update's mean coefficient, frozen independently of these current paths.
        The returned raw coefficient mean is for the next accepted update.
        """
        cfg = self.cfg
        train = np.asarray(normalized_train)
        if (train.ndim != 2 or len(train) == 0 or train.shape[1] != cfg.dim
                or not np.isfinite(train).all()):
            raise ValueError("Training data must be a finite (N, dim) array")
        if cfg.loss_mode != "forward_kl" and cfg.n_neighbor > len(train):
            raise ValueError("n_neighbor cannot exceed the actual training data count")
        if cfg.update_clip_mode == "kl" and cfg.density_self_weight == 0 and len(train) < 2:
            raise ValueError("KL update clipping with density_self_weight=0 needs at least two training points")
        if (cfg.update_clip_mode == "kl"
                and cfg.n_neighbor > min(len(train) - 1, cfg.paths - 1)):
            raise ValueError(
                "n_neighbor must fit actual training data and production paths "
                "after self-exclusion for KL update clipping")
        if previous_sigma_mean is not None:
            if (isinstance(previous_sigma_mean, (bool, np.bool_))
                    or np.ndim(previous_sigma_mean) != 0 or not np.isfinite(previous_sigma_mean)
                    or previous_sigma_mean <= 0):
                raise ValueError("previous_sigma_mean must be finite and positive")
            previous_sigma_mean = float(previous_sigma_mean)
        if previous_response_mean is not None:
            value = np.asarray(previous_response_mean)
            if (isinstance(previous_response_mean, (bool, np.bool_))
                    or value.shape != () or value.dtype.kind not in "fiu"
                    or not np.isfinite(value)):
                raise ValueError("previous_response_mean must be None or a finite scalar")
            previous_response_mean = float(value)
        pilot_key, production_key = jax.random.split(key)
        started = time.perf_counter()
        if cfg.alpha_mode == "adaptive":
            lambda_hat, pilot_stderr = self.pilot(params, pilot_key)
            pilot_seconds = time.perf_counter() - started
            reported_growth = lambda_hat
        else:
            # Keep the production key unchanged; constant and sigma modes need no pilot.
            lambda_hat = 0.0  # Unused numeric placeholder for the kernel signature.
            reported_growth, pilot_stderr, pilot_seconds = None, None, 0.0
        multiplier_kwargs = {}
        alpha_sigma_multiplier = None
        alpha_sigma_reference_mean = None
        alpha_sigma_reference_source = None
        production_batches = self._batches(production_key, cfg.paths)
        if cfg.alpha_mode == "sigma":
            if cfg.alpha_sigma_target is not None:
                if previous_sigma_mean is None:
                    # Use the unused independent key so the frozen multiplier
                    # does not depend on any current production path's past.
                    # This evaluates only initial points, with no SDE pass.
                    initial_sigma_sum = sum(float(self.initial_sigma_sum_kernel(
                        params, keys, mask, **self._field_kwargs()))
                        for _, _, keys, mask in self._batches(pilot_key, cfg.paths))
                    alpha_sigma_reference_mean = initial_sigma_sum / cfg.paths
                    alpha_sigma_reference_source = "initial_states"
                    if (not np.isfinite(alpha_sigma_reference_mean)
                            or alpha_sigma_reference_mean <= 0):
                        raise FloatingPointError(
                            "Initial mean diffusion must be finite and positive")
                else:
                    alpha_sigma_reference_mean = previous_sigma_mean
                    alpha_sigma_reference_source = "previous_update"
                multiplier = jnp.asarray(
                    cfg.alpha_sigma_target / alpha_sigma_reference_mean, cfg.dtype)
                alpha_sigma_multiplier = float(multiplier)
                if not np.isfinite(alpha_sigma_multiplier) or alpha_sigma_multiplier <= 0:
                    raise FloatingPointError(
                        "Normalized alpha sigma multiplier must be finite and positive")
                multiplier_kwargs["alpha_sigma_multiplier"] = multiplier
            else:
                alpha_sigma_multiplier = (cfg.alpha_cap / cfg.sigma_cap
                                          if cfg.alpha_sigma_multiplier is None
                                          else cfg.alpha_sigma_multiplier)
        parts, response_batches, traces = [], [], []
        totals = np.zeros(4, np.float64)
        for start, size, keys, mask in production_batches:
            n_traces = min(size, max(0, 10 - start)) if snapshot is not None else 0
            result = self.forward_kernel(
                params, keys, mask, jnp.asarray(lambda_hat, cfg.dtype), n_traces=n_traces,
                **multiplier_kwargs, **self._field_kwargs())
            xn, vn, responses, stats = result[:4]
            # Finish each batch before launching another. Keep only endpoints
            # and accumulated responses once that batch's time scan finishes.
            jax.block_until_ready(result)
            xn_host, vn_host = np.asarray(xn)[:size], np.asarray(vn)[:size]
            stats_host = np.asarray(stats)
            if not (np.isfinite(xn_host).all() and np.isfinite(vn_host).all()
                    and np.isfinite(stats_host).all()):
                raise FloatingPointError(
                    "Nonfinite forward simulation; no parameter update was applied")
            responses = jax.tree.map(lambda a: a[:size], responses)
            if not bool(_responses_finite(responses)):
                raise FloatingPointError(
                    "Nonfinite parameter response; no parameter update was applied")
            parts.append(xn_host.copy())
            response_batches.append(responses)
            if n_traces:
                traces.append(np.asarray(result[4]).copy())
            totals[:3] += stats_host[:3]
            totals[3] = max(totals[3], stats_host[3])

        endpoints_host = np.concatenate(parts, axis=0)
        # Retain configured k-neighbor counts for plots and matching diagnostics.
        # Responses using counts always assign each datum to one endpoint.
        weights = nearest_neighbor_weights(train, endpoints_host, cfg.neighbors)
        density_diagnostics = {}
        log_ratio = None
        response_coefficient_mean = None
        response_baseline = 0.0
        response_baseline_source = "not_used"
        if cfg.loss_mode != "forward_kl":
            assignment_weights = None
            if cfg.response_weight_mode in ("count_squared", "count_log", "count_only", "three_term"):
                assignment_weights = (weights if cfg.neighbors == 1 else
                                      nearest_neighbor_weights(train, endpoints_host, 1))
            response_weights, log_ratio = averaged_kl_weights(
                train, endpoints_host, cfg.n_neighbor, cfg.density_bandwidth,
                self_weight=cfg.density_self_weight, weight_mode=cfg.response_weight_mode,
                assignment_weights=assignment_weights, loss_mode=cfg.loss_mode,
                **self._density_kwargs)
            # These weights already include 1/L, so their sum is the ordinary
            # mean of the selected terminal coefficient, not a response norm mean.
            try:
                response_coefficient_mean = math.fsum(response_weights)
            except OverflowError as error:
                raise FloatingPointError("Nonfinite mean response coefficient") from error
            if not math.isfinite(response_coefficient_mean):
                raise FloatingPointError("Nonfinite mean response coefficient")
            if cfg.response_baseline == "previous_mean":
                response_baseline = (0.0 if previous_response_mean is None
                                     else previous_response_mean)
                response_baseline_source = ("initial_zero" if previous_response_mean is None
                                            else "previous_update")
            else:
                response_baseline_source = "disabled"
            density_diagnostics = {
                "density_log_ratio_min": float(np.min(log_ratio)),
                "density_log_ratio_max": float(np.max(log_ratio)),
                "density_log_ratio_mean": float(np.mean(log_ratio)),
            }
        else:
            response_weights = weights - cfg.score_baseline / cfg.paths
        if cfg.update_clip_mode == "kl":
            training_kl = averaged_kl_estimate(
                train, endpoints_host, cfg.n_neighbor, cfg.density_bandwidth,
                self_weight=cfg.density_self_weight,
                sample_log_ratio=log_ratio, **self._density_kwargs)
            density_diagnostics.update({
                f"training_{name}": value for name, value in training_kl.items()})
            loss_key = {"forward_kl_kde": "forward_kl", "backward_kl": "reverse_kl"}.get(
                cfg.loss_mode, cfg.loss_mode)
            density_diagnostics["training_kl"] = training_kl[loss_key]
        if cfg.loss_mode != "forward_kl" and response_baseline != 0.0:
            # Subtract only after recording the uncentered coefficient mean.
            # No derivative of the baseline or density estimates is taken.
            response_weights = response_weights - response_baseline / cfg.paths
        # Padding was removed before estimating densities or summing responses.
        # The weights are frozen; neither neighbors nor ratios are differentiated.
        with np.errstate(over="ignore"):
            response_weights_device = np.asarray(response_weights, dtype=cfg.dtype)
        if not np.isfinite(response_weights_device).all():
            raise FloatingPointError(
                "Terminal response weights exceed the configured dtype; "
                "inspect the density ratio or use float64")
        responses = jax.tree.map(lambda *a: jnp.concatenate(a, axis=0), *response_batches)
        gradient = _weighted_response_sum(jnp.asarray(response_weights_device), responses)
        leaves = jax.tree.leaves(gradient)
        gradient_arrays = [np.asarray(a, dtype=np.float64) for a in leaves]
        if not all(np.isfinite(a).all() for a in gradient_arrays):
            raise FloatingPointError("Nonfinite parameter gradient; no parameter update was applied")
        gradient_norm = _global_norm(gradient_arrays, name="parameter gradient")
        if snapshot is not None:
            # Counts describe nearest-data selections in either loss mode.
            # They are distinct from the Gaussian ratio weights in averaged KL.
            selection_counts = np.rint(weights * (len(train) * cfg.neighbors)).astype(np.int64)
            snapshot.update(
                generated_normalized=endpoints_host.copy(),
                trajectories_normalized=np.concatenate(traces, axis=0),
                selection_counts=selection_counts,
            )
        denom = cfg.paths * cfg.steps
        alpha_mean = float(totals[0] / denom)
        return gradient, {
            "lambda_hat": reported_growth, "pilot_standard_error": pilot_stderr,
            "pilot_seconds": pilot_seconds, "update_estimator_seconds": time.perf_counter() - started,
            "alpha_mean": alpha_mean,
            "alpha_sigma_multiplier": alpha_sigma_multiplier,
            "alpha_sigma_reference_mean": alpha_sigma_reference_mean,
            "alpha_sigma_reference_source": alpha_sigma_reference_source,
            "sigma_mean": (alpha_mean / alpha_sigma_multiplier
                           if alpha_sigma_multiplier is not None
                           and alpha_sigma_multiplier > 0 else None),
            "kernel_increment_rms": float(np.sqrt(totals[1] / denom)),
            "covector_rms": float(np.sqrt(totals[2] / denom)),
            "covector_max_norm": float(np.sqrt(totals[3])),
            "gradient_norm": gradient_norm,
            # These two matching diagnostics describe the unsigned data
            # assignments, not the signed weights applied to the responses.
            "effective_weighted_paths": float(1.0 / np.sum(weights**2)),
            "matched_paths": int(np.count_nonzero(weights)),
            "response_weight_sum": float(np.sum(response_weights)),
            "response_weight_l1": float(np.sum(np.abs(response_weights))),
            "response_weight_min": float(np.min(response_weights)),
            "response_weight_max": float(np.max(response_weights)),
            "gradient_passes": 1,
            "response_coefficient_mean": response_coefficient_mean,
            "response_weight_mode": (cfg.response_weight_mode
                                     if cfg.loss_mode == "averaged_kl" else cfg.loss_mode),
            "loss_mode": cfg.loss_mode,
            "response_baseline": response_baseline,
            "response_baseline_source": response_baseline_source,
            **density_diagnostics,
        }

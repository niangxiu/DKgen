"""Weight-regularized SGD and Adam with clipping of parameter increments."""

import math

import jax
import jax.numpy as jnp
import numpy as np


def _global_norm(arrays, name="parameter-update"):
    """Compute an L2 norm without squaring large, unscaled entries."""
    scale = max((float(np.max(np.abs(array))) for array in arrays if array.size),
                default=0.0)
    if not math.isfinite(scale):
        raise FloatingPointError(f"Nonfinite {name} norm")
    if scale == 0.0:
        return 0.0
    squares = sum(float(np.sum(np.square(array / scale), dtype=np.float64))
                  for array in arrays)
    norm = scale * math.sqrt(squares)
    if not math.isfinite(norm):
        raise FloatingPointError(f"Nonfinite {name} norm")
    return norm


def _clipping_settings(cfg, kl_budget):
    mode = getattr(cfg, "update_clip_mode", "norm")
    if mode == "norm":
        cap = float(cfg.max_update_norm)
        if not math.isfinite(cap) or cap < 0:
            raise ValueError("max_update_norm must be finite and nonnegative")
        return mode, cap, None
    if mode != "kl":
        raise ValueError("update_clip_mode must be kl or norm")
    budget = np.asarray(kl_budget)
    if budget.shape != () or budget.dtype.kind not in "iuf":
        raise ValueError("kl_budget must be a finite nonnegative scalar in KL clipping mode")
    budget = float(budget)
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("kl_budget must be a finite nonnegative scalar in KL clipping mode")
    return mode, None, budget


def _kl_clip_proposal(proposal, gradient, budget, raw_norm):
    """Bound each raw KL gradient's projection onto its proposed increment.

    ``proposal`` is subtracted from the parameters, so the signed predicted
    KL change is minus its inner product with the supplied DK gradient.
    Learning rates, regularization and Adam affect the proposal only.
    """
    clip_scales, branch_norms = {}, {}
    diagnostics = {"kl_clip_budget": float(budget)}
    for branch, increments in proposal.items():
        branch_norms[branch] = _global_norm(jax.tree_util.tree_leaves(increments))
        with np.errstate(over="ignore", invalid="ignore"):
            products = [float(np.sum(np.asarray(grad, dtype=np.float64) * increment,
                                     dtype=np.float64))
                        for grad, increment in zip(jax.tree_util.tree_leaves(gradient[branch]),
                                                   jax.tree_util.tree_leaves(increments))]
        if not all(math.isfinite(product) for product in products):
            raise FloatingPointError(f"Nonfinite {branch} projected KL change")
        try:
            change = -math.fsum(products)
        except OverflowError as error:
            raise FloatingPointError(f"Nonfinite {branch} projected KL change") from error
        if not math.isfinite(change):
            raise FloatingPointError(f"Nonfinite {branch} projected KL change")
        magnitude = abs(change)
        # A zero projection satisfies even a zero budget, including updates
        # caused entirely by regularization or by orthogonal Adam momentum.
        scale = budget / magnitude if magnitude > budget else 1.0
        clip_scales[branch] = scale
        norm = branch_norms[branch]
        diagnostics.update({
            f"{branch}_update_norm_unclipped": float(norm),
            f"{branch}_update_norm": float(norm * scale),
            f"{branch}_update_clip_scale": float(scale),
            f"{branch}_projected_kl_change_unclipped": float(change),
            f"{branch}_projected_kl_change": float(change * scale),
        })
    update_norm = (math.hypot(*(branch_norms[branch] * clip_scales[branch]
                               for branch in proposal))
                   if any(scale < 1.0 for scale in clip_scales.values()) else raw_norm)
    clip_scale = min(1.0, update_norm / raw_norm) if raw_norm else 1.0
    return clip_scales, update_norm, clip_scale, diagnostics


def _project_initial_proposal(params, proposal, cfg):
    """Make the proposed scale feasible before assessing its KL projection."""
    if "initial" not in params:
        return
    if np.shape(params["initial"]["mean"]) != (cfg.dim,):
        raise ValueError("Initial mean must have shape (dim,)")
    floor = float(cfg.initial_s0_min)
    scale = np.asarray(params["initial"]["s0"])
    if not math.isfinite(floor) or floor <= 0 or float(scale) < floor:
        raise ValueError("Initial s0 must be at least the positive initial_s0_min")
    # Use the smallest representable value at or above the requested floor.
    stored_floor = np.asarray(floor, dtype=scale.dtype)
    if float(stored_floor) < floor:
        stored_floor = np.nextafter(stored_floor, np.asarray(np.inf, dtype=scale.dtype))
    proposal["initial"]["s0"] = np.minimum(
        proposal["initial"]["s0"], float(scale) - float(stored_floor))


def _apply_initial_proposal(params, proposal, clip_scales, cfg):
    """Apply a feasible initial-law step in float64, then store its dtype."""
    result = {}
    for name, value in params["initial"].items():
        candidate = np.asarray(value, dtype=np.float64) - proposal["initial"][name] * clip_scales["initial"]
        if name == "s0":
            # Guard cancellation at the projected boundary before dtype rounding.
            stored_floor = np.asarray(cfg.initial_s0_min, dtype=np.asarray(value).dtype)
            if float(stored_floor) < cfg.initial_s0_min:
                stored_floor = np.nextafter(stored_floor, np.asarray(np.inf, dtype=stored_floor.dtype))
            candidate = np.maximum(candidate, stored_floor)
        result[name] = jnp.asarray(candidate, dtype=value.dtype)
    return result


def _initial_update_diagnostics(params, candidate, gradient, diagnostics):
    if "initial" not in params:
        return
    delta = {name: np.asarray(candidate["initial"][name], dtype=np.float64)
                  - np.asarray(value, dtype=np.float64)
             for name, value in params["initial"].items()}
    diagnostics["initial_mean_update"] = delta["mean"].tolist()
    diagnostics["initial_s0_update"] = float(delta["s0"])
    diagnostics["initial_s0"] = float(candidate["initial"]["s0"])
    diagnostics["initial_update_norm"] = _global_norm(list(delta.values()))
    if "initial_projected_kl_change" in diagnostics:
        diagnostics["initial_projected_kl_change"] = float(sum(
            np.sum(np.asarray(gradient["initial"][name], dtype=np.float64) * step)
            for name, step in delta.items()))


def sgd_update(params, gradient, cfg, *, kl_budget=None):
    """Return updated parameters and norms of the proposed SGD step.

    Add each branch's L2 penalty gradient to every layer's ``w`` matrix;
    biases are not penalized.  The supplied gradient remains the raw DK
    gradient.  Regularization precedes learning rates and update clipping.
    KL mode bounds each group's absolute projected KL change by the supplied
    ``kl_budget``, using the raw DK gradient. Norm mode clips each network to
    ``max_update_norm`` for neural diffusion and uses a combined cap otherwise;
    a zero norm cap disables clipping. Norms describe the intended step before
    rounding the updated parameters to their storage dtype. In neural mode,
    ``update_clip_scale`` is the combined norm ratio, not an applied factor;
    the branch-specific diagnostics record the actual factors.
    An optional initial mean/scale group uses the base rate without decay;
    its scale proposal is projected onto its floor before either clipping rule.
    """
    _validate_parameter_tree(params)
    if set(gradient) != set(params):
        raise ValueError("Parameters and gradient must contain drift and diffusion")
    if jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(gradient):
        raise ValueError("Parameters and gradient must have matching tree structures")
    clip_mode, cap, kl_budget = _clipping_settings(cfg, kl_budget)
    rates = {"drift": cfg.learning_rate * cfg.drift_learning_rate_multiplier,
             "diffusion": cfg.learning_rate}
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates.values()):
        raise ValueError("Branch learning rates must be finite and positive")
    decays = {"drift": float(cfg.drift_weight_decay),
              "diffusion": float(cfg.diffusion_weight_decay)}
    if any(not math.isfinite(decay) or decay < 0 for decay in decays.values()):
        raise ValueError("Weight-decay coefficients must be finite and nonnegative")
    if "initial" in params:
        rates["initial"] = cfg.learning_rate if cfg.train_initial else 0.0
        decays["initial"] = 0.0

    proposal = {}
    regularization_gradients = []
    regularized_gradients = []
    penalty_factors = []
    for branch, rate in rates.items():
        decay = decays[branch]

        def propose(path, value, grad):
            value_array, grad_array = np.asarray(value), np.asarray(grad)
            if (value_array.shape != grad_array.shape
                    or value_array.dtype != grad_array.dtype
                    or value_array.dtype.kind != "f"):
                raise ValueError("Parameter and gradient leaves must have matching floating shapes and dtypes")
            if not np.isfinite(value_array).all() or not np.isfinite(grad_array).all():
                raise FloatingPointError("Nonfinite parameter or gradient")
            # Host float64 avoids overflow in otherwise finite float32
            # regularized gradients and steps, before clipping and casting.
            with np.errstate(over="ignore", invalid="ignore"):
                combined = np.asarray(grad_array, dtype=np.float64)
                if decay > 0 and path and getattr(path[-1], "key", None) == "w":
                    weights = np.asarray(value_array, dtype=np.float64)
                    regularization = decay * weights
                    regularization_gradients.append(regularization)
                    penalty_factors.append(math.sqrt(decay) * weights)
                    combined = combined + regularization
                step = combined * rate
            if not np.isfinite(combined).all() or not np.isfinite(step).all():
                raise FloatingPointError("Nonfinite proposed parameter update")
            regularized_gradients.append(combined)
            return step
        proposal[branch] = jax.tree_util.tree_map_with_path(
            propose, params[branch], gradient[branch])

    _project_initial_proposal(params, proposal, cfg)
    regularization_norm = _global_norm(regularization_gradients, "regularization-gradient")
    regularized_norm = _global_norm(regularized_gradients, "regularized-gradient")
    penalty_norm = _global_norm(penalty_factors, "weight-regularization")
    penalty = (0.5 * penalty_norm) * penalty_norm
    if not math.isfinite(penalty):
        raise FloatingPointError("Nonfinite weight regularization penalty")
    raw_norm = _global_norm(jax.tree_util.tree_leaves(proposal))
    clip_scale = cap / raw_norm if clip_mode == "norm" and cap > 0 and raw_norm > cap else 1.0
    branch_diagnostics = {}
    if clip_mode == "kl":
        clip_scales, update_norm, clip_scale, branch_diagnostics = _kl_clip_proposal(
            proposal, gradient, kl_budget, raw_norm)
    elif cfg.diffusion_mode == "neural" or "initial" in params:
        branch_norms = {branch: _global_norm(jax.tree_util.tree_leaves(step))
                        for branch, step in proposal.items()}
        clip_scales = {branch: cap / norm if cap > 0 and norm > cap else 1.0
                       for branch, norm in branch_norms.items()}
        for branch, norm in branch_norms.items():
            branch_diagnostics.update({
                f"{branch}_update_norm_unclipped": float(norm),
                f"{branch}_update_norm": float(norm * clip_scales[branch]),
                f"{branch}_update_clip_scale": float(clip_scales[branch]),
            })
        # Preserve exactly equal before/after norms when neither branch clips.
        update_norm = (math.hypot(*(branch_norms[branch] * clip_scales[branch]
                                   for branch in rates))
                       if any(scale < 1.0 for scale in clip_scales.values()) else raw_norm)
        clip_scale = min(1.0, update_norm / raw_norm) if raw_norm else 1.0
    else:
        clip_scales = {branch: clip_scale for branch in rates}
        update_norm = raw_norm * clip_scale
    candidate = {}
    for branch, rate in rates.items():
        if branch == "initial":
            candidate[branch] = _apply_initial_proposal(params, proposal, clip_scales, cfg)
            continue
        if clip_mode == "kl" or clip_scales[branch] < 1.0 or bool(penalty_factors):
            # Scale in float64 before casting: a very small clip coefficient
            # can underflow if cast to float32 before multiplying a large step.
            candidate[branch] = jax.tree_util.tree_map(
                lambda value, step: value - jnp.asarray(step * clip_scales[branch], dtype=value.dtype),
                params[branch], proposal[branch])
        else:
            # Preserve the original operation order for an unclipped branch
            # with no penalized weights, regardless of the other branch.
            candidate[branch] = jax.tree_util.tree_map(
                lambda value, grad: value - rate * grad, params[branch], gradient[branch])
    for value in jax.tree_util.tree_leaves(candidate):
        if not np.isfinite(np.asarray(value)).all():
            raise FloatingPointError("Nonfinite updated parameters")
    _initial_update_diagnostics(params, candidate, gradient, branch_diagnostics)
    return candidate, {"update_norm_unclipped": float(raw_norm),
                       "update_norm": float(update_norm),
                       "update_clip_scale": float(clip_scale),
                       "weight_regularization": float(penalty),
                       "regularization_gradient_norm": float(regularization_norm),
                       "regularized_gradient_norm": float(regularized_norm),
                       **branch_diagnostics}


def _validate_parameter_tree(params):
    if (not isinstance(params, dict) or not {"drift", "diffusion"}.issubset(params)
            or set(params) - {"drift", "diffusion", "initial"}):
        raise ValueError("Parameters must contain drift and diffusion")
    for value in jax.tree_util.tree_leaves(params):
        array = np.asarray(value)
        if array.dtype.kind != "f":
            raise ValueError("Parameter leaves must have floating dtypes")
        if not np.isfinite(array).all():
            raise FloatingPointError("Nonfinite parameter")
    if "initial" in params:
        initial = params["initial"]
        if not isinstance(initial, dict) or set(initial) != {"mean", "s0"}:
            raise ValueError("Initial parameters must contain mean and s0")
        if np.ndim(initial["mean"]) != 1 or np.asarray(initial["mean"]).size < 1:
            raise ValueError("Initial mean must be a nonempty vector")
        if np.ndim(initial["s0"]) != 0 or float(initial["s0"]) <= 0:
            raise ValueError("Initial s0 must be a positive scalar")


def validate_optimizer_state(params, state):
    """Validate host Adam moments without changing their float64 precision.

    The caller separately checks that the step matches the checkpoint iteration.
    Moment arrays intentionally use float64 even when parameters use float32.
    """
    _validate_parameter_tree(params)
    if not isinstance(state, dict) or set(state) != {"step", "m", "v"}:
        raise ValueError("Adam state must contain step, m, and v")
    step = np.asarray(state["step"])
    if step.shape != () or step.dtype.kind not in "iu" or int(step) < 0:
        raise ValueError("Adam step must be a nonnegative integer")
    structure = jax.tree_util.tree_structure(params)
    for name in ("m", "v"):
        if jax.tree_util.tree_structure(state[name]) != structure:
            raise ValueError("Adam moment and parameter tree structures must match")
        for value, moment in zip(jax.tree_util.tree_leaves(params),
                                 jax.tree_util.tree_leaves(state[name])):
            array = np.asarray(moment)
            if array.shape != np.asarray(value).shape or array.dtype != np.dtype("float64"):
                raise ValueError("Adam moments must match parameter shapes and have float64 dtype")
            if not np.isfinite(array).all():
                raise FloatingPointError("Nonfinite Adam moment")
            if name == "v" and np.any(array < 0):
                raise ValueError("Adam second moments must be nonnegative")


def init_optimizer(params, cfg):
    """Return host float64 Adam state, or None for stateless SGD."""
    if cfg.optimizer == "sgd":
        return None
    if cfg.optimizer != "adam":
        raise ValueError("optimizer must be adam or sgd")
    _validate_parameter_tree(params)
    zeros = lambda value: np.zeros(np.asarray(value).shape, dtype=np.float64)
    return {"step": 0, "m": jax.tree_util.tree_map(zeros, params),
            "v": jax.tree_util.tree_map(zeros, params)}


@jax.jit
def _subtract_adam_proposals(network_params, typed_steps, initial_candidate):
    """Apply the already clipped increments and check the result on device."""
    candidate = jax.tree.map(lambda value, step: value - step,
                             network_params, typed_steps)
    if initial_candidate is not None:
        candidate["initial"] = initial_candidate
    checks = [jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(candidate)]
    finite = jnp.all(jnp.stack(checks)) if checks else jnp.asarray(True)
    return candidate, finite


def _apply_adam_candidate(params, proposal, clip_scales, cfg):
    """Batch final Adam subtractions without changing host-float64 clipping."""
    network_params = {branch: values for branch, values in params.items()
                      if branch != "initial"}
    # Scale in float64 and cast before subtracting, as in the unfused update.
    # A tiny clip scale can underflow if cast before multiplying a large step.
    with np.errstate(over="ignore", invalid="ignore"):
        typed_steps = {
            branch: jax.tree.map(
                lambda value, increment: np.asarray(
                    increment * clip_scales[branch], dtype=value.dtype),
                network_params[branch], proposal[branch])
            for branch in network_params}
    initial_candidate = (
        _apply_initial_proposal(params, proposal, clip_scales, cfg)
        if "initial" in params else None)
    candidate, finite = _subtract_adam_proposals(
        network_params, typed_steps, initial_candidate)
    if not bool(finite):
        raise FloatingPointError("Nonfinite updated parameters")
    return candidate


def adam_update(params, gradient, state, cfg, *, kl_budget=None):
    """Apply bias-corrected Adam, then clip the proposed parameter increment.

    L2 gradients are added to weights before updating Adam's moments. Biases
    are not penalized. The drift learning-rate multiplier and the independent
    per-network increment caps have the same meanings as in ``sgd_update``.
    In KL mode, the raw DK gradient is projected onto the actual Adam proposal,
    rather than approximating that projection by a squared gradient norm.
    Clipping changes the applied increment, not the stored moments. No raw
    gradient clipping is performed. Host float64 moments prevent squaring a
    large finite float32 DK gradient from overflowing float32. Input parameters
    and optimizer state are never modified in place; callers accept both return
    values together only after the update succeeds. Moments and clipping run on
    the host; final network subtractions and finite checking are JIT compiled.
    """
    validate_optimizer_state(params, state)
    if not isinstance(gradient, dict) or set(gradient) != set(params):
        raise ValueError("Parameters and gradient must contain drift and diffusion")
    if jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(gradient):
        raise ValueError("Parameters and gradient must have matching tree structures")
    if cfg.optimizer != "adam":
        raise ValueError("adam_update requires optimizer='adam'")
    for name in ("adam_beta1", "adam_beta2"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{name} must be finite and in [0, 1)")
    if not math.isfinite(cfg.adam_epsilon) or cfg.adam_epsilon <= 0:
        raise ValueError("adam_epsilon must be finite and positive")
    clip_mode, cap, kl_budget = _clipping_settings(cfg, kl_budget)
    rates = {"drift": cfg.learning_rate * cfg.drift_learning_rate_multiplier,
             "diffusion": cfg.learning_rate}
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates.values()):
        raise ValueError("Branch learning rates must be finite and positive")
    decays = {"drift": float(cfg.drift_weight_decay),
              "diffusion": float(cfg.diffusion_weight_decay)}
    if any(not math.isfinite(decay) or decay < 0 for decay in decays.values()):
        raise ValueError("Weight-decay coefficients must be finite and nonnegative")
    if "initial" in params:
        rates["initial"] = cfg.learning_rate if cfg.train_initial else 0.0
        decays["initial"] = 0.0
    step = int(state["step"]) + 1
    correction1 = 1 - cfg.adam_beta1 ** step
    correction2 = 1 - cfg.adam_beta2 ** step
    regularization_gradients, regularized_gradients, penalty_factors = [], [], []
    proposal, first_moments, second_moments = {}, {}, {}
    for branch, rate in rates.items():
        decay = decays[branch]

        if branch == "initial" and not cfg.train_initial:
            first_moments[branch] = jax.tree.map(lambda x: np.array(x, copy=True), state["m"][branch])
            second_moments[branch] = jax.tree.map(lambda x: np.array(x, copy=True), state["v"][branch])
            proposal[branch] = jax.tree.map(lambda x: np.zeros(np.shape(x), np.float64), params[branch])
            continue

        def regularize(path, value, grad):
            value_array, grad_array = np.asarray(value), np.asarray(grad)
            if (value_array.shape != grad_array.shape
                    or value_array.dtype != grad_array.dtype):
                raise ValueError("Parameter and gradient leaves must have matching floating shapes and dtypes")
            if not np.isfinite(grad_array).all():
                raise FloatingPointError("Nonfinite gradient")
            with np.errstate(over="ignore", invalid="ignore"):
                combined = np.asarray(grad_array, dtype=np.float64)
                if decay > 0 and path and getattr(path[-1], "key", None) == "w":
                    weights = np.asarray(value_array, dtype=np.float64)
                    regularization = decay * weights
                    regularization_gradients.append(regularization)
                    penalty_factors.append(math.sqrt(decay) * weights)
                    combined = combined + regularization
            if not np.isfinite(combined).all():
                raise FloatingPointError("Nonfinite regularized gradient")
            regularized_gradients.append(combined)
            return combined

        combined = jax.tree_util.tree_map_with_path(
            regularize, params[branch], gradient[branch])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            first_moments[branch] = jax.tree_util.tree_map(
                lambda old, grad: cfg.adam_beta1 * old + (1 - cfg.adam_beta1) * grad,
                state["m"][branch], combined)
            second_moments[branch] = jax.tree_util.tree_map(
                lambda old, grad: cfg.adam_beta2 * old + ((1 - cfg.adam_beta2) * grad) * grad,
                state["v"][branch], combined)
            proposal[branch] = jax.tree_util.tree_map(
                lambda first, second: rate * (first / correction1)
                / (np.sqrt(second / correction2) + cfg.adam_epsilon),
                first_moments[branch], second_moments[branch])
    new_state = {"step": step, "m": first_moments, "v": second_moments}
    validate_optimizer_state(params, new_state)
    _project_initial_proposal(params, proposal, cfg)
    regularization_norm = _global_norm(regularization_gradients, "regularization-gradient")
    regularized_norm = _global_norm(regularized_gradients, "regularized-gradient")
    penalty_norm = _global_norm(penalty_factors, "weight-regularization")
    penalty = (0.5 * penalty_norm) * penalty_norm
    if not math.isfinite(penalty):
        raise FloatingPointError("Nonfinite weight regularization penalty")
    raw_norm = _global_norm(jax.tree_util.tree_leaves(proposal))
    clip_scale = cap / raw_norm if clip_mode == "norm" and cap > 0 and raw_norm > cap else 1.0
    branch_diagnostics = {}
    if clip_mode == "kl":
        clip_scales, update_norm, clip_scale, branch_diagnostics = _kl_clip_proposal(
            proposal, gradient, kl_budget, raw_norm)
    elif cfg.diffusion_mode == "neural" or "initial" in params:
        branch_norms = {branch: _global_norm(jax.tree_util.tree_leaves(increment))
                        for branch, increment in proposal.items()}
        clip_scales = {branch: cap / norm if cap > 0 and norm > cap else 1.0
                       for branch, norm in branch_norms.items()}
        for branch, norm in branch_norms.items():
            branch_diagnostics.update({
                f"{branch}_update_norm_unclipped": float(norm),
                f"{branch}_update_norm": float(norm * clip_scales[branch]),
                f"{branch}_update_clip_scale": float(clip_scales[branch]),
            })
        update_norm = (math.hypot(*(branch_norms[branch] * clip_scales[branch]
                                   for branch in rates))
                       if any(scale < 1.0 for scale in clip_scales.values()) else raw_norm)
        clip_scale = min(1.0, update_norm / raw_norm) if raw_norm else 1.0
    else:
        clip_scales = {branch: clip_scale for branch in rates}
        update_norm = raw_norm * clip_scale
    candidate = _apply_adam_candidate(params, proposal, clip_scales, cfg)
    _initial_update_diagnostics(params, candidate, gradient, branch_diagnostics)
    return candidate, new_state, {
        "update_norm_unclipped": float(raw_norm),
        "update_norm": float(update_norm),
        "update_clip_scale": float(clip_scale),
        "weight_regularization": float(penalty),
        "regularization_gradient_norm": float(regularization_norm),
        "regularized_gradient_norm": float(regularized_norm),
        **branch_diagnostics}

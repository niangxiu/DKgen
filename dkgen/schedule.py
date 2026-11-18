"""Shared coefficient schedules and their saved-state consistency checks."""

from dataclasses import asdict, dataclass, replace
import math
from numbers import Integral, Real


@dataclass(frozen=True)
class ScheduleSettings:
    """Edit schedule defaults here for both 2D and higher-dimensional runs.

    Explicit CLI options override these defaults. Each run saves its resolved
    settings: resume continues that recipe; warm start may override it.
    """

    schedule: str = "prescribed"  # prescribed or none
    schedule_updates: int = 4000  # Reach the endpoints here, then hold.
    schedule_drift_cap_start: float = 4.0
    schedule_drift_cap_end: float = 4.0
    schedule_alpha_sigma_target_start: float = 42.0
    schedule_alpha_sigma_target_end: float = 42.0
    schedule_sigma_min_start: float = 0.1
    schedule_sigma_min_end: float = 0.1
    schedule_density_bandwidth_start: float = 0.3
    schedule_density_bandwidth_end: float = 0.03
    # Fixed validation/test bandwidth throughout the prescribed schedule.
    schedule_evaluation_bandwidth: float = 0.03


SCHEDULE_FIELDS = (
    "drift_cap", "alpha_sigma_target", "sigma_min", "density_bandwidth",
)


def validate_schedule_settings(cfg) -> None:
    scheduled_fields(cfg.schedule)
    duration = cfg.schedule_updates
    if isinstance(duration, bool) or not isinstance(duration, Integral) or duration < 2:
        raise ValueError("schedule_updates must be an integer of at least 2")
    names = ("schedule_evaluation_bandwidth",)
    names += tuple(f"schedule_{name}_{end}" for name in SCHEDULE_FIELDS
                   for end in ("start", "end"))
    for name in names:
        value = getattr(cfg, name)
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")


def linear_progress(iteration: int, duration: int) -> float:
    if isinstance(duration, bool) or not isinstance(duration, Integral) or duration < 2:
        raise ValueError("schedule_updates must be an integer of at least 2")
    if isinstance(iteration, bool) or not isinstance(iteration, Integral) or iteration < 0:
        raise ValueError("Schedule iteration must be a nonnegative integer")
    return min(1.0, max(0.0, (iteration - 1) / (duration - 1)))


def linear_value(start, end, iteration: int, duration: int):
    """Use the start at updates 0/1 and hold the exact endpoint from duration."""
    progress = linear_progress(iteration, duration)
    return start if progress == 0.0 else end if progress == 1.0 else start + progress * (end - start)


# Retain the private name used by older callers/tests.
_progress = linear_progress


def scheduled_fields(mode):
    if mode == "prescribed":
        return SCHEDULE_FIELDS
    if mode == "none":
        return ()
    raise ValueError("schedule must be prescribed or none")


def _values(config: dict, iteration: int) -> dict:
    settings = asdict(ScheduleSettings()) | config
    return {
        name: linear_value(settings[f"schedule_{name}_start"],
                           settings[f"schedule_{name}_end"], iteration,
                           settings["schedule_updates"])
        for name in SCHEDULE_FIELDS
    }


def scheduled_config(cfg, iteration: int):
    """Return effective fields for this update, holding the end values afterward.

    The first update and initial (step-zero) validation use the start values.
    Update ``schedule_updates`` reaches the endpoint. An ordinary resume uses
    the saved global iteration; a warm start begins a new schedule at zero.
    """
    if cfg.schedule == "none":
        return cfg
    if cfg.schedule != "prescribed":
        raise ValueError("schedule must be prescribed or none")
    return replace(cfg, **_values(vars(cfg), iteration))


def evaluation_bandwidth(cfg) -> float:
    """Keep scheduled validation and test KL on a fixed scale."""
    return cfg.schedule_evaluation_bandwidth if cfg.schedule == "prescribed" else cfg.density_bandwidth


def schedule_record(cfg, iteration: int) -> dict:
    """Record the effective model and training/evaluation density bandwidths."""
    effective = scheduled_config(cfg, iteration)
    return {
        "name": cfg.schedule,
        "updates": cfg.schedule_updates,
        "progress": (_progress(iteration, cfg.schedule_updates)
                     if cfg.schedule == "prescribed" else None),
        **({"start": {name: getattr(cfg, f"schedule_{name}_start") for name in SCHEDULE_FIELDS},
            "end": {name: getattr(cfg, f"schedule_{name}_end") for name in SCHEDULE_FIELDS}}
           if cfg.schedule == "prescribed" else {}),
        **{name: getattr(effective, name) for name in SCHEDULE_FIELDS},
        "evaluation_density_bandwidth": evaluation_bandwidth(effective),
    }


def validate_schedule_state(config: dict, iteration: int) -> None:
    """Reject checkpoints whose saved fields disagree with their schedule stage."""
    mode = config.get("schedule", "none")
    if mode == "none":
        return
    if mode != "prescribed":
        raise ValueError("Checkpoint schedule must be prescribed or none")
    values = _values(config, iteration)
    for name, expected in values.items():
        actual = config.get(name)
        if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                or not math.isfinite(actual)
                or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError(
                f"Checkpoint {name} does not match prescribed schedule at iteration {iteration}")


def compatible_scheduled_configs(first: dict, first_iteration: int,
                                 second: dict, second_iteration: int) -> bool:
    """Allow only stage-dependent fields to differ between scheduled states."""
    mode = first.get("schedule")
    if mode == second.get("schedule") and mode == "prescribed":
        validate_schedule_state(first, first_iteration)
        validate_schedule_state(second, second_iteration)
        variable_fields = scheduled_fields(mode)
        return ({name: value for name, value in first.items() if name not in variable_fields}
                == {name: value for name, value in second.items() if name not in variable_fields})
    return first == second

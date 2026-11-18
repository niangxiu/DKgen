"""Settings for the parameter-matched dense EDM benchmark adaptation."""

from dataclasses import asdict, dataclass, replace
import math
from numbers import Real

from .config import Config
from .schedule import evaluation_bandwidth
from .score_config import dk_parameter_budget
from .score_learning_rate import validate_learning_rate_schedule


def edm_parameter_count(dim, width, depth, time_features, backbone="residual"):
    """Count the trainable parameters of the selected dense backbone."""
    if backbone == "mlp":
        return ((dim + time_features + 1) * width
                + (depth - 1) * (width + 1) * width + (width + 1) * dim)
    if backbone != "residual":
        raise ValueError("backbone must be residual or mlp")
    return ((dim + 1) * width + (time_features + 1) * width
            + depth * (3 * width**2 + 7 * width) + (width + 1) * dim)


def closest_width(target, dim, depth, time_features, backbone="residual"):
    """Choose the nearest parameter count, breaking ties toward smaller widths."""
    def count(width):
        return edm_parameter_count(dim, width, depth, time_features, backbone)
    lower, upper = 1, 2
    while count(upper) < target:
        lower, upper = upper, 2 * upper
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if count(middle) < target:
            lower = middle
        else:
            upper = middle
    return min((lower, upper), key=lambda width: (abs(count(width) - target), width))


@dataclass(frozen=True)
class EDMConfig:
    dim: int = 8
    width: int = 15
    # Residual blocks for residual; hidden SiLU layers for mlp.
    depth: int = 2
    backbone: str = "residual"
    time_features: int = 16
    target_parameter_count: int = 2139
    sigma_data: float = 0.5
    noise_log_mean: float = -1.2
    noise_log_std: float = 1.2
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    sampling_steps: int = 32
    sampling_batch_size: int = 1024
    # Computed from training data only. EDM uses data_scale * normalized data;
    # generated outputs are divided by this scalar before evaluation or plotting.
    data_scale: float = 1.0
    batch_size: int = 1024
    learning_rate: float = 1e-3
    learning_rate_schedule: str = "constant"
    learning_rate_final: float = 1e-4
    # Global optimizer update numbers, also when resuming from a checkpoint.
    learning_rate_decay_start: int = 0
    learning_rate_decay_end: int = 20000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    # Zero disables gradient clipping.
    max_gradient_norm: float = 0.0
    ema_decay: float = 0.99
    updates: int = 1000
    eval_every: int = 50
    log_every: int = 10
    seed: int = 0
    data_seed: int = 12345
    mixture_modes: int = 4
    n_train: int = 1024
    n_val: int = 1024
    n_test: int = 1024
    eval_samples: int = 1024
    # Benchmark KL always sums all available data and samples.
    n_neighbor: int = 0
    density_bandwidth: float = 0.2
    density_self_weight: float = 0.5
    dtype: str = "float32"

    @property
    def parameter_count(self):
        return edm_parameter_count(
            self.dim, self.width, self.depth, self.time_features, self.backbone)

    def validate(self):
        validate_learning_rate_schedule(self)
        if self.backbone not in ("residual", "mlp"):
            raise ValueError("backbone must be residual or mlp")
        for name in ("dim", "width", "depth", "time_features", "target_parameter_count",
                     "sampling_steps", "sampling_batch_size", "batch_size", "eval_every",
                     "log_every", "mixture_modes", "n_train", "n_val", "n_test", "eval_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.time_features % 2:
            raise ValueError("time_features must be even for sine/cosine pairs")
        if self.sampling_steps < 2:
            raise ValueError("sampling_steps must be at least 2 for the EDM noise schedule")
        if (isinstance(self.n_neighbor, bool) or not isinstance(self.n_neighbor, int)
                or self.n_neighbor != 0):
            raise ValueError("n_neighbor must be 0: benchmark KL uses all points")
        if isinstance(self.updates, bool) or not isinstance(self.updates, int) or self.updates < 0:
            raise ValueError("updates must be a nonnegative integer")
        for name in ("seed", "data_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
                raise ValueError(f"{name} must be an integer in [0, 2**32)")
        for name in ("sigma_data", "noise_log_std", "sigma_min", "sigma_max", "rho",
                     "data_scale", "learning_rate", "adam_epsilon", "density_bandwidth"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.sigma_max <= self.sigma_min:
            raise ValueError("sigma_max must be greater than sigma_min")
        if (isinstance(self.noise_log_mean, bool) or not isinstance(self.noise_log_mean, Real)
                or not math.isfinite(self.noise_log_mean)):
            raise ValueError("noise_log_mean must be finite")
        for name in ("adam_beta1", "adam_beta2", "ema_decay"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or not 0 <= value < 1):
                raise ValueError(f"{name} must lie in [0, 1)")
        if (isinstance(self.max_gradient_norm, bool)
                or not isinstance(self.max_gradient_norm, Real)
                or not math.isfinite(self.max_gradient_norm) or self.max_gradient_norm < 0):
            raise ValueError("max_gradient_norm must be finite and nonnegative")
        if (isinstance(self.density_self_weight, bool)
                or not isinstance(self.density_self_weight, Real)
                or not math.isfinite(self.density_self_weight)
                or not 0 <= self.density_self_weight <= 1):
            raise ValueError("density_self_weight must be finite and in [0, 1]")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        return self


def matched_config(dk_cfg=None, **overrides):
    """Match DK data, evaluation, and parameter budget while retaining EDM settings."""
    dk_cfg = Config() if dk_cfg is None else dk_cfg
    dk_cfg.validate()
    values = asdict(EDMConfig())
    for name in ("dim", "updates", "eval_every", "seed", "data_seed", "mixture_modes",
                 "n_train", "n_val", "n_test", "eval_samples", "density_self_weight", "dtype"):
        values[name] = getattr(dk_cfg, name)
    values.update(target_parameter_count=dk_parameter_budget(dk_cfg), batch_size=dk_cfg.paths,
                  sampling_batch_size=dk_cfg.batch_size,
                  density_bandwidth=evaluation_bandwidth(dk_cfg))
    cfg = replace(EDMConfig(**values), **overrides).validate()
    if "width" not in overrides:
        cfg = replace(cfg, width=closest_width(
            cfg.target_parameter_count, cfg.dim, cfg.depth, cfg.time_features, cfg.backbone))
    return cfg.validate()

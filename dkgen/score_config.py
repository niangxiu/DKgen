"""Settings for a dense, time-conditioned VP score-diffusion baseline."""

from dataclasses import asdict, dataclass, replace
import math
from numbers import Real

from .config import Config
from .schedule import evaluation_bandwidth
from .score_learning_rate import validate_learning_rate_schedule


def dense_parameter_count(inputs, width, depth, outputs):
    return (inputs + 1) * width + (depth - 1) * (width + 1) * width + (width + 1) * outputs


def dk_parameter_budget(cfg):
    count = dense_parameter_count(cfg.dim, cfg.width, cfg.depth, cfg.dim)
    if cfg.diffusion_mode == "neural":
        count += dense_parameter_count(cfg.dim, cfg.diffusion_width, cfg.diffusion_depth, 1)
    if getattr(cfg, "train_initial", False):
        count += cfg.dim + 1
    return count


def closest_width(target, dim, depth, time_features):
    """Choose the nearest attainable total count, with smaller width breaking ties."""
    def count(width):
        return dense_parameter_count(dim + time_features, width, depth, dim)
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
class ScoreConfig:
    dim: int = 2
    width: int = 51
    depth: int = 6
    # Fixed sine/cosine features; these introduce no trainable parameters.
    time_features: int = 32
    time_max_frequency: float = 1000.0
    target_parameter_count: int = 15299
    beta_min: float = 0.1
    beta_max: float = 20.0
    time_epsilon: float = 1e-3
    # t = time_epsilon + (1 - time_epsilon) * U**power; 1 preserves uniform time.
    # The 8D benchmark overrides this default to 2 to emphasize lower noise.
    time_sampling_power: float = 1.0
    sampling_steps: int = 500
    sampling_batch_size: int = 256
    denoise: bool = True
    # Training examples are noisy data at independently sampled times, not paths.
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
    max_gradient_norm: float = 1.0
    # Use EMA parameters for validation/test; zero evaluates the current weights.
    ema_decay: float = 0.99
    updates: int = 1000
    eval_every: int = 50
    log_every: int = 10
    seed: int = 0
    data_seed: int = 12345
    mixture_modes: int = 4
    n_train: int = 1024
    n_val: int = 1024
    n_test: int = 10_000
    eval_samples: int = 10_000
    # 0 sums all points for KL; positive values retain nearest-neighbor estimates.
    n_neighbor: int = 0
    density_bandwidth: float = 0.2
    # Zero excludes self and divides its density by population size minus one.
    # Positive weights retain the full-population normalization.
    density_self_weight: float = 0.5
    dtype: str = "float32"

    @property
    def parameter_count(self):
        return dense_parameter_count(self.dim + self.time_features, self.width, self.depth, self.dim)

    def validate(self):
        validate_learning_rate_schedule(self)
        for name in ("dim", "width", "depth", "time_features", "target_parameter_count",
                     "sampling_steps", "sampling_batch_size", "batch_size", "eval_every", "log_every",
                     "mixture_modes", "n_train", "n_val", "n_test", "eval_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.n_neighbor, bool) or not isinstance(self.n_neighbor, int)
                or self.n_neighbor < 0):
            raise ValueError("n_neighbor must be a nonnegative integer (0 uses all points)")
        if (isinstance(self.density_self_weight, bool)
                or not isinstance(self.density_self_weight, (int, float))
                or not math.isfinite(self.density_self_weight)
                or not 0 <= self.density_self_weight <= 1):
            raise ValueError("density_self_weight must be finite and in [0, 1]")
        if self.time_features % 2:
            raise ValueError("time_features must be even for sine/cosine pairs")
        if isinstance(self.updates, bool) or not isinstance(self.updates, int) or self.updates < 0:
            raise ValueError("updates must be a nonnegative integer")
        for name in ("seed", "data_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
                raise ValueError(f"{name} must be an integer in [0, 2**32)")
        for name in ("time_max_frequency", "beta_min", "beta_max", "time_epsilon",
                     "learning_rate", "adam_epsilon", "density_bandwidth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.time_max_frequency < 1:
            raise ValueError("time_max_frequency must be at least 1")
        if self.beta_max < self.beta_min:
            raise ValueError("beta_max must be at least beta_min")
        if self.time_epsilon >= 1:
            raise ValueError("time_epsilon must lie strictly between zero and one")
        if (isinstance(self.time_sampling_power, bool)
                or not isinstance(self.time_sampling_power, Real)
                or not math.isfinite(self.time_sampling_power) or self.time_sampling_power < 1):
            raise ValueError("time_sampling_power must be finite and at least 1")
        for name in ("adam_beta1", "adam_beta2", "ema_decay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must lie in [0, 1)")
        if (isinstance(self.max_gradient_norm, bool) or not math.isfinite(self.max_gradient_norm)
                or self.max_gradient_norm < 0):
            raise ValueError("max_gradient_norm must be finite and nonnegative")
        if not isinstance(self.denoise, bool):
            raise ValueError("denoise must be boolean")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        return self


def matched_config(dk_cfg=None, **overrides):
    """Inherit data/evaluation settings and match the current DK trainable budget."""
    dk_cfg = Config() if dk_cfg is None else dk_cfg
    dk_cfg.validate()
    values = asdict(ScoreConfig())
    for name in ("dim", "depth", "updates", "eval_every", "seed", "data_seed", "mixture_modes",
                 "n_train", "n_val", "n_test", "eval_samples", "n_neighbor", "density_bandwidth",
                 "density_self_weight", "dtype"):
        values[name] = getattr(dk_cfg, name)
    # A scheduled DK run uses a fixed bandwidth for comparison and selection.
    values["density_bandwidth"] = evaluation_bandwidth(dk_cfg)
    values.update(target_parameter_count=dk_parameter_budget(dk_cfg), batch_size=dk_cfg.paths,
                  sampling_batch_size=dk_cfg.batch_size, sampling_steps=dk_cfg.steps)
    # Validate sizing inputs before using them in the width search.
    cfg = replace(ScoreConfig(**values), **overrides).validate()
    if "width" not in overrides:
        cfg = replace(cfg, width=closest_width(
            cfg.target_parameter_count, cfg.dim, cfg.depth, cfg.time_features))
    return cfg.validate()

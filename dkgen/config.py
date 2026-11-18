"""Model settings and explicit, configurable experiment defaults."""

from dataclasses import dataclass
import math

from .schedule import ScheduleSettings, validate_schedule_settings


@dataclass(frozen=True)
class Config(ScheduleSettings):
    dim: int = 2
    # Drift network: hidden width and number of hidden SiLU layers.
    width: int = 24
    depth: int = 3
    # Independent diffusion network architecture, used in neural mode.
    diffusion_width: int = 18
    diffusion_depth: int = 2
    drift_initialization: str = "data_rms"
    diffusion_initialization: str = "data_rms"
    dt: float = 0.002
    T: float = 1.0
    # Generated SDE paths per gradient update, advanced in one forward pass.
    paths: int = 1024
    # Independent paths for growth estimation; used only with adaptive alpha.
    pilot_paths: int = 1024
    batch_size: int = 1024
    # KDE losses: averaged_kl, forward_kl_kde, backward_kl.
    # Legacy forward_kl retains assignment-count weights.
    loss_mode: str = "averaged_kl"
    # Terminal coefficient: kde = (p/h + log(p/h))/2.
    # Alternatives: count_squared = (L*c/N_data)**2/2,
    # with one nearest-endpoint assignment per datum and half outside the square.
    # count_only = L*c/(2*N_data),
    # count_log = (L*c/N_data + log(p/h))/2,
    # three_term = (p/h + L*c/N_data + log(p/h))/3.
    response_weight_mode: str = "kde"
    # KDE coefficient baseline: freeze the preceding accepted update's
    # ordinary mean; the first update uses zero. "none" retains uncentered weights.
    response_baseline: str = "previous_mean"
    # Gaussian density estimates: 0 uses all points; positive k uses k neighbors.
    n_neighbor: int = 0
    # Half the query's own Gaussian peak, normalized by the full population.
    # 0 excludes self and uses population - 1; positive weights use population.
    density_self_weight: float = 0.5
    # Gaussian kernel standard deviation in normalized coordinates, not SDE sigma.
    density_bandwidth: float = 0.1
    # Schedule defaults are inherited from ScheduleSettings in schedule.py.
    # The fixed model values here apply when that field is not scheduled.
    # Assignment neighbors for forward_kl and count-color diagnostics.
    neighbors: int = 1
    # Used only in forward_kl: subtract this constant divided by paths.
    score_baseline: float = 1.0
    # constant_norm uses an inward tail of magnitude drift_cap; linear uses -lambda*x.
    drift_tail: str = "constant_norm"
    # Attraction coefficient lambda, used only with drift_tail="linear".
    tail_attraction: float = 1.0
    # In sigma mode, alpha_n is proportional to sigma(x_(n+1)).
    alpha_mode: str = "sigma"
    # Target mean alpha using the previous update's mean sigma; None uses a fixed multiplier.
    alpha_sigma_target: float | None = 35.0
    # Used only with alpha_mode="constant".
    alpha: float = 25.0
    # Used only by the legacy normalized rule when alpha_sigma_multiplier is None.
    alpha_cap: float = 100.0
    # Used when alpha_sigma_target is None; None preserves the legacy normalized rule.
    alpha_sigma_multiplier: float | None = 40.0
    field_bounds: str = "smooth"
    # Neural drift cap, also the tail magnitude and full-field bound in constant_norm mode.
    drift_cap: float = 4.0
    # drift_norm uses the full drift magnitude, including the analytic tail.
    diffusion_mode: str = "neural"
    # None uses drift_cap; a saved numeric value preserves an older run's threshold.
    sigma_drift_scale: float | None = None
    # Used only in constant mode, everywhere in normalized coordinates.
    constant_sigma: float = 0.5
    # Diffusion bounds for drift_norm and smoothly bounded neural modes.
    sigma_cap: float = 4.0
    sigma_min: float = 0.3
    # None: small random diffusion output weights and zero bias.
    # A number retains constant interior initialization using zero output weights.
    initial_sigma: float | None = None
    initial_rms_radius: float = 0.9
    # Learn the mean and per-coordinate standard deviation of the initial Gaussian.
    train_initial: bool = True
    # Lower bound for the learned per-coordinate standard deviation s0.
    initial_s0_min: float = 0.3
    # optimizer: str = "sgd"
    # learning_rate: float = 0.003
    optimizer: str = "adam"
    learning_rate: float = 0.001
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    drift_learning_rate_multiplier: float = 1.0
    # kl: cap each network's absolute linearized KL change using its proposed
    # Adam/SGD step. norm retains the earlier parameter-increment norm cap.
    update_clip_mode: str = "kl"
    # Each network gets this fraction of the absolute current training KL.
    kl_clip_fraction: float = 1.0 / 30.0
    # Used only in norm mode, per network; zero disables norm clipping.
    max_update_norm: float = 0.01
    # L2 penalties on weight matrices before update clipping; biases are excluded.
    # Zero disables the penalty for that network.
    drift_weight_decay: float = 1e-5
    diffusion_weight_decay: float = 1e-5
    updates: int = 5000
    eval_every: int = 50
    log_every: int = 10
    # Trigger on the displayed raw gradient norm; zero disables this diagnostic.
    nu_diagnostic_threshold: float = 20.0
    nu_diagnostic_paths: int = 100
    # At most one diagnostic per this many updates while above the threshold.
    nu_diagnostic_interval: int = 50
    seed: int = 0
    data_seed: int = 12345
    mixture_modes: int = 4
    n_train: int = 1024
    # Held-out data count AND fresh generated sample count at each validation.
    n_val: int = 1024
    n_test: int = 10_000
    eval_samples: int = 10_000
    dtype: str = "float32"

    @property
    def steps(self):
        return int(round(self.T / self.dt))

    @property
    def initial_raw_diffusion_output(self):
        # A random output layer has no single, spatially constant raw output.
        if self.diffusion_mode != "neural" or self.initial_sigma is None:
            return None
        excess = self.initial_sigma - self.sigma_min
        if self.field_bounds == "smooth":
            # Inverse of sigma_min + (sigma_cap - sigma_min) * sigmoid(raw).
            return math.log(excess) - math.log(self.sigma_cap - self.initial_sigma)
        # Stable inverse softplus, including small and large positive excess.
        return excess + math.log(-math.expm1(-excess))

    def validate(self):
        if not isinstance(self.train_initial, bool):
            raise ValueError("train_initial must be a boolean")
        if (isinstance(self.initial_s0_min, bool)
                or not isinstance(self.initial_s0_min, (int, float))
                or not math.isfinite(self.initial_s0_min) or self.initial_s0_min <= 0):
            raise ValueError("initial_s0_min must be finite and positive")
        for name in ("dim", "width", "depth", "diffusion_width", "diffusion_depth",
                     "paths", "pilot_paths", "batch_size",
                     "neighbors", "eval_every", "log_every", "n_train", "n_val", "n_test", "eval_samples",
                     "mixture_modes", "nu_diagnostic_paths", "nu_diagnostic_interval"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.n_neighbor, bool) or not isinstance(self.n_neighbor, int)
                or self.n_neighbor < 0):
            raise ValueError("n_neighbor must be a nonnegative integer (0 uses all points)")
        validate_schedule_settings(self)
        if self.schedule == "prescribed":
            if self.alpha_mode != "sigma" or self.alpha_sigma_target is None:
                raise ValueError("prescribed schedule requires normalized sigma-based alpha")
            if self.field_bounds != "smooth" or self.diffusion_mode not in ("neural", "drift_norm"):
                raise ValueError("prescribed schedule requires smooth bounds and neural or drift_norm diffusion")
            if self.sigma_cap <= max(self.schedule_sigma_min_start, self.schedule_sigma_min_end):
                raise ValueError("prescribed schedule requires sigma_cap above both scheduled floors")
            if (self.diffusion_mode == "neural" and self.initial_sigma is not None
                    and self.initial_sigma <= self.schedule_sigma_min_start):
                raise ValueError("initial_sigma must be above the scheduled starting sigma_min")
        for name in ("dt", "T", "tail_attraction", "sigma_min", "initial_rms_radius",
                     "learning_rate", "drift_learning_rate_multiplier", "drift_cap", "sigma_cap",
                     "constant_sigma", "density_bandwidth", "adam_epsilon"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.optimizer not in ("adam", "sgd"):
            raise ValueError("optimizer must be adam or sgd")
        if self.update_clip_mode not in ("kl", "norm"):
            raise ValueError("update_clip_mode must be kl or norm")
        if (isinstance(self.kl_clip_fraction, bool)
                or not math.isfinite(self.kl_clip_fraction) or not 0 < self.kl_clip_fraction <= 1):
            raise ValueError("kl_clip_fraction must be finite and in (0, 1]")
        for name in ("adam_beta1", "adam_beta2"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must lie in [0, 1)")
        if self.sigma_drift_scale is not None:
            if not math.isfinite(self.sigma_drift_scale) or self.sigma_drift_scale <= 0:
                raise ValueError("sigma_drift_scale must be None or finite and positive")
        if self.alpha_sigma_multiplier is not None:
            if not math.isfinite(self.alpha_sigma_multiplier) or self.alpha_sigma_multiplier < 0:
                raise ValueError("alpha_sigma_multiplier must be None or finite and nonnegative")
        if self.alpha_sigma_target is not None:
            if not math.isfinite(self.alpha_sigma_target) or self.alpha_sigma_target <= 0:
                raise ValueError("alpha_sigma_target must be None or finite and positive")
        if self.diffusion_mode not in ("constant", "neural", "drift_norm"):
            raise ValueError("diffusion_mode must be constant, neural, or drift_norm")
        if self.diffusion_mode == "drift_norm" and self.sigma_cap <= self.sigma_min:
            raise ValueError("sigma_cap must be greater than sigma_min in drift_norm mode")
        if self.initial_sigma is not None and (
                not math.isfinite(self.initial_sigma) or self.initial_sigma <= 0):
            raise ValueError("initial_sigma must be None or finite and positive")
        if (self.schedule != "prescribed" and self.diffusion_mode == "neural"
                and self.initial_sigma is not None and self.initial_sigma <= self.sigma_min):
            raise ValueError("initial_sigma must be greater than sigma_min")
        for name in ("alpha", "max_update_norm", "score_baseline",
                     "drift_weight_decay", "diffusion_weight_decay", "nu_diagnostic_threshold",
                     "alpha_cap"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.alpha_mode not in ("constant", "adaptive", "sigma"):
            raise ValueError("alpha_mode must be constant, adaptive, or sigma")
        if self.field_bounds not in ("smooth", "none"):
            raise ValueError("field_bounds must be smooth or none")
        if self.drift_tail not in ("constant_norm", "linear"):
            raise ValueError("drift_tail must be constant_norm or linear")
        if self.drift_tail == "constant_norm" and self.field_bounds != "smooth":
            raise ValueError("constant_norm drift_tail requires smooth field_bounds")
        if self.diffusion_mode == "neural" and self.field_bounds == "smooth":
            if self.sigma_cap <= self.sigma_min:
                raise ValueError("sigma_cap must be greater than sigma_min for smooth bounds")
            if self.initial_sigma is not None and self.initial_sigma >= self.sigma_cap:
                raise ValueError("initial_sigma must be less than sigma_cap for smooth bounds")
        if self.steps < 1 or not math.isclose(self.steps * self.dt, self.T, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("T must be an integer multiple of dt")
        if self.neighbors > self.paths:
            raise ValueError("neighbors cannot exceed paths")
        if self.loss_mode not in ("forward_kl", "averaged_kl", "forward_kl_kde", "backward_kl"):
            raise ValueError("loss_mode must be forward_kl, averaged_kl, forward_kl_kde, or backward_kl")
        if self.loss_mode in ("forward_kl_kde", "backward_kl") and self.response_weight_mode != "kde":
            raise ValueError("Directional KDE losses require response_weight_mode=kde")
        if self.response_weight_mode not in ("count_squared", "count_log", "count_only", "three_term", "kde"):
            raise ValueError("response_weight_mode must be count_squared, count_log, count_only, three_term, or kde")
        if self.response_baseline not in ("none", "previous_mean"):
            raise ValueError("response_baseline must be none or previous_mean")
        if (isinstance(self.density_self_weight, bool)
                or not isinstance(self.density_self_weight, (int, float))
                or not math.isfinite(self.density_self_weight)
                or not 0 <= self.density_self_weight <= 1):
            raise ValueError("density_self_weight must be finite and in [0, 1]")
        if self.loss_mode != "forward_kl":
            if (self.n_neighbor > 0 or self.density_self_weight == 0) and self.paths < 2:
                raise ValueError("KDE losses require at least two paths for self-exclusion")
            if self.n_neighbor > min(self.n_train, self.paths - 1):
                raise ValueError("n_neighbor cannot exceed n_train or paths - 1 for KDE losses")
        if self.update_clip_mode == "kl" and self.n_neighbor > min(self.n_train - 1, self.paths - 1):
            raise ValueError("KL update clipping requires n_neighbor <= min(n_train - 1, paths - 1)")
        if (self.update_clip_mode == "kl" and self.density_self_weight == 0
                and min(self.n_train, self.paths) < 2):
            raise ValueError("KL update clipping with density_self_weight=0 needs at least two data and paths")
        if not isinstance(self.updates, int) or self.updates < 0:
            raise ValueError("updates must be a nonnegative integer")
        for name in ("seed", "data_seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or not 0 <= value < 2**32:
                raise ValueError(f"{name} must be an integer in [0, 2**32)")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        for name in ("drift_initialization", "diffusion_initialization"):
            if getattr(self, name) not in ("data_rms", "glorot"):
                raise ValueError(f"{name} must be data_rms or glorot")
        return self

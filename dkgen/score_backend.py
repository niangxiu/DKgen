"""Model-specific operations used by the shared VP/EDM training loop.

Metadata inspection stays independent of JAX. Construct a backend after setting
JAX precision; the model implementations and their numerical formulas are reused.
"""

from dataclasses import dataclass
import json
from typing import Callable

import numpy as np


def checkpoint_metadata(path):
    """Read VP/EDM JSON metadata without importing JAX or loading any weights."""
    with np.load(path, allow_pickle=False) as archive:
        encoded = archive["metadata_json"]
        if encoded.dtype != np.uint8 or encoded.ndim != 1:
            raise ValueError("Malformed score checkpoint metadata")
        metadata = json.loads(encoded.tobytes().decode("utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Score checkpoint metadata must be an object")
    if metadata.get("model_kind") not in ("edm", "score_vp"):
        raise ValueError("Checkpoint is neither an EDM nor a VP score model")
    return metadata


def checkpoint_method(path):
    """Identify a VP or EDM checkpoint by reading only its numeric JSON metadata."""
    return {"edm": "edm", "score_vp": "vp"}[checkpoint_metadata(path)["model_kind"]]


@dataclass(frozen=True)
class ScoreBackend:
    method: str
    model_kind: str
    config_type: type
    engine_type: type
    init_params: Callable
    load_checkpoint: Callable
    save_checkpoint: Callable
    loss_key: str
    score_field: Callable

    @property
    def diagnostic_level_name(self):
        return "sigma" if self.method == "edm" else "t"

    def diagnostic_levels(self, cfg):
        """VP times or EDM Gaussian standard deviations in normalized z units."""
        return (.1, .5, .9)

    def network_evaluations(self, cfg):
        return (2 * cfg.sampling_steps - 1 if self.method == "edm"
                else cfg.sampling_steps + int(cfg.denoise))

    def description(self, cfg):
        """Training and sampler metadata, without changing either method's recipe."""
        common = {
            "model_kind": self.model_kind, "optimizer": "Adam",
            "evaluation_weights": "EMA" if cfg.ema_decay else "current",
            "evaluation_density_bandwidth": cfg.density_bandwidth,
            "network_evaluations_per_sample": self.network_evaluations(cfg),
        }
        if self.method == "edm":
            backbone = getattr(cfg, "backbone", "residual")
            architecture = ("plain SiLU MLP with concatenated fixed sine/cosine noise features"
                            if backbone == "mlp" else
                            "dense residual MLP with separate noise embedding and per-block conditioning")
            return common | {
                "training_objective": "EDM weighted denoising MSE, averaged over examples and coordinates",
                "training_noise": {"distribution": "lognormal", "log_mean": cfg.noise_log_mean,
                                   "log_std": cfg.noise_log_std},
                "sampler": {"name": "EDM deterministic Heun", "rho": cfg.rho,
                            "sigma_min": cfg.sigma_min, "sigma_max": cfg.sigma_max,
                            "steps": cfg.sampling_steps, "churn": 0,
                            "precision": cfg.dtype, "final_step": "Euler to zero"},
                "backbone": backbone,
                "architecture": architecture,
                "model_coordinates": {"formula": "u = data_scale * z; return z = u / data_scale",
                                      "data_scale": cfg.data_scale, "sigma_data": cfg.sigma_data},
                "recipe_reference": "https://arxiv.org/abs/2206.00364",
                "note": "KL is an evaluation metric, not the EDM training objective. "
                        "All metrics use the full state dimension.",
            }
        return common | {
            "training_objective": "Gaussian noise prediction MSE (variance-weighted denoising score matching)",
            "training_time_sampling": {"formula": "t = epsilon + (1-epsilon) * U**power",
                                       "power": cfg.time_sampling_power},
            "sampler": {"name": "VP reverse SDE Euler-Maruyama", "steps": cfg.sampling_steps,
                        "time_epsilon": cfg.time_epsilon, "denoise": cfg.denoise,
                        "precision": cfg.dtype},
            "architecture": "dense SiLU MLP with fixed sine/cosine time features",
            "note": "KL is an evaluation metric, not the denoising training objective. "
                    "All metrics use the full state dimension.",
        }


def _edm_score(params, x, level, cfg):
    """Score in normalized z units, with level the noise standard deviation in z."""
    from .edm_model import denoise

    denoised = denoise(params, cfg.data_scale * x, cfg.data_scale * level, cfg) / cfg.data_scale
    return (denoised - x) / level**2


def backend_for(method):
    """Return adapters to existing model code; callers set JAX precision first."""
    if method == "edm":
        from .edm_config import EDMConfig
        from .edm_model import EDMEngine, init_params
        from .edm_state import load_edm_checkpoint, save_edm_checkpoint

        return ScoreBackend("edm", "edm", EDMConfig, EDMEngine, init_params,
                            load_edm_checkpoint, save_edm_checkpoint, "edm_loss", _edm_score)
    if method == "vp":
        from .score_config import ScoreConfig
        from .score_model import ScoreEngine, init_params, score
        from .score_state import load_score_checkpoint, save_score_checkpoint

        return ScoreBackend("vp", "score_vp", ScoreConfig, ScoreEngine, init_params,
                            load_score_checkpoint, save_score_checkpoint, "noise_mse", score)
    raise ValueError("Score method must be edm or vp")

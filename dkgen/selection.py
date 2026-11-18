"""Recover actual validation-selected states when resuming a run."""

from copy import deepcopy
import math
from pathlib import Path

from .checkpoint import load_checkpoint
from .schedule import compatible_scheduled_configs


_SW1 = "validation_sliced_wasserstein1"
_KL = "validation_averaged_kl"


def _valid_score(value, metric):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (metric == _KL or value >= 0))


def _require_prior(candidate, iteration, score, description):
    if (candidate is None or candidate["metadata"]["iteration"] < iteration
            or candidate["metadata"]["best_validation"] > score):
        raise ValueError(
            f"Resume requires a compatible {description} checkpoint containing "
            "the recorded best state or a later improvement")


def load_resume_bests(source_path, saved, data_hash, *, checkpoint_loader=None):
    """Return available SW1-best and KL-best states at or before the source.

    Sibling ``best.npz`` and ``best_kl.npz`` must retain the source's saved
    configuration and dataset, allowing the prescribed fields to differ at
    their respective schedule stages. Future states are not inherited when restarting
    from an older selected checkpoint. A source best checkpoint's own validation
    metrics may also seed the other selection rule using its actual parameters,
    PRNG key, and method-specific saved state. Scalar pointers alone never create
    a state. A custom loader supports methods with a different checkpoint format.
    """
    dk_checkpoint_format = checkpoint_loader is None
    checkpoint_loader = load_checkpoint if dk_checkpoint_format else checkpoint_loader
    source_meta = saved["metadata"]
    source_iteration = source_meta["iteration"]
    if source_meta["data_fingerprint"] != data_hash:
        raise ValueError("Resume source dataset fingerprint does not match the saved data")
    if source_meta["role"] not in ("best", "latest"):
        raise ValueError("Resume source must be a best or latest checkpoint")
    if source_meta["selection_metric"] not in (_SW1, _KL):
        raise ValueError("Resume source has an unsupported selection metric")

    candidates = {_SW1: [], _KL: []}
    if source_meta["role"] == "best":
        source_metric = source_meta["selection_metric"]
        candidates[source_metric].append(saved)
        metrics = source_meta.get("extra", {}).get("validation_metrics", {})
        other_metric = _KL if source_metric == _SW1 else _SW1
        other_key = "averaged_kl" if other_metric == _KL else "sliced_wasserstein1"
        if isinstance(metrics, dict) and _valid_score(metrics.get(other_key), other_metric):
            metadata = deepcopy(source_meta)
            metadata.update(selection_metric=other_metric,
                            best_validation=float(metrics[other_key]),
                            best_iteration=source_iteration,
                            pending_validation=False)
            candidates[other_metric].append(dict(saved, metadata=metadata))

    directory = Path(source_path).resolve().parent
    for filename, metric in (("best.npz", _SW1), ("best_kl.npz", _KL)):
        path = directory / filename
        if not path.exists():
            continue
        try:
            candidate = checkpoint_loader(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Cannot load resume candidate {filename}: {exc}") from exc
        metadata = candidate["metadata"]
        if metadata["iteration"] > source_iteration:
            continue
        configs_match = (compatible_scheduled_configs(
            metadata["config"], metadata["iteration"], source_meta["config"], source_iteration)
            if dk_checkpoint_format else metadata["config"] == source_meta["config"])
        if (metadata["role"] != "best" or metadata["selection_metric"] != metric
                or metadata["data_fingerprint"] != data_hash
                or not configs_match):
            raise ValueError(f"Incompatible resume candidate {filename}")
        candidates[metric].append(candidate)

    chosen = {metric: min(values, key=lambda item: (
        item["metadata"]["best_validation"], item["metadata"]["iteration"]))
        if values else None for metric, values in candidates.items()}

    if source_meta["role"] == "latest":
        # Historical dimension benchmarks tracked only KL in latest.npz.
        # Require the actual selected state for whichever metric it records;
        # the common runner can establish a missing SW1 winner by validation.
        primary = source_meta["selection_metric"]
        description = "best.npz (SW1)" if primary == _SW1 else "best_kl.npz (KL)"
        _require_prior(chosen[primary], source_meta["best_iteration"],
                       source_meta["best_validation"], description)
        extra = source_meta.get("extra", {})
        pointer_names = ("best_kl_iteration", "best_validation_averaged_kl")
        present = [name in extra for name in pointer_names]
        if any(present):
            if not all(present):
                raise ValueError("Incomplete saved KL selection state")
            kl_iteration, kl_score = (extra[name] for name in pointer_names)
            if kl_iteration is not None or kl_score is not None:
                if (isinstance(kl_iteration, bool) or not isinstance(kl_iteration, int)
                        or not 0 <= kl_iteration <= source_iteration
                        or not _valid_score(kl_score, _KL)):
                    raise ValueError("Invalid saved KL selection state")
                _require_prior(chosen[_KL], kl_iteration, kl_score, "best_kl.npz (KL)")
    return chosen[_SW1], chosen[_KL]

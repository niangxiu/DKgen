"""Summarize the paper's test tables, target reference, and four-seed comparison."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from dkgen.data import make_mixture_data, target_metadata
from dkgen.metrics import evaluate_samples
from run_paper import matrix, run_name


def evaluate_target_reference(dim):
    """Compare two independent target test sets using the paper's estimator."""
    reference = make_mixture_data(12345, 1024, 1024, 10000, dim=dim)
    comparison = make_mixture_data(12346, 1024, 1024, 10000, dim=dim)
    radius = float(reference["R"])
    target = target_metadata(dim, 1729, radius) if dim == 8 else None
    return evaluate_samples(
        comparison["test_raw"], reference["test_raw"], n_neighbor=0,
        density_bandwidth=0.03, density_self_weight=0.5,
        normalization_scale=radius, target=target,
    )


def read_result(directory):
    with (directory / "summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    metrics = summary["test_metrics_best_kl"]
    if (metrics["generated_count"], metrics["reference_count"],
            metrics["kl_n_neighbor"], metrics["kl_density_bandwidth"],
            metrics["kl_density_self_weight"]) != (10000, 10000, 0, 0.03, 0.5):
        raise ValueError(f"{directory} does not use the paper's test settings")
    return {
        "checkpoint_update": summary["best_kl_iteration"],
        "parameters": summary["parameter_count"],
        "validation_averaged_kl": summary["best_validation_averaged_kl"],
        "test_averaged_kl": metrics["averaged_kl"],
        "test_forward_kl": metrics["forward_kl"],
        "test_backward_kl": metrics["reverse_kl"],
        "precision_percent": 100 * metrics["knn_precision"],
        "off_mode_percent": 100 * metrics["knn_outside_fraction"],
        "mean_outside_distance": metrics["knn_outside_distance_mean"],
        "p95_outside_distance": metrics["knn_outside_distance_p95"],
        "mode_probability_error": metrics["region_probability_error"],
        "wall_seconds": summary["wall_seconds_this_run"],
    }


def report(root, dimensions):
    rows = {}
    for dim, method, seed, objective in matrix(dimensions, None):
        name = run_name(dim, method, seed, objective)
        rows[name] = read_result(root / name)
    results = {"runs": rows, "target_reference": {}}
    for dim in dimensions:
        target = evaluate_target_reference(dim)
        results["target_reference"][f"{dim}d"] = {
            "averaged_kl": target["averaged_kl"],
            "precision_percent": 100 * target["knn_precision"],
        }
    if 8 in dimensions:
        comparison = {}
        for method in ("dk", "edm"):
            four_runs = [rows[run_name(8, method, seed)] for seed in range(4)]
            stats = {}
            for metric in ("off_mode_percent", "mode_probability_error"):
                values = [run[metric] for run in four_runs]
                stats[metric] = {"mean": mean(values), "sample_sd": stdev(values)}
            comparison[method] = stats
        results["four_seed_8d"] = comparison
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/paper"))
    parser.add_argument("--dim", type=int, choices=(2, 8), help="Report one dimension")
    args = parser.parse_args()
    result = report(args.results, (args.dim,) if args.dim else (2, 8))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

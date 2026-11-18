"""Run the DK and EDM experiments in Sections 4.2 and 4.3."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


OBJECTIVES = {
    "averaged": "averaged_kl",
    "forward": "forward_kl_kde",
    "backward": "backward_kl",
}
WIDTHS = {2: 25, 8: 26}
ROOT = Path(__file__).resolve().parent


def run_name(dim, method, seed, objective="averaged"):
    if method == "dk":
        return f"dk-{dim}d-{objective}-seed{seed}"
    return f"edm-{dim}d-seed{seed}"


def command(dim, method, seed, objective, output, *, smoke=False):
    """Build a command from the paper settings or a tiny installation check."""
    updates, count, test_count = (1, 16, 32) if smoke else (5000, 1024, 10000)
    shared = [
        "--dim", str(dim), "--seed", str(seed), "--data-seed", "12345",
        "--rotation-seed", "1729", "--mixture-modes", "4",
        "--n-train", str(count), "--n-val", str(count),
        "--n-test", str(test_count), "--eval-samples", str(test_count),
        "--batch-size", str(count), "--updates", str(updates),
        "--eval-every", "50", "--learning-rate", "0.001",
        "--n-neighbor", "0", "--density-self-weight", "0.5",
        "--output", str(output),
    ]
    if method == "dk":
        options = [
            "--loss-mode", OBJECTIVES[objective], "--optimizer", "adam",
            "--width", "24", "--depth", "3", "--diffusion-width", "18",
            "--diffusion-depth", "2", "--diffusion-mode", "neural",
            "--sigma-cap", "2", "--paths", str(count), "--pilot-paths", str(count),
            "--initial-rms-radius", "0.9", "--initial-s0-min", "0.3",
            "--schedule", "prescribed", "--schedule-updates", "4000",
            "--schedule-drift-cap-start", "4", "--schedule-drift-cap-end", "4",
            "--schedule-sigma-min-start", "0.1", "--schedule-sigma-min-end", "0.1",
            "--schedule-alpha-sigma-target-start", "42",
            "--schedule-alpha-sigma-target-end", "42",
            "--schedule-density-bandwidth-start", "0.3",
            "--schedule-density-bandwidth-end", "0.03",
            "--schedule-evaluation-bandwidth", "0.03",
            "--response-weight-mode", "kde", "--response-baseline", "previous_mean",
            "--nu-diagnostic-threshold", "0",
            "--dt", "0.1" if smoke else "0.002",
        ]
        return [sys.executable, "-m", "dkgen.train", *shared, *options]
    options = [
        "--method", "edm", "--backbone", "mlp", "--depth", "3",
        "--width", str(WIDTHS[dim]), "--density-bandwidth", "0.03",
        "--sampling-steps", "4" if smoke else "128",
    ]
    return [sys.executable, "-m", "dkgen.score_train", *shared, *options]


def matrix(dimensions, method):
    for dim in dimensions:
        if method != "edm":
            for objective in OBJECTIVES:
                yield dim, "dk", 0, objective
        if method != "dk":
            yield dim, "edm", 0, "averaged"
        if dim == 8:
            for seed in (1, 2, 3):
                if method != "edm":
                    yield dim, "dk", seed, "averaged"
                if method != "dk":
                    yield dim, "edm", seed, "averaged"


def completed(output, *, smoke):
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        return False
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    test = summary.get("test_metrics_best_kl") or {}
    updates, count = (1, 32) if smoke else (5000, 10000)
    finished = summary.get("updates_completed")
    if finished is None:
        # The 2D DK runner writes its summary only after training and testing,
        # but records the requested update count in config.json.
        with (output / "config.json").open(encoding="utf-8") as stream:
            finished = json.load(stream)["updates"]
    return (finished == updates
            and test.get("generated_count") == test.get("reference_count") == count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="Root for the new run folders")
    parser.add_argument("--dim", type=int, choices=WIDTHS)
    parser.add_argument("--method", choices=("dk", "edm"))
    parser.add_argument("--objective", choices=OBJECTIVES)
    parser.add_argument("--seed", type=int, choices=range(4))
    parser.add_argument("--smoke", action="store_true", help="One tiny update per method in each selected dimension")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without training")
    args = parser.parse_args()
    if args.objective and args.method != "dk":
        parser.error("--objective requires --method dk")
    if args.seed is not None and (args.dim is None or args.method is None):
        parser.error("--seed requires --dim and --method")
    if args.seed and (args.dim != 8 or args.objective not in (None, "averaged")):
        parser.error("Seeds 1-3 apply only to the 8D averaged-KL DK and EDM runs")
    if args.seed and args.smoke:
        parser.error("The smoke check uses seed 0")
    default_root = Path("results") / ("smoke" if args.smoke else "paper")
    results = (args.results or default_root).resolve()
    dimensions = [args.dim] if args.dim else WIDTHS
    for dim, method, seed, objective in matrix(dimensions, args.method):
        if args.smoke and seed != 0:
            continue
        if args.seed is not None and seed != args.seed:
            continue
        if args.objective and objective != args.objective:
            continue
        output = results / run_name(dim, method, seed, objective)
        launch = command(dim, method, seed, objective, output, smoke=args.smoke)
        print(shlex.join(launch), flush=True)
        if args.dry_run:
            continue
        if output.exists():
            if completed(output, smoke=args.smoke):
                print(f"Skipping completed run {output.name}", flush=True)
                continue
            parser.error(f"Existing run does not match this recipe: {output}. Use a fresh --results folder.")
        subprocess.run(launch, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

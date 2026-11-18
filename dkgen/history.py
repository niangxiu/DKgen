"""Plot recorded validation errors without running the model or loading JAX."""

import argparse
import json
import math
from pathlib import Path
import tempfile

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np

from .file_io import replace_with_retry


METRICS = (
    ("averaged_kl", "Estimated averaged KL", "Estimated averaged KL (nats)"),
    ("sliced_wasserstein1", "Sliced Wasserstein-1", "Sliced Wasserstein-1 (coordinate units)"),
    ("covariance_error", "Covariance error", "Frobenius error (squared coordinate units)"),
    ("region_probability_error", "Region probability error", "Sum of absolute probability differences"),
)


def read_validation_history(history_path):
    """Read one run's validation records, excluding updates and test results.

    Ignore an incomplete final JSON line while another process is writing.
    A completed malformed record is an error. Repeated iterations use the last
    validation record; no smoothing, interpolation, or missing values are added.
    Missing or unavailable averaged KL is allowed. Finite negative estimates
    are retained.
    """
    path = Path(history_path)
    if path.is_dir():
        path = path / "history.jsonl"
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    by_step = {}
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if number == len(lines) and not text.endswith("\n"):
                break
            raise ValueError(f"Malformed JSON at {path}:{number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected a JSON object at {path}:{number}")
        if record.get("event") != "validation":
            continue
        step = record.get("iteration")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError(f"Invalid validation iteration at {path}:{number}")
        metrics = record.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError(f"Invalid validation metrics at {path}:{number}")
        for key, _, _ in METRICS:
            value = metrics.get(key)
            if key == "averaged_kl" and value is None:
                continue
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or (key != "averaged_kl" and value < 0)):
                raise ValueError(f"Invalid or missing {key} at {path}:{number}")
        by_step[step] = record
    return [by_step[step] for step in sorted(by_step)]


def save_convergence_plot(history_path, output_path=None):
    """Save four validation-error panels against training update number.

    Return the saved Path, or None when the history has no validation records.
    Only this run's recorded iterations are used, including after a resume.
    """
    history_path = Path(history_path)
    if history_path.is_dir():
        history_path = history_path / "history.jsonl"
    records = read_validation_history(history_path)
    if not records:
        return None
    output_path = Path(output_path) if output_path else history_path.with_name("convergence.png")
    if output_path.resolve() == history_path.resolve():
        raise ValueError("Plot output must differ from the history file")
    if output_path.suffix.lower() != ".png":
        raise ValueError("Convergence plot output must have a .png extension")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.array([record["iteration"] for record in records])
    kl = np.array([record["metrics"].get("averaged_kl") for record in records], dtype=float)
    best_index = None if np.all(np.isnan(kl)) else int(np.nanargmin(kl))
    figure = Figure(figsize=(11, 7.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2, sharex=True).ravel()
    colors = ("#167d8d", "#3666a3", "#9467bd", "#ad6621")
    for axis, (key, title, ylabel), color in zip(axes, METRICS, colors):
        values = [record["metrics"].get(key) for record in records]
        values = np.array([np.nan if value is None else value for value in values])
        if key == "averaged_kl":
            axis.set_yscale("log")
        if key == "averaged_kl" and all(np.isnan(values)):
            axis.text(0.5, 0.5, "KL was not recorded or is unavailable", transform=axis.transAxes,
                      ha="center", va="center", color="0.4")
        elif key == "averaged_kl" and not np.any(values > 0):
            axis.text(0.5, 0.5, "No positive KL estimates to plot", transform=axis.transAxes,
                      ha="center", va="center", color="0.4")
        else:
            if key == "averaged_kl":
                nonpositive = int(np.count_nonzero(values <= 0))
                if nonpositive:
                    axis.text(0.02, 0.98, f"Nonpositive KL values omitted: {nonpositive}",
                              transform=axis.transAxes, ha="left", va="top", fontsize=8, color="0.4")
                # Do not take absolute values or clamp nonpositive estimates.
                values = np.where(values > 0, values, np.nan)
            # NaNs leave gaps at missing or nonpositive KL estimates.
            axis.plot(steps, values, color=color, linewidth=1.6, marker="o", markersize=3.5)
        axis.set(title=title, xlabel="Training update", ylabel=ylabel)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        axis.grid(True, alpha=0.22)
        axis.margins(x=0.035, y=0.12)
        axis.spines[["top", "right"]].set_visible(False)
        if len(steps) == 1:
            axis.set_xlim(steps[0] - 0.5, steps[0] + 0.5)
    if best_index is not None and kl[best_index] > 0:
        axes[0].plot(steps[best_index], kl[best_index], marker="*", markersize=12,
                     color="#d27d16", markeredgecolor="white", markeredgewidth=0.5,
                     linestyle="none", label="Lowest estimated KL")
        axes[0].legend(frameon=False, fontsize=9)
    best_message = ("Estimated averaged KL unavailable" if best_index is None else
                    f"Lowest estimated KL: {kl[best_index]:.5f} at update {steps[best_index]}")
    figure.suptitle(
        f"Validation error history: {history_path.parent.name}\n"
        f"{len(records)} evaluations | Lower is better | "
        f"{best_message}",
        fontsize=12,
    )
    temporary = None
    try:
        # Replace the old image only after the new PNG is complete, so viewers
        # never see a partly written plot during automatic refreshes.
        with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=".convergence-",
                                          suffix=".png", delete=False) as stream:
            temporary = Path(stream.name)
        figure.savefig(temporary, format="png", dpi=160)
        replace_with_retry(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        figure.clear()
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, help="Run directory or history.jsonl file")
    parser.add_argument("--output", type=Path, help="PNG path; defaults to convergence.png beside history")
    args = parser.parse_args(argv)
    try:
        output = save_convergence_plot(args.history, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Could not plot history: {exc}\n")
    if output is None:
        print("No validation records yet; no convergence plot written.")
    else:
        print(f"Saved convergence plot to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

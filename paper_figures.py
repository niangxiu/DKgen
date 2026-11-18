"""Generate Figures 1 through 5 from the seed-0 runs of the paper experiments."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from dkgen.checkpoint import load_checkpoint
from dkgen.config import Config
from dkgen.history import read_validation_history
from dkgen.model import fields


BLUE, ORANGE = "#2476B7", "#DE862B"


def dk_run(results, dimension, objective="averaged"):
    return results / f"dk-{dimension}d-{objective}-seed0"


def edm_run(results, dimension):
    return results / f"edm-{dimension}d-seed0"


def plane_basis(run, dimension):
    """Give the first two original coordinate directions in model coordinates."""
    if dimension == 2:
        return np.eye(2)
    with np.load(run / "target.npz", allow_pickle=False) as target:
        seed = int(target["rotation_seed"])
    gaussian = np.random.default_rng(seed).standard_normal((dimension, dimension))
    rotation, triangular = np.linalg.qr(gaussian)
    rotation *= np.where(np.diag(triangular) < 0.0, -1.0, 1.0)[None, :]
    return rotation[:, :2]


def samples(run):
    with np.load(run / "data.npz", allow_pickle=False) as data:
        reference = data["test"]
    with np.load(run / "test_samples_kl.npz", allow_pickle=False) as output:
        generated = output["generated_normalized"]
    return generated, reference


def coordinate_labels(dimension):
    return ((r"Normalized $x_1$", r"Normalized $x_2$") if dimension == 2
            else (r"$U_1/R$", r"$U_2/R$"))


def setup_plane(axis, limit, xlabel, ylabel=None):
    axis.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel=xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    axis.set_aspect("equal", adjustable="box")
    axis.spines[["top", "right"]].set_visible(False)


def save(figure, path):
    figure.savefig(path, dpi=400, metadata={"CreationDate": None})
    plt.close(figure)
    print(path)


def plot_comparison(results, dimension, basis, path):
    dk, ref = samples(dk_run(results, dimension))
    edm, edm_ref = samples(edm_run(results, dimension))
    np.testing.assert_allclose(ref, edm_ref, rtol=1e-12, atol=1e-12)
    dk, edm, ref = (points @ basis for points in (dk, edm, ref))
    maximum = max(np.max(np.abs(points)) for points in (dk, edm, ref))
    limit = (np.ceil((maximum + 0.025) * 10) / 10 if dimension == 2
             else max(1.1, np.ceil((1.06 * maximum + 0.01) * 20) / 20))
    xlabel, ylabel = coordinate_labels(dimension)
    figure, axes = plt.subplots(1, 2, figsize=(6.7, 3.55), sharex=True, sharey=True)
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.15, top=0.84, wspace=0.19)
    for axis, title, generated in zip(axes, ("Averaged-KL DK", "EDM"), (dk, edm)):
        setup_plane(axis, limit, xlabel)
        axis.set_title(title)
        axis.scatter(*ref.T, s=1.3, color=ORANGE, alpha=0.24, linewidths=0, rasterized=True)
        axis.scatter(*generated.T, s=1.3, color=BLUE, alpha=0.29, linewidths=0, rasterized=True)
    axes[0].set_ylabel(ylabel)
    generated_label = f"Generated ({len(dk):,})" if len(dk) == len(edm) else "Generated"
    labels = (f"Reference ({len(ref):,})", generated_label)
    handles = [Line2D([], [], color=color, marker="o", linestyle="none", markersize=4,
                      label=label) for color, label in zip((ORANGE, BLUE), labels)]
    figure.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    save(figure, path)
    return limit


def plot_fields(run, dimension, basis, limit, path):
    checkpoint = load_checkpoint(run / "best_kl.npz")
    cfg = Config(**checkpoint["metadata"]["config"]).validate()
    params = jax.tree_util.tree_map(jnp.asarray, checkpoint["params"])
    evaluate = jax.jit(jax.vmap(lambda x: fields(params, x, cfg)))

    def grid(size):
        coords = np.linspace(-limit, limit, size)
        xx, yy = np.meshgrid(coords, coords)
        pairs = np.column_stack((xx.ravel(), yy.ravel()))
        f, sigma = evaluate(jnp.asarray(pairs @ basis.T, dtype=cfg.dtype))
        return xx, yy, np.asarray(f) @ basis, np.asarray(sigma).reshape(xx.shape)

    with np.load(run / "data.npz", allow_pickle=False) as data:
        training = data["train"] @ basis
    arrow_x, arrow_y, drift, _ = grid(25)
    xx, yy, _, diffusion = grid(301)
    xlabel, ylabel = coordinate_labels(dimension)
    figure, (ax_f, ax_sigma) = plt.subplots(1, 2, figsize=(7.2, 3.35), layout="constrained")
    for axis in (ax_f, ax_sigma):
        setup_plane(axis, limit, xlabel)
    ax_f.set_ylabel(ylabel)
    ax_f.set_title("(a) Drift" if dimension == 2 else "(a) Projected drift", loc="left")
    ax_sigma.set_title(r"(b) Diffusion $\sigma^\gamma(x)$", loc="left")
    shape = arrow_x.shape
    quiver = ax_f.quiver(arrow_x, arrow_y, drift[:, 0].reshape(shape), drift[:, 1].reshape(shape),
                         color=BLUE, angles="xy", scale_units="xy", scale=35, width=0.0031)
    ax_f.quiverkey(quiver, X=0.8, Y=1.04, U=2, label="2", labelpos="E", coordinates="axes")
    upper = max(cfg.sigma_min + 0.01, np.ceil(10 * diffusion.max()) / 10)
    contour = ax_sigma.contourf(xx, yy, diffusion, levels=np.linspace(cfg.sigma_min, upper, 41),
                                cmap="Blues", zorder=0)
    ax_sigma.set_rasterization_zorder(1)
    for axis in (ax_f, ax_sigma):
        axis.scatter(*training.T, s=3.2, color=ORANGE, alpha=0.3, linewidths=0, rasterized=True)
    figure.colorbar(contour, ax=ax_sigma, label=r"$\sigma^\gamma(x)$")
    save(figure, path)


def plot_validation(results, path):
    runs = [(dk_run(results, 8, name.lower()), f"DK, {name.lower()} KL", color, style)
            for name, color, style in (("Averaged", BLUE, "-"), ("Forward", ORANGE, "--"),
                                       ("Backward", "#37916B", "-."))]
    runs.append((edm_run(results, 8), "EDM", "#9868AA", ":"))
    figure, axis = plt.subplots(figsize=(7.2, 3.1))
    figure.subplots_adjust(left=0.10, right=0.965, bottom=0.18, top=0.76)
    histories = [read_validation_history(run) for run, _, _, _ in runs]
    if any(not history for history in histories):
        raise ValueError("Every 8D seed-0 run must have validation records")
    final_update = max(1, max(history[-1]["iteration"] for history in histories))
    for (run, label, color, style), history in zip(runs, histories):
        steps = np.array([r["iteration"] for r in history])
        values = np.array([r["metrics"].get("averaged_kl") for r in history], dtype=float)
        values[values <= 0] = np.nan
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        best = summary["best_validation_averaged_kl"]
        best_step = summary["best_kl_iteration"]
        axis.semilogy(steps, values, color=color, linestyle=style, label=label)
        if best is not None and best > 0:
            axis.plot(best_step, best, marker="o", markerfacecolor="white",
                      markeredgecolor=color, linestyle="none", markersize=4.5)
    axis.grid(axis="y", which="major", color="0.9")
    axis.spines[["top", "right"]].set_visible(False)
    axis.set(xlim=(0, final_update), title="Full optimization history",
             xlabel="Training update", ylabel="Validation averaged KL")
    handles, labels = axis.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.2", markerfacecolor="white", marker="o", linestyle="none"))
    figure.legend(handles, labels + ["Selected checkpoint"], loc="upper center", ncol=3, frameon=False)
    save(figure, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/paper"))
    parser.add_argument("--output", type=Path, help="Default: figures/ under --results")
    args = parser.parse_args()
    output = args.output or args.results / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8, "pdf.fonttype": 42})
    for dimension, sample_name, field_name in (
        (2, "best_kl_sample_comparison.pdf", "learned_2d_fields.pdf"),
        (8, "best_kl_8d_sample_comparison.pdf", "learned_8d_diffusion.pdf"),
    ):
        run = dk_run(args.results, dimension)
        basis = plane_basis(run, dimension)
        limit = plot_comparison(args.results, dimension, basis, output / sample_name)
        plot_fields(run, dimension, basis, 1.1 if dimension == 2 else limit, output / field_name)
    plot_validation(args.results, output / "validation_kl_8d.pdf")


if __name__ == "__main__":
    main()

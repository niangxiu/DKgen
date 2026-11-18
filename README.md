# DKgen

Code for [Generative SDEs with Trainable Diffusion and Density-Based Losses](https://arxiv.org/abs/2610.41243) by Angxiu Ni. This package generates the synthetic data, trains the models, and produces the figures and metrics for the two- and eight-dimensional Gaussian-mixture experiments.

## Setup

From the repository root, use Python 3.11 or newer in a virtual environment:

```bash
python -m pip install -r requirements.txt
```

The training code runs on CPU, but a GPU is recommended for the full experiments. For GPU training, install the accelerator build of JAX 0.8.2 appropriate for your system. See the [JAX installation instructions](https://docs.jax.dev/en/latest/installation.html).

## Reproduce the experiments

```bash
python run_paper.py
python paper_metrics.py
python paper_figures.py
```

The first command trains DK with forward, backward, and averaged KL and trains the plain-MLP EDM baseline in both dimensions. It also repeats averaged-KL DK and EDM in 8D with seeds 1, 2, and 3, for a total of 14 runs. Each run uses 5,000 updates, 1,024 training and validation points, and 10,000 test points and generated test samples. Every method uses the same data splits in a given dimension and selects its checkpoint by the lowest validation averaged KL with bandwidth 0.03.

The outputs go to `results/paper/`. The metrics script reports both KL directions, averaged KL, off-mode fractions and distances, mode-probability errors, the target-versus-target reference, and the 8D statistics across four seeds. The figure script writes five PDFs to `results/paper/figures/`. In 8D, the sample and field plots display the first two coordinates before the target rotation.

Use `--dry-run` to print the training commands or `--smoke` to check the training and evaluation pipeline with one update per seed-0 run. To launch just one experiment, for example:

```bash
python run_paper.py --dim 8 --method dk --objective averaged --seed 0
```

The package uses the training code saved with the paper runs. It generates fresh checkpoints and samples. Floating-point differences across devices and library versions can change the selected update and the final metrics.

## Timing

The metrics report includes the wall time saved by each training program. To repeat the network-only timing comparison, install CUDA-enabled PyTorch in a compatible environment and make one NVIDIA GPU visible:

```bash
python paper_timing.py
```

This benchmark times the two DK networks on parallel CUDA streams and the EDM network on the same GPU. It uses the normalized training batch of 1,024 points and the best validation-KL checkpoints. Timings depend on the hardware and software.

## License

Angxiu Ni provides this code under the [Attribution Non-Commercial Share Alike License](LICENSE). The license permits eligible academic and nonprofit researchers to use and share the code for educational or academic research under its stated terms.

"""Measure the paper's network-only DK and EDM forwards on one NVIDIA GPU.

Requires NumPy and CUDA-enabled PyTorch with CUDA graph support. The paper
used an NVIDIA L40S and PyTorch 2.6.0 built for CUDA 12.4. Results depend on the
GPU, software versions, and device load. Run on a GPU reserved for this job.

Use the normalized training batch and best-validation averaged-KL checkpoints
from the seed-0 paper runs. Capture one float32 forward per graph, with TF32
disabled. The DK drift and diffusion run on two side streams and join before
the replay finishes. Each reported time is the median of 31 CUDA-event round
means with 200 replays per round. EDM input features are prepared before timing.
The timed region contains only the original affine layers and SiLU activations.
"""

import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np


BATCH_SIZE = 1024
WARMUP = 50
ROUNDS = 31
REPLAYS = 200


def checkpoint(run, dim):
    with np.load(run / "data.npz", allow_pickle=False) as saved:
        train = np.asarray(saved["train"], dtype=np.float32)
    if train.shape != (BATCH_SIZE, dim):
        raise ValueError(f"{run}: expected {BATCH_SIZE} normalized training points in {dim}D")
    return train, np.load(run / "best_kl.npz", allow_pickle=False)


def mlp_reference(layers, inputs):
    for weights, bias in layers[:-1]:
        inputs = inputs @ weights.T + bias
        inputs /= 1 + np.exp(-np.clip(inputs, -80, 80))
    weights, bias = layers[-1]
    return inputs @ weights.T + bias


def capture(torch, forwards, *, parallel=False):
    capture_stream = torch.cuda.Stream()
    side_streams = tuple(torch.cuda.Stream() for _ in forwards) if parallel else ()

    def execute():
        if not parallel:
            return tuple(forward() for forward in forwards)
        results = []
        for stream, forward in zip(side_streams, forwards, strict=True):
            stream.wait_stream(capture_stream)
            with torch.cuda.stream(stream):
                results.append(forward())
        for stream in side_streams:
            capture_stream.wait_stream(stream)
        return tuple(results)

    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(WARMUP):
            execute()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        outputs = execute()
    graph.replay()
    torch.cuda.synchronize()
    return graph, outputs, (capture_stream, *side_streams)


def compare_dimension(torch, results, dim):
    dk_run = results / f"dk-{dim}d-averaged-seed0"
    edm_run = results / f"edm-{dim}d-seed0"
    train, dk_saved = checkpoint(dk_run, dim)
    with dk_saved:
        dk_cfg = json.loads(dk_saved["metadata_json"].tobytes())["config"]
        drift, diffusion = (
            tuple((dk_saved[f"{field}.{i}.w"], dk_saved[f"{field}.{i}.b"])
                  for i in range(dk_cfg[depth] + 1))
            for field, depth in (("drift", "depth"), ("diffusion", "diffusion_depth"))
        )
    edm_train, edm_saved = checkpoint(edm_run, dim)
    with edm_saved:
        edm_cfg = json.loads(edm_saved["metadata_json"].tobytes())["config"]
        if edm_cfg["backbone"] != "mlp":
            raise ValueError(f"{edm_run}: expected the plain MLP backbone")
        names = [f"hidden_{i}" for i in range(edm_cfg["depth"])] + ["output"]
        edm = tuple((edm_saved[f"ema.{name}_w"], edm_saved[f"ema.{name}_b"])
                    for name in names)
    if dk_cfg["dtype"] != "float32" or edm_cfg["dtype"] != "float32":
        raise ValueError("The comparison requires float32 checkpoints")
    np.testing.assert_allclose(train, edm_train, rtol=1e-6, atol=1e-7)

    def on_gpu(array):
        return torch.as_tensor(array, dtype=torch.float32, device="cuda")

    def make_forward(layers, inputs, *, scalar=False):
        weights_and_biases = [(on_gpu(w), on_gpu(b)) for w, b in layers]

        def forward():
            hidden = inputs
            for w, b in weights_and_biases[:-1]:
                hidden = torch.nn.functional.silu(torch.nn.functional.linear(hidden, w, b))
            output = torch.nn.functional.linear(hidden, *weights_and_biases[-1])
            return output[:, 0] if scalar else output

        return forward

    x = on_gpu(train)
    # For EDM noise level 1, log(noise)/4 is zero, so the embedding is
    # exactly zeros followed by ones. Conditioning and scaling precede timing.
    half = edm_cfg["time_features"] // 2
    noise = on_gpu(1.0)
    preconditioned = torch.rsqrt(noise**2 + edm_cfg["sigma_data"]**2) * x * edm_cfg["data_scale"]
    embedding = torch.cat((x.new_zeros(half), x.new_ones(half)))
    features = torch.cat((preconditioned, embedding.expand(BATCH_SIZE, -1)), dim=-1)

    # Keep these closures alive because the graphs use their weight buffers.
    dk_forwards = (make_forward(drift, x), make_forward(diffusion, x, scalar=True))
    edm_forward = make_forward(edm, features)
    captures = {
        "dk_parallel": capture(torch, dk_forwards, parallel=True),
        "edm": capture(torch, (edm_forward,)),
    }
    references = {
        "dk_parallel": (mlp_reference(drift, train), mlp_reference(diffusion, train)[:, 0]),
        "edm": (mlp_reference(edm, features.cpu().numpy()),),
    }
    for name, (graph, outputs, _) in captures.items():
        for output, expected in zip(outputs, references[name], strict=True):
            np.testing.assert_allclose(output.cpu().numpy(), expected, rtol=1e-4, atol=3e-5)
        for _ in range(WARMUP):
            graph.replay()
    torch.cuda.synchronize()

    times = {name: [] for name in captures}
    for index in range(ROUNDS):
        order = list(captures) if index % 2 == 0 else reversed(captures)
        for name in order:
            graph = captures[name][0]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(REPLAYS):
                graph.replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end) * 1000 / REPLAYS)
    return {name: median(rounds) for name, rounds in times.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/paper"))
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("CUDA-enabled PyTorch and exactly one visible NVIDIA GPU are required")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        timings = {f"{dim}d": compare_dimension(torch, args.results, dim) for dim in (2, 8)}
    print(json.dumps({
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "batch_size": BATCH_SIZE,
        "median_us_per_forward": timings,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

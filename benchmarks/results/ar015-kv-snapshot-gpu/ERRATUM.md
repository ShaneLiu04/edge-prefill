# ERRATUM — these CSVs were measured on CPU, not on the GPU

**Status: corrected by AR016.** The CSVs in this directory were produced with
`--n-gpu-layers 99` on the command line, which the `bench` command silently
ignored at the time: only the `n_gpu_layers` **config key** reached the runner
(`src/main.cpp`, bench path). llama.cpp therefore reported

```
load_tensors: offloaded 0/29 layers to GPU
```

and the whole "GPU" arm ran on the CPU. The numbers are still real measurements,
they are just **not GPU measurements**.

What is affected:

| Claim in the AR015 report | Reality |
|---|---|
| "GPU cold TTFT len4k 3688 ms, 0.031x of cold" | CPU measurement (16 threads, no offload) |
| "GPU cold TTFT multifact8k 2671 ms, 0.035x" | CPU measurement (16 threads, no offload) |
| "identical quality on GPU" | true, but measured on CPU |
| "GPU warm TTFT 116 ms / 95 ms" | **valid**: restoring a snapshot is disk + H2D and does not depend on where the prefill ran |

Root cause (fixed in AR016):

1. `BenchmarkConfig` had no `n_threads` / `n_ubatch` plumbing and
   `src/main.cpp` only copied `cfg.n_gpu_layers` (the config-file value) into the
   benchmark, so the CLI flag never got there;
2. nothing verified the placement after load, so "0 layers offloaded" was
   indistinguishable from success.

What AR016 changed:

- CLI `--n-gpu-layers` / `--threads` now reach the benchmark (CLI > conf);
- after load the runner compares the **requested** layer count with the count
  llama.cpp reports and fails with `ErrorCode::Unavailable` unless
  `allow_cpu_fallback = 1`; the audited values are written to every raw CSV row
  (`device`, `gpu_layers_requested`, `gpu_layers_offloaded`, `gpu_layers_total`,
  `device_state`).

How to reproduce the correction:

```bash
# prove the offload happens before trusting any timing
./build-cuda/edge-prefill probe --model M --prompt-file CTX --threads 4 --ngl 99 \
    --out /tmp/probe_gpu
cat /tmp/probe_gpu/probe_device.txt     # requested_layers=99 offloaded_layers=29
cat /tmp/probe_gpu/plan.conf            # cost model actually measured on the GPU
```

Real GPU measurements (Quadro RTX 5000, 29/29 layers, `nvidia-smi` reports
P8 / 645 MHz because WSL cannot lock clocks):

| measurement | CPU (12 threads) | GPU | ratio |
|---|---:|---:|---:|
| prefill slope `b` (probe, ms/token) | 3.46 | 2.60 | 1.33x |
| dense 4K prefill (3962 tokens) | 35.0 s | 24.0 s | 1.4x |

The GPU on this host is clock-limited (645 MHz of a 2100 MHz maximum, 31 W of a
230 W limit, `nvidia-smi -lgc` refused). AR016 records this honestly instead of
implying a large GPU win that the hardware cannot deliver here.

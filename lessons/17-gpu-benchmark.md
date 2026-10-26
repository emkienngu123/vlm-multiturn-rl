# 17 — Measure GPU and activation offloading costs

The training job's speed depends on more than GPU FLOPs. Each multi-turn
episode grows the image history; exact cached replay builds autograd graphs,
and CPU offloading transfers saved activations out and back during backward.

Run a paired, disposable benchmark:

```bash
.venv/bin/modal run modal_app.py --task browser-gpu-benchmark \
  --run-id stage16-20261019T153118Z-940cb473
.venv/bin/python -m unittest discover -s tests -q
```

This requires the completed source group and its SFT adapter locally and on
the existing Modal volume. It runs one L40S job then one H100 job, with a
30-minute timeout per GPU job, two requested host CPU cores and 64 GiB host
memory. `H100!` prevents automatic H200 substitution during the benchmark.
The model and dependencies remain pinned to the earlier lessons.

If the L40S artifacts are already complete locally, continue with H100 only:

```bash
.venv/bin/modal run modal_app.py --task browser-gpu-benchmark \
  --run-id stage16-20261019T153118Z-940cb473 \
  --resume-run-id L40S_BENCHMARK_RUN_ID
```

The controller verifies the saved manifest before reusing a result. A failure
to deserialize a return does not necessarily mean remote computation failed:
check committed artifacts first. In our first run, `torch.__version__` was a
`TorchVersion` string subclass. Returning it directly caused Modal unpickling
to import Torch locally, despite the value appearing to be an ordinary string.
The remote boundary now normalizes the entire response through JSON so only
plain JSON values reach the CPU-only local controller.

Both jobs process the first Stage 16 mixed group: four trajectories, 43
actions, the same saved screenshots and exact assistant token IDs, returns
`[0, 0, 0, 1]`, and the original early SFT adapter. Each GPU runs two updates
from identical initial adapter parameters and fresh Adam state: the existing
CPU offload, then a bounded GPU retention policy (16 GiB on L40S, 40 GiB on
H100). These are disposable timing updates; no checkpoint is promoted and
the final test remains closed.

Each large nonparameter saved tensor either stays on GPU or is copied to CPU
exactly. The cap counts saved references conservatively; it does not include
parameters, small tensors, temporary forward tensors, or backward workspace.
The allocator must also report at least 12 GiB plus the saved tensor's size
free before retaining it. A budget is not an OOM guarantee. If budgeted mode
runs out of memory, its failure is recorded alongside the completed baseline.

We check numerical gradients on a small real CUDA graph, validate fresh
sampling versus replay on short and long histories, and require the complete
clipped LoRA gradient vectors and updated adapter fingerprints to agree
between offload modes on each GPU. The objective still scores assistant
tokens only and uses the Stage 12 per-trajectory normalization and strict
unchanged-policy ratios.

Across GPU architectures, BF16 kernels may yield different scores. The
benchmark records that historical score difference and measures a fresh
fixed-token behavior baseline on each device before its two timed updates.
That rebasing is valid for isolating timing, **not** for treating archived
trajectories as newly sampled on-policy RL training.

The report separates model load, browser action reexecution, fixed-token
generation replay, scoring forward, backward, and total update time. Forced
replay is a generation proxy: it does not measure fresh live rollout success
or the full cost of collecting new trajectories. Forward time includes
activation packing; backward includes unpacking. Transfer byte totals count
saved references, not unique storage, and do not measure transfer time.

For GPU rate \(c\) and update duration \(t\), estimated GPU-only update cost is
\(ct\). Costs exclude host CPU/memory, image startup, idle shutdown, and
storage; they are not billing receipts. See the current
[Modal prices](https://modal.com/pricing) and
[GPU options](https://modal.com/docs/guide/gpu).

One group per configuration gives preliminary timings. CPU mode runs first,
so warmup and order effects can affect comparisons. Use repeated alternating
measurements for a firm purchasing decision. Faster updates also do not
solve sparse exploration: equal-return groups still have zero advantages.

## Measured result, 2026-10-19

The [completed comparison](../artifacts/stage17-20261019T164758Z-5471fe18-comparison.html)
uses L40S run `stage17-20261019T163512Z-12d94d3c` and H100 run
`stage17-20261019T164758Z-c0c0f12d`. Both processed 43 actions and 1,015
assistant tokens from the identical saved group.

| GPU and saved-activation policy | Update seconds | Forward / backward seconds | Peak allocated / reserved GiB | GPU-only update cost |
| --- | ---: | ---: | ---: | ---: |
| L40S, CPU offload | 264.2 | 143.3 / 120.7 | 7.5 / 9.1 | $0.1432 |
| L40S, 16 GiB retention budget | 252.1 | 136.1 / 115.9 | 17.5 / 32.9 | $0.1367 |
| H100, CPU offload | 251.8 | 143.4 / 108.3 | 7.6 / 9.1 | $0.2763 |
| H100, 40 GiB retention budget | 114.6 | 65.4 / 49.1 | 22.9 / 56.7 | $0.1257 |

The GPU swap with the existing offload policy improved update speed only
1.05×. The best tested H100 retention configuration was 2.31× faster than
the original L40S configuration and 2.20× faster than budgeted L40S, with
12.2% lower GPU-only update cost than the original. The complete two-device,
four-update benchmark consumed about $0.84 in estimated GPU time, excluding
host resources, startup, storage and shutdown.

Both CPU-offload runs copied 405.6 GB cumulatively to CPU. L40S retention
still offloaded 347.9 GB; H100 retention offloaded zero. This supports
reducing transfers, rather than assuming a GPU swap alone solves the cost.
These byte totals count repeated saved references. The conservative retention
cap and free-memory guard interact with the caching allocator: allocated
memory is not the same as reserved memory, and these peaks do not establish
the minimum GPU capacity the workload could use after further optimization.

Fresh sampling/scoring checks passed with zero ratio drift on both one-image
and twelve-image histories. The small CUDA gradient check and the full clipped
LoRA gradient comparison had maximum delta zero. Updated adapters were
identical between offload modes **within each device**. H100's historical-token
scores differed from L40S by up to 0.606 log-probability units, so archived
L40S behavior probabilities must not be reused blindly for H100 training.
Collect fresh trajectories after switching devices.

All 104 CPU tests passed, source manifests and image/token receipts matched,
and all timed Python modules were byte-identical across the devices. Only
the Modal entrypoint changed to normalize the response and reuse the recovered
L40S result. The benchmark updates were discarded; Stage 16's learning
checkpoint and two-step optimizer counter remain unchanged. No final-test
cases were opened. Live rollout and evaluation performance still need to be
measured for a subsequent H100 training run.

Exercise: use the reported forward/backward times to estimate the maximum
possible overall speedup if backward became twice as fast while rollout and
browser work stayed constant. Explain why that differs from a 2× job speedup.

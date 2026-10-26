# Multi-turn VLM Reinforcement Learning

A research and engineering project for training vision-language agents to act from images across multiple environment steps. The system combines **Qwen2.5-VL-3B**, **LoRA supervised fine-tuning**, and **Group Relative Policy Optimization (GRPO)** in visual grid worlds and screenshot-driven browser tasks.

The implementation covers the complete experiment pipeline: environment design, demonstration collection, assistant-token supervision, online grouped rollouts, policy-gradient updates, checkpoint recovery, and paired evaluation. GPU workloads run on Modal using NVIDIA L40S and H100 GPUs; local configuration, replay, and contract checks run without PyTorch.

## Results

| Experiment | Recorded result | Scope |
| --- | --- | --- |
| Form and recovery SFT | Development success **2/16 → 16/16** after 214 optimizer updates on 853 action targets | One pass over 100 clean and 20 synthetic recovery demonstrations |
| GRPO from an early SFT checkpoint | Development success **1/16 → 3/16** | Four sampled groups, two optimizer updates, two zero-variance skips |
| H100 GRPO continuation | Development success **3/16 → 4/16** | Seven fresh groups, five additional updates, 25.7 minutes of recorded training work |
| GPU and activation-memory benchmark | Fixed-group update time **264.2 s → 114.6 s** (**2.31×** speedup) | L40S CPU offload versus H100 with a 40 GiB activation-retention budget |

These are small experiments on development instances of a familiar form template. Full SFT remains the strongest recorded reference. The 32 reserved final-test instances remain unopened, so the results do not establish final-test performance or general web-navigation ability. The GPU benchmark measures one recorded group per configuration; gradient vectors and adapter updates matched between memory modes within each device.

Experiment configurations, measurements, and analysis are documented in the notes on [form SFT](lessons/14-form-sft.md), [form GRPO](lessons/16-form-grpo-learning.md), [GPU benchmarking](lessons/17-gpu-benchmark.md), and [H100 continuation](lessons/18-h100-grpo-continuation.md). Raw run bundles and model weights are excluded from this repository; artifact links in the experiment notes refer to locally generated files.

## Core capabilities

- **Visual environments:** deterministic grid-world dynamics, solver baselines, and MiniWoB++/BrowserGym tasks with screenshot observations and coordinate actions. Privileged browser state is restricted to demonstration generation and validation.
- **Multi-turn histories:** ordered image observations, tool feedback, exact generated token IDs, and explicit assistant-only loss masks.
- **LoRA SFT:** language-attention adapters with frozen base and vision weights, demonstration validation, recovery examples, and adapter save/reload checks.
- **Explicit GRPO:** four same-state trajectories per group, episode-relative advantages, a clipped assistant-token objective, and sampling/replay probability checks. Equal-return groups skip the optimizer and are counted separately.
- **Resumable training:** adapter weights, Adam state, parameter ordering, RNG state, schedule position, and source fingerprints persisted across GPU jobs.
- **Evaluation and diagnostics:** paired task instances, separated training/development/final-test pools, replayable trajectories, grounding audits, uncertainty intervals, and learning curves.
- **GPU memory optimization:** exact saved-activation CPU offloading and bounded GPU retention, checked against full gradients and adapter updates.
- **TRL integration:** a separately pinned trainer path with changing-image tool responses and native mask, objective, and gradient audits.

## System design

```text
Grid world / browser
        │ image + instruction + public feedback
        ▼
Multi-turn history → Qwen2.5-VL + language LoRA → structured action
        ▲                                            │
        └──────────── environment transition ─────────┘

Solver / browser demonstrations → assistant-only SFT
Same-state rollout groups       → relative rewards → GRPO
Training state                  → durable checkpoints → paired evaluation
```

GRPO uses a token-level clipped surrogate averaged within each trajectory, with one episode advantage broadcast across its assistant actions. Browser learning experiments use sparse binary task success and no reference-policy KL term. Strict replay checks verify that the unchanged policy reproduces behavior probabilities before an update. New trajectories are collected after switching GPU architectures.

## Quick start

Use Python 3.12+ and `uv`. The local dependency lock installs the Modal client; PyTorch, browser packages, and model execution live in the remote images.

```bash
git clone https://github.com/emkienngu123/vlm-multiturn-rl.git
cd vlm-multiturn-rl
uv sync --locked

# Configuration and CPU tests
uv run vlm-multiturn-rl --seed 42
uv run python -m unittest discover -s tests -v

# Generate and verify random/oracle grid-world episodes locally
uv run vlm-multiturn-rl --stage 3 --episodes 4

# Inspect the GRPO objective with synthetic checks
uv run vlm-multiturn-rl --stage 5
```

The grid-world command writes a self-contained `replay.html` under `artifacts/RUN_ID/`. Replay a saved run without inference:

```bash
uv run vlm-multiturn-rl --stage 3 --replay artifacts/RUN_ID
```

## GPU experiments

Authenticate with `uv run modal setup` before launching remote jobs. These commands allocate remote compute; the first run also builds the pinned image and caches model files.

### Grid-world training

```bash
# Check the remote GPU and inspect model inputs
uv run modal run modal_app.py --task check-gpu
uv run modal run modal_app.py --task inspect-vlm

# Evaluate the base model, then train and evaluate a LoRA adapter
uv run modal run modal_app.py --task evaluate-grid --episodes 4
uv run modal run modal_app.py --task sft-grid --updates 2 --episodes 4

# Replace GRID_SFT_RUN_ID with the ID printed by the SFT job
uv run modal run modal_app.py --task grpo-grid --updates 2 --episodes 4 \
  --sft-run-id GRID_SFT_RUN_ID
```

The two-update defaults verify the training path. Longer experiments use the checkpointed `train-grid` workflow, including cross-container continuation checks; see the [training and evaluation notes](lessons/06-training-evaluation.md).

### Browser training

```bash
# Validate demonstrations in real browsers on remote CPUs
uv run modal run modal_app.py --task browser-check --episodes 2

# Use the printed CPU preflight ID for both jobs
uv run modal run modal_app.py --task browser-baseline --episodes 2 \
  --run-id BROWSER_CPU_RUN_ID
uv run modal run modal_app.py --task browser-sft --episodes 2 --updates 100 \
  --run-id BROWSER_CPU_RUN_ID
```

The browser stack includes Chromium and checksum-verified MiniWoB++ assets. Initial tasks cover clicking, text entry, and login; later experiments introduce shuffled forms, recovery actions, tab search, and popup interruptions. Form SFT and GRPO require the matching source adapters, dataset receipts, and preflight bundles described in their experiment notes. A fresh clone must generate those prerequisites before running the later workflows.

## Reproducibility

- **Pinned inputs:** immutable model revision, resolved Linux dependency files, and source hashes recorded with each run.
- **Durable storage:** Modal volumes `vlm-multiturn-rl-cache` and `vlm-multiturn-rl-runs` hold model caches and experiment state; local copies are checksum-verified.
- **Verifiable recovery:** checkpoints preserve optimizer/RNG continuity and are validated by reloading and comparing policy scores.
- **Inspectability:** saved runs include configurations, observations, actions, token masks, policy scores, metrics, and replay reports as appropriate to each workflow.

Generated artifacts, datasets, downloaded weights, adapters, checkpoints, logs, local credentials, and rendered reports are ignored by Git. The repository retains source code, tests, dependency locks, experiment notes, and report-building tools.

## Repository layout

| Path | Purpose |
| --- | --- |
| `modal_app.py` | Remote images, resources, volumes, and experiment orchestration |
| `src/vlm_multiturn_rl/gridworld.py` | Visual grid environment and solver |
| `src/vlm_multiturn_rl/browser.py` | Browser task contracts, demonstrations, and action execution |
| `src/vlm_multiturn_rl/trajectories.py` | Multi-turn histories and assistant-token masks |
| `src/vlm_multiturn_rl/grpo.py` | Group advantages, token scoring, and clipped objective |
| `src/vlm_multiturn_rl/browser_policy.py` | Shared browser sampling, differentiable replay, and activation handling |
| `src/vlm_multiturn_rl/checkpoints.py` | Training-state persistence and restore validation |
| `src/vlm_multiturn_rl/stage*.py` | Numbered experiment workflows, from workspace checks to H100 continuation |
| `src/vlm_multiturn_rl/gpu_benchmark.py` | GPU and activation-memory comparison |
| `src/vlm_multiturn_rl/reporting.py` | Paired outcomes, uncertainty, and learning curves |
| `tests/` | CPU contract, masking, objective, replay, and checkpoint tests |
| `requirements-{gpu,browser,trl}.{in,txt}` | Direct and resolved remote dependency pins |
| `scripts/` | Benchmark asset installation |
| `lessons/` | Detailed experiment notes, commands, and analysis |
| `reports/source/`, `reports/tools/` | Report source and build/render tools |

**Stack:** Python · PyTorch · Transformers · PEFT/LoRA · TRL · BrowserGym · MiniWoB++ · Playwright · Modal · NVIDIA L40S/H100

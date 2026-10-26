# Stage 10 — Fresh evaluation and GRPO exploration readiness

The 100-update SFT adapter solved six training and six evaluation instances.
Before browser GRPO, we need two additional measurements: how it behaves on
more unseen instances, and whether sampled training groups have a relative
reward signal. This lesson collects both **without optimizer updates**.

**Execution status:** Completed and GPU-validated as
`stage10-20261019T064016Z-fe766db8`, reusing CPU preflight
`stage10-20261019T061542Z-a5a2959d`. All 71 CPU tests and all 78 saved-episode
checks pass. Prefix behavior scores agree exactly with sampling. No optimizer
updates were performed. The measured findings appear below.

## Run the lesson

```bash
# Local source/split configuration checks; no Torch, Pillow, or browser needed.
uv run vlm-multiturn-rl --stage 10 --episodes 2 \
  --sft-run-id stage9-20261019T041931Z-88adc041
uv run python -m unittest discover -s tests -v

# Real-browser CPU preflight, then frozen-policy evaluation on one L40S.
uv run modal run modal_app.py --task browser-readiness --episodes 2 \
  --fresh-cases 8 --probe-cases 1 \
  --sft-run-id stage9-20261019T041931Z-88adc041

# Reuse the completed CPU preflight when retrying the GPU phase.
uv run modal run modal_app.py --task browser-readiness --episodes 2 \
  --fresh-cases 8 --probe-cases 1 \
  --sft-run-id stage9-20261019T041931Z-88adc041 \
  --run-id stage10-20261019T061542Z-a5a2959d

# Offline verification after completion; this does not regenerate model outputs.
uv run vlm-multiturn-rl --stage 10 --replay artifacts/STAGE10_RUN_ID
```

Use your own completed Stage 9 browser SFT ID if these artifacts are not present.
`--episodes 2` must match the source adapter's browser configuration. It does not
set the fresh evaluation size: `--fresh-cases` controls instances per task.
`--probe-cases` controls instances per task in each training pool. The command
can select the latest completed local browser SFT report if `--sft-run-id` is
omitted; selection is by completion/run order, not evaluation score.

The command saves a CPU preflight run before requesting a GPU, and then a
separate GPU report. Expect `Readiness oracle passed ...`, `Fresh eval ...`, and
`Probe ... member .../4` progress messages. The CPU job has a 15-minute timeout;
the GPU job has a 30-minute timeout. Both use the existing pinned browser image.
When `--run-id` identifies a completed CPU preflight, its artifacts, source
adapter, task configuration, dependency lock, and protocol are verified instead
of re-running oracle episodes. A new GPU run ID is always created.

## 1. Define three separate instance pools

The first three MiniWoB task templates remain `click-test`, `enter-text`, and
`login-user`. We broaden instance seeds, not the task vocabulary.

| Pool | Seeds per task by default | Size | Purpose |
| --- | --- | --- | --- |
| Seen training | 0 | 3 cases | Check whether SFT training cases have saturated |
| New training | 1000 | 3 cases | Probe new instances reserved for future RL training |
| Fresh evaluation | 20000–20007 | 24 cases | Greedy evaluation, excluded from training groups |

The earlier SFT dataset used training seeds 0 and 1 and evaluation seeds 10000
and 10001. Neither new pool reuses those instances. We check identities as
`(task, seed)` pairs, because the same integer seed across different task types
does not imply the same task instance. Public observations still contain only
instructions, screenshots, and action feedback; pool labels and seeds remain
bookkeeping.

The privileged CPU oracle must solve all 30 selected cases, and its exact
actions must re-execute in fresh browsers. These oracle outcomes validate the
environment. Oracle coordinates/targets are not supplied to the model.

## 2. Measure generalization separately from exploration

The adapter uses the same greedy decoding as Stage 9 on all 24 fresh evaluation
instances. We report per-task success, action validity, execution errors, episode
length, and an illustrative Wilson interval for success. These results concern
new instances of three familiar templates; they do not establish transfer to
unseen websites or task types. Dependence within a template also limits a simple
binomial interval's interpretation.

Evaluation rewards never enter the exploration advantages. Improving an
evaluation score is not itself a reason to include that instance in training.

For each training case, collect four independent complete trajectories. Every
member starts from the same browser task/seed, public goal, initial screenshot,
and action cap. A fresh browser is created for each member. Distinct deterministic
Torch/CUDA RNG streams are set once per trajectory, not at each turn.

Default totals are **24 greedy model episodes and 24 sampled model episodes**.
The 30 oracle episodes are retained too, giving 78 saved episodes if the full
run completes. More samples may be needed before estimating exploration success
reliably; four trajectories are a first readiness check.

## 3. Why reward variance determines the next step

For four same-case trajectory returns, compute

\[
A_i=\frac{R_i-\overline R}{\operatorname{std}_{\mathrm{population}}(R)+10^{-6}}.
\]

Our reward remains sparse and binary: success yields an episode return of one,
failure zero. There is no action cost or distance shaping in this browser track.
Different successful path lengths therefore still receive the same return.

```text
returns [1, 1, 1, 1] -> all advantages zero: saturated group
returns [0, 0, 0, 0] -> all advantages zero: no successful exploration
returns [1, 0, 1, 0] -> advantages approximately [+1, -1, +1, -1]
```

Variation across different cases cannot replace variation within a same-case
group. Mixing successful login episodes with failed button episodes would
confound task difficulty with policy quality. `readiness.json` counts mixed,
all-success, and all-failure groups separately in both training pools.

## 4. Keep sampling and scoring consistent

`browser_policy.BrowserPrefixPolicy` reuses the loader and history interface
from `stage5.GRPOPolicy`, with an explicit shared prefix loop. Sampling uses
temperature 1, `top_p=1`, `top_k=0`, no repetition penalty, and no KV cache. These
settings make the sampling distribution the ordinary policy softmax used by
the scorer. Changing temperature or applying a top-k/top-p filter would require
accounting for that transformed distribution when scoring.

For each generated token, both sampling and scoring call the model on the same
prompt plus **only the previously generated tokens**, requesting
`logits_to_keep=1` and `use_cache=False`. Sampling draws from that last-position
softmax; scoring selects the recorded next ID from the same computation. The
loop includes EOS in scored tokens and also supports actions cut off by the
generation limit.

This replaces the initial comparison between one-token-at-a-time generation
and one full-sequence teacher-forced scoring pass. Mathematically those
conditionals should agree, but floating-point implementations are not guaranteed
to give identical values when computation shapes differ. See
[PyTorch's numerical-accuracy discussion](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html#batched-computations-or-slice-computations).
Qwen supports projecting just the final position via `logits_to_keep`; see the
[official Qwen2.5-VL forward interface](https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl).

The observed 0.430226 mismatch motivates matching both paths; it does not by
itself prove which kernel, precision choice, or input-preparation difference
caused that failed run. We retain the old full-sequence scorer as
`batched_diagnostic_log_probs`, with a separate maximum discrepancy per group.
Its values are **not substituted for the actual behavior-policy probabilities**.
The prefix scorer must still pass the unchanged 0.1 behavior consistency check.
This defines a new numeric sampling protocol, `prefix-v1`. Identical seeds need
not reproduce trajectories from the failed `generate()` implementation; the
rerun collects every sampled group afresh and reuses only the CPU preflight.

A log-probability discrepancy of 0.430226 corresponds to a probability ratio of
about `exp(0.430226) = 1.54`. Before any policy update, a consistent behavior/new
policy comparison should give ratio one. An artificial ratio this large could
already activate a 0.2 clipping boundary. This is why accepting a wider
tolerance would undermine the later GRPO experiment. Its gradient scorer must
use the prefix-consistent computation or an independently validated faster path.

Before full browser evaluation, the GPU runs a smoke check on training-only
`enter-text` histories with one, two, and three images. It reports prefix and
batched discrepancies separately. A failure stops the job before the 24 fresh
evaluation episodes. `environment.json` records these checks and the scoring
protocol `prefix-v1`.

Each turn preserves exact generated token IDs, EOS/length termination, prompt
IDs, ordered image paths/grids, and sampled token log probabilities. Scoring
reconstructs the same input history and appends the **actual generated IDs**;
it does not re-tokenize decoded output. The predicting logit for the first
action token at position P is at P−1.

Only the generated assistant continuation is selected for token log
probabilities. The saved `action_loss_span=[P,P+L)` identifies this span; prompts,
images, prior actions, and tool feedback are outside it. No loss is backpropagated
in this lesson. Both generation and scoring run without gradients.

Freshly scored behavior log probabilities must agree with sampling scores to
within the reference implementation's fixed maximum absolute tolerance of 0.1.
BF16 evaluation can introduce numerical differences. A larger mismatch
fails the check; we do not silently widen the tolerance or proceed to training.

Adapter parameter fingerprints must remain identical throughout collection.
The base revision is immutable and source artifact hashes are checked before
loading. Every sampled and greedy action sequence is re-executed in a fresh
browser without model generation to verify recorded outcomes.

CPU tests also build a synthetic 78-episode report, verify it through the saved
artifact reader, and ensure corrupted advantages are rejected even when file
checksums are recomputed. This validates the report contract without pretending
to exercise real model sampling or browser dynamics locally.

## 5. Read the code and artifacts

`stage10.py` defines `BrowserReadinessConfig`, `build_cases`, `source_receipt`,
`group_record`, `readiness_report`, and the CPU/GPU execution paths. Reusable
environment behavior stays in `browser.py`; browser prefix sampling/scoring is
in `browser_policy.py`. The earlier grid policy in `stage5.py` is unchanged.

| Artifact | What to inspect |
| --- | --- |
| `source-adapter.json`, `cpu-source.json` | SFT adapter and preflight checksums |
| `cases.json`, `pools.json` | Exact task seeds and pool assignments |
| `episodes.json`, `images.zip`, `report.html` | Oracle, greedy, and sampled histories/actions/rewards |
| `groups.json` | Same-case members, RNG streams, returns, advantages, fingerprints, scoring drift |
| `readiness.json` | Fresh evaluation and training-only reward-variance summaries |
| `metrics.json` | Evaluation instances only; oracle and SFT reported separately |
| `environment.json` | Exact decoding settings, GPU, memory/runtime, zero optimizer steps |
| `manifest.json` | Source/dependency and artifact checksums |

The remote run also writes incomplete progress files after evaluation and each
group, and on caught exceptions: `progress.json`, `partial-episodes.json`,
`partial-groups.json`, and `partial-images.zip`. These preserve diagnostics even
when final group validation fails. The Modal wrapper closes Playwright and
commits the volume in `finally`. Incomplete progress has no completion manifest
and cannot be mistaken for a finished run. A completed run retains its normal
checksum-verified artifact set.

Prefix re-scoring is slower than one full-sequence pass. It is the correctness
path for this small experiment. A future gradient update must handle its memory
cost explicitly rather than retaining a full graph for every prefix at once;
this readiness job uses no-grad scoring and performs no optimization.
Before a subsequent GRPO update, also compare behavior scores against the actual
gradient-enabled scoring path at unchanged weights. This run validates no-grad
re-scoring; it does not establish that every future autograd/kernel configuration
will produce identical probabilities.

The original adapter is unchanged. This lesson writes no new model checkpoint,
optimizer state, or training demonstrations. The HTML report includes a readiness
section and all episode replays. Offline replay verifies contracts and saved
records; it does not open Chromium or re-sample the model.

## 6. Decide what to train next

### Measured result after fixing sampling/scoring

| Measurement | Result |
| --- | --- |
| Fresh greedy evaluation | **24/24 successes**: 8/8 per task |
| Seen-training sampled episodes | **12/12 successes** |
| New-training sampled episodes | **12/12 successes** |
| Invalid actions / execution errors | **0 / 0** |
| Mixed-return groups | **0/6** |
| Maximum sampling/prefix-scoring discrepancy | **0.0** |
| Maximum old batched-scoring discrepancy | **0.430226** |
| Optimizer updates | **0** |

The old batched diagnostic reproduced **0.43022632598876953** on
`enter-text-rltrain-000`, the same task implicated in the original failure.
The prefix scorer gave exactly the sampled token log probabilities for that
group and every other group. This validates the forward-path fix without
raising the 0.1 tolerance. The one-/two-/three-image smoke checks also all gave
prefix discrepancy zero; their batched discrepancies were approximately
0.188522, 0.0, and 0.091943 respectively.

The completed run retained 30 oracle episodes, 24 greedy evaluation episodes,
and 24 sampled training episodes. Initial-state, image-order, exact-token,
behavior-weight, and live browser re-execution checks all passed. Offline replay
verified all 78 records and recomputed their groups/metrics. The saved source
hashes match the implemented Python code at completion.

The GPU function took **897.1 seconds (15.0 minutes)** and peaked at **7.47 GiB**
allocated CUDA memory on one L40S. Prefix replay is a slower correctness
reference; the original frozen adapter was not changed or saved again.

All six groups returned `[1,1,1,1]`. Every within-group variance and advantage is
therefore zero. Both training pools are saturated under this sparse reward.
**The next experiment needs harder training instances or task templates before
GRPO optimization.** Increasing update count on these same groups would not
produce a reward-driven policy gradient. The 24/24 fresh result concerns three
familiar task types, with the earlier task-template uncertainty caveat.

Open [the completed report](../artifacts/stage10-20261019T064016Z-fe766db8/report.html),
[the group scores](../artifacts/stage10-20261019T064016Z-fe766db8/groups.json), and
[the readiness decision](../artifacts/stage10-20261019T064016Z-fe766db8/readiness.json).

| Observed training groups | Next step |
| --- | --- |
| Some mixed-return groups | Implement a bounded GRPO correctness update on training cases |
| Seen cases all succeed, new training cases are mixed | Use the new training pool for a controlled RL experiment |
| Both training pools all succeed | Add harder training instances/tasks; the current sparse reward is saturated |
| New training cases all fail | Inspect exploration and failed trajectories; consider additional training demonstrations before a long RL run |
| Sampling/scoring mismatch | Fix history, token, distribution, or numerical consistency before optimization |

Fresh evaluation remains measurement throughout these decisions. Use training
group outcomes to select an RL curriculum or exploration settings. A single
mixed group demonstrates a nonzero relative signal, not that GRPO will improve
generalization.

## 7. Frequent issues

| Symptom | Check |
| --- | --- |
| Source rejected before the GPU job | Supply a completed browser Stage 9 run with matching `--episodes`, seed, processor settings, prompt, benchmark, and dependency lock |
| Unexpectedly zero advantages | Inspect each group's returns; all-success is as zero-variance as all-failure |
| Different resets within a group | Compare public instruction and raw initial pixel hashes; each rollout must use the same task seed |
| Independent samples look identical | Inspect distinct trajectory RNG seeds and exact IDs; a sharp policy can legitimately produce the same actions |
| Good across-case success variance but no useful groups | Each case may be consistently easy or consistently impossible; across-case variance is not the GRPO signal |
| Missing/dropped images or stale observations | Check every turn's ordered `image_paths`, grids, and actual prompt IDs before accepting its old log probabilities |
| Capacity error before any command runs | An execution-approval service failure is separate from Modal/GPU capacity; no result exists unless the command actually starts |

## Exercise

Find one all-success, all-failure, or mixed group in `groups.json`. Recompute its
population mean/std and advantages from the four recorded returns. Then inspect
the four replays and explain how model choices changed later screenshots. If
all groups have zero variance, propose a training-only change that could produce
useful exploration without changing the evaluation set or reward definition.

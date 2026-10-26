# Stage 16 — Run GRPO after a short SFT warmup

Stage 15 found one mixed group for the early form adapter and no mixed groups
for the completed SFT adapter. This stage actually optimizes the early adapter,
with fresh browser trajectories, persistent optimizer state, and paired
development measurements. It is a bounded learning experiment, not a claim of
convergence or of superior performance to the completed SFT policy.

Execution status: completed as a bounded experiment. **Four groups produced two
actual GRPO updates and two zero-variance skips. Paired development success
improved from 1/16 to 3/16.** The twenty-round ceiling was not reached: the next
round was projected to exceed the thirty-minute training-work budget.
The 32 final-test cases remain closed.

## Commands

```bash
# Source validation and the complete experiment plan; no local PyTorch.
uv run vlm-multiturn-rl --stage 16 \
  --sft-run-id stage14-20261019T100154Z-0bf33123 \
  --comparison-run-id stage14-20261019T102008Z-1255a893 \
  --readiness-run-id stage15-20261019T144956Z-cc0a6ba8

uv run python -m unittest discover -s tests -v

# Before evaluation, two smoke rounds, smoke evaluation, bounded continuation,
# then evaluation of the last completed saved adapter.
uv run modal run modal_app.py --task browser-form-grpo --updates 20 \
  --sft-run-id stage14-20261019T100154Z-0bf33123 \
  --comparison-run-id stage14-20261019T102008Z-1255a893 \
  --audit-run-id stage15-20261019T144956Z-cc0a6ba8

# Use --updates 2 to request only the smoke part of the same declared study.
# Resume a completed round after an infrastructure interruption, with identical
# core source/configuration and the original CPU preflight:
uv run modal run modal_app.py --task browser-form-grpo --updates 20 \
  --sft-run-id stage14-20261019T100154Z-0bf33123 \
  --comparison-run-id stage14-20261019T102008Z-1255a893 \
  --audit-run-id stage15-20261019T144956Z-cc0a6ba8 \
  --run-id STAGE16_CPU_PREFLIGHT --resume-run-id STAGE16_CHECKPOINT

uv run vlm-multiturn-rl --stage 16 --replay artifacts/STAGE16_RUN_ID
uv run python lessons/tools/summarize_form_grpo.py artifacts/STAGE16_WORKFLOW.json
# Optional CPU-only PNG/SVG export after the verified summary exists.
uv run modal run lessons/tools/plot_form_grpo.py --workflow artifacts/STAGE16_WORKFLOW.json
```

The local reader needs the Stage 9, Stage 13–15, and any parent Stage 16
artifact directories alongside the downloaded bundle. The Modal volume already
contains the inputs from the previous stages. Each round saves its own complete
checkpoint and report under a unique ID. Interrupted partial rounds do not
replace the last completed round. Resume repeats development baseline evaluation
but continues the training cursor and Adam state, rather than starting training
over. Recorded work from parent rounds still counts against the budget.

## 1. Separate initialization, exploration, and evaluation

The initial policy is Stage 14's update-two checkpoint, after eight additional
form targets. It already includes Stage 9's 100 browser-SFT updates. We load its
existing language-side LoRA; no new adapter is initialized. GRPO starts with a
fresh AdamW optimizer because this is a new objective, not continuation of SFT
Adam moments. Later GRPO rounds restore their own Adam moments and counters.

The source validator requires the exact mixed Stage 15 group for this adapter,
the complete-pass SFT comparison adapter, matching pinned model/processor
settings, and their actual ancestor relationship. The mixed probe is evidence
for choosing the starting policy. Its cached episodes are not optimization data.

| Pool | Instances | Use |
| --- | --- | --- |
| Training | Ten form seeds 80000–80009 | Oracle preflight and fresh GRPO collection |
| Development | Sixteen existing seeds 60000–60015 | Greedy before/smoke/final comparisons |
| Final test | Thirty-two reserved seeds 70000–70031 | Remains unopened |

Two training seeds were inspected in Stage 15; eight are new. Every training
state must pass a seven-action oracle and a fresh-browser reexecution. Oracle
coordinates remain privileged demonstration metadata and are not passed to
the policy during its episodes. No new SFT loss is applied in this stage.

The first two rounds both use seed 80000, the state that previously produced
mixed returns. The remaining declared rounds visit seeds 80001–80009 twice.
This small curriculum is chosen from training evidence. It tests learning on
one familiar form template, not general web navigation or memory ability.
Every round uses new sampling streams, starting after all 32 Stage 15 streams.
There is no silent resampling to replace a zero-variance group.

## 2. Optimize complete episodes with an explicit surrogate

For episode i, let n_i count all its generated assistant tokens, including EOS
when sampled. For token k at turn t, the ratio uses its actual observation and
action history, not the final screenshot:

\[
r_{itk}(\theta)=\exp\{\log\pi_\theta(a_{itk}\mid h_{it},a_{it,<k})
 -\log\pi_{\rm behavior}(a_{itk}\mid h_{it},a_{it,<k})\}.
\]

With four same-state episodes and episode-level normalized returns A_i, we use

\[
\mathcal L(\theta)=-\frac14\sum_i\frac1{n_i}\sum_{t,k}
\min\left[r_{itk}A_i,
\operatorname{clip}(r_{itk},1-0.2,1+0.2)A_i\right].
\]

This gives each trajectory equal total weight before its advantage, rather than
allowing a long failing episode to dominate just because it generated more
tokens. Within a trajectory it averages assistant tokens, not turns. It is a
practical token-level clipped surrogate with length normalization; it is not
the exact trajectory-likelihood ratio or an unnormalized REINFORCE estimator.

Observations, instructions, and earlier tool results are context, not loss
targets. Gradients still flow through the model's computation on that context.
Only the recorded generated assistant spans contribute output-token losses.
Malformed and unsupported actions remain actual policy outputs and receive
the same episode advantage; they are not replaced with oracle actions.

The sparse reward stays binary success with no shaping or formatting bonus.
One episode advantage is broadcast across its actions, so correct early moves
in a failed episode can receive negative credit. This coarse assignment is an
explicit limitation, not a turn-specific explanation of causality.

We make exactly one optimizer step for each nonconstant group and then collect
again under the updated policy. Before that step, all token ratios should be
one within the 1e-4 tolerance. Consequently the clipping fraction will normally
be zero in this one-step setup. A scalar loss near zero is also expected when
the normalized advantages cancel; gradients can still be nonzero because the
episodes have different conditional token gradients. Check gradient norms and
parameter changes rather than treating zero loss as proof of no learning.

This experiment uses beta=0, so there is no separate reference-policy KL term.
Small learning rate (1e-5), gradient clipping at one, fresh groups, and a bounded
run limit drift; they are not substitutes for measuring behavioral regression.

## 3. Preserve sampling/scoring consistency through backward

The implementation reuses Stage 12's `BrowserTrainPolicy`, strict token replay,
and clipped objective. The behavior adapter stays fixed throughout each group.
Every action is collected and replay-scored from the exact sampled IDs, prompt
IDs, image order/grids, and causal KV prefixes. Probability ratios must agree
before any update. The model remains in eval mode; adapter dropout is zero.

On the first round, synthetic objective/masking checks and a real saved-tensor
offload gradient comparison run on CUDA. One-, three-, and seven-image teacher
histories then verify finite nonzero language-LoRA gradients. Their gradients
are cleared, their RNG is restored, and their probes do not step the optimizer.
Actual sampled histories are checked during collection and again during their
backward pass, including twelve-action failures when present.

Only BF16 frozen base/vision weights and FP32 language `q_proj`/`v_proj` LoRA
are used. During collection, backward saved tensors are discarded. During
optimization, large backward tensors are copied exactly to CPU and restored
for backward; GPU parameter tensors stay on GPU. KV gradients are preserved.
This takes more host memory and time than ordinary inference. Detaching KV
state or replacing the gradient-enabled scorer with a faster BF16 batch scorer
would change the computation and is not an accepted memory workaround.

Each action is scored and backpropagated separately, accumulating gradients
across the group. Its graph is released before the next action. AdamW steps
once after the group; frozen base/vision parameters must have no gradients.
Zero-variance groups skip AdamW entirely and retain the identical fingerprint.

## 4. Continue real training state across bounded jobs

Every round runs in a separate 30-minute Modal function on one L40S, with two
CPUs and 64 GiB requested host memory for exact activation offload. Jobs run
sequentially. Development evaluation uses 24 GiB host memory and no backward.

The checkpoint contains:

- `checkpoint/adapter.zip`, excluding frozen base weights.
- `checkpoint/training-state.pt`, with Adam and Python/NumPy/Torch/CUDA RNG.
- Parameter ordering, fingerprints, Adam/RNG digests, and checksums.
- The complete attempted-round history, actual step counters, sampled group,
  exact images/actions/token scores, source/preflight receipts, and reload proof.

After saving, the same job unloads/reloads the adapter and restores Adam/RNG,
then compares updated-policy scores before and after reload. The next job
repeats restoration in a fresh container. A cursor into the declared schedule
and unique RNG streams prevents repeating the first group by accident.

Attempted rounds and optimizer updates are distinct: for example, ten attempted
groups with three nonconstant groups mean **three actual updates**, not ten.
The reader recomputes advantages, verifies state/weight/RNG continuity, checks
that skipped groups do not change parameters, and verifies optimizer counters.

The ceiling is twenty attempted groups and 1,800 seconds of recorded training
work. Before another job, the launcher conservatively projects the next round
from 1.25 times the slowest completed round plus thirty seconds. It stops before
launching a round predicted to exceed the remaining budget. This is a runtime
guard, not a monetary billing cap; container startup, artifact validation/save,
and development evaluation have additional cost. Actual runtime can differ from
the estimate. First two rounds are smoke checks; if neither has nonconstant
returns, the workflow stops rather than presenting skipped rounds as training.

## 5. Measure behavior before calling the experiment successful

All sixteen development cases run greedily from the original early adapter
before training. After the two-round smoke phase, the saved adapter is loaded
for another paired evaluation. If the budget permits continuation, the last
completed saved adapter receives the final evaluation. If the last checkpoint
is already the smoke checkpoint, its existing evaluation is reused.

Every model episode is reexecuted in a fresh browser. Initial goal/pixel hashes,
model/processor settings, decoding, case membership, and frozen evaluation
fingerprints must agree. The final report also includes Stage 14's verified
16/16 full-SFT result on the same development instances as a stronger supervised
comparison; it is not silently relabeled as the early policy's baseline.

Report successes, action lengths, invalid actions, group reward variance,
gradient norms, actual update count, reload checks, runtime, and peak allocated
GPU memory. Training reward changes occur under different states/policies and
are not a generalization curve. Development is one small familiar template and
has already been used during this tutorial; it is not a final-test estimate.

## Measured two-update checkpoint

The initial early-SFT adapter scored **1/16** on greedy development evaluation.
After two actual GRPO updates it scored **3/16**. Cases 013 and 014 changed from
failure to success, the original success on case 008 remained successful, and
no successful case regressed. This is a small observed development gain, not
a reliable generalization estimate or a result on the closed final test.
The separate completed SFT comparison remains **16/16**.

For case 013 (satire movies directed by Mcclure from 2015), the baseline's
seventh action clicked `(116.5, 270.75)` below the usable Submit region and
the episode reached twelve actions without success. The updated policy clicked
`(116.5, 260.25)` and succeeded on action seven. In case 014, its seventh click
changed from `(109.75, 272.25)` to `(109.5, 267.9)`, also producing success.
For both cases the screenshot after filling the fields and immediately before
action seven had **identical pixel hashes before and after training**.
This isolates a useful change in generated click coordinates under the same
visible filled-form state. It does not establish better visual perception or
memory, and the histories can still contain different earlier action strings.

The fresh training returns were `[0,0,0,1]` and `[0,1,1,0]`. Both groups stepped
Adam once. Gradient norms before clipping were 1.57 and 1.80; probability-ratio
deviation was zero. Both adapters changed and passed exact Adam/RNG and policy
score reload checks. The second job restored the first job's actual Adam step,
then saved step two. Recorded training work for those rounds was **910.2
seconds / 15.2 minutes**. This is GRPO optimization, not additional SFT.

## Final bounded result

| Attempted round | Training seed | Fresh returns | Adam step? | Cumulative Adam steps | Recorded work |
| --- | --- | --- | --- | ---: | ---: |
| 1 | 80000 | `[0,0,0,1]` | Yes | 1 | 491.1 s |
| 2 | 80000 | `[0,1,1,0]` | Yes | 2 | 419.1 s |
| 3 | 80001 | `[0,0,0,0]` | No | 2 | 241.4 s |
| 4 | 80002 | `[0,0,0,0]` | No | 2 | 173.0 s |

The two additional states produced no relative advantage and no parameter
change. Thus the final saved policy is exactly the two-update policy, rather
than a four-update policy. Three training instances were visited; only one
provided nonzero training updates. All ten declared training states passed
the oracle preflight, but the runtime budget did not allow visiting the rest.

| Paired development measurement | Early SFT | Final GRPO policy | Completed SFT reference |
| --- | ---: | ---: | ---: |
| Successful episodes | **1/16 (6.25%)** | **3/16 (18.75%)** | **16/16 (100%)** |
| Mean actions over all episodes | 11.0625 | 10.125 | 7.0 |
| Invalid actions | 5/177 | 6/162 | 0/112 |
| Execution errors | 0 | 0 | 0 |
| Illustrative 95% Wilson interval | 1.1%–28.3% | 6.6%–43.0% | 80.6%–100% |

Success improved on two cases; invalid-action frequency did not improve.
Shorter mean episodes include early failures and are not themselves proof of
better path efficiency. The intervals overlap substantially, and all sixteen
instances share one template. This is an observed development gain in one
small seeded experiment, not a statistically established generalization gain.
It does not outperform the fully supervised adapter or demonstrate convergence.

| Validation / compute | Measured result |
| --- | --- |
| CPU tests | **101 passed**, without local PyTorch |
| Completed model episodes / fresh-browser replays | **48 / 48**: sixteen sampled training episodes plus 32 development episodes |
| Training oracle episodes / fresh-browser replays | **10 / 10** |
| Sampled training actions | **177** |
| Training/development exact initial goal-and-pixel overlaps | **0** |
| Sampling/replay probability drift | **0.0** in collection and actual updates |
| Cross-job adapter/Adam/RNG restores | All three continuation jobs passed, with optimizer counters 1, 2, 2 |
| Save/reload policy-score deviation | **0.0** for all four saved checkpoints |
| Recorded training work | **1324.6 seconds / 22.1 minutes** |
| Recorded completed development work | **893.4 seconds / 14.9 minutes** |
| Peak allocated training GPU memory | **7.61 GiB** |
| Requested training host memory | **64 GiB**; host peak was not measured |
| Final adapter ZIP | **6.49 MiB** |
| Adapter + Adam/RNG + checkpoint metadata | **20.70 MiB** per checkpoint, plus trajectories/reports |
| Final-test cases opened | **0/32** |

The next-round projection was 643.9 seconds. Adding that to 1324.6 seconds
would exceed the 1800-second work budget, so the launcher stopped before
round five. It did **not** complete twenty rounds or twenty optimizer updates.
Recorded work excludes startup, preceding artifact verification, final
serialization, and the deliberately stopped duplicate evaluation described
below. CPU reporting and plotting use no GPU.

The final checkpoint is `stage16-20261019T155940Z-0927d223`; its adapter
fingerprint exactly equals `stage16-20261019T153959Z-51df9b05`, the completely
evaluated two-update checkpoint. After verifying the four committed manifests,
zero-variance skips, and fingerprint equality, we stopped the launcher's
redundant final evaluation and reused the complete sixteen-case smoke
evaluation. The workflow explicitly records this reuse and the stopped app ID.
The cancellation/`TargetClosedError` in the app log comes from that intentional
evaluation shutdown, not from failed training or lost optimizer state.
The summary independently requires the final saved fingerprint to match the
reused evaluation. These are the same measurements, not two independent
evaluations or selection of a better policy over a different final policy.

Open the [combined report and round replays](../artifacts/stage16-20261019T152005Z-b83a0b3b-workflow.html),
[machine-readable summary](../artifacts/stage16-20261019T152005Z-b83a0b3b-workflow-summary.json),
[training diagnostics](../artifacts/stage16-20261019T152005Z-b83a0b3b-workflow-training.png),
or [final checkpoint metadata](../artifacts/stage16-20261019T155940Z-0927d223/checkpoint/checkpoint.json).
The [workflow](../artifacts/stage16-20261019T152005Z-b83a0b3b-workflow.json)
records the preflight, four training jobs, before/smoke evaluation shards,
runtime stop, and explicit unchanged-policy evaluation reuse.

This establishes real multi-turn sparse-reward LoRA optimization and a small
measured improvement on this form template. It is still a short learning
experiment. Before a twenty-round run, profile and speed up the exact
generation/backward path while preserving its probability and gradient checks.
The all-failure groups also expose an exploration problem on additional states;
longer duration alone does not guarantee useful updates on them.

## Frequent problems

| Symptom | Check |
| --- | --- |
| Many rounds but few changed checkpoints | Count nonconstant groups and actual Adam steps |
| Loss is nearly zero | Normalized advantages cancel; inspect nonzero finite gradients and fingerprints |
| Clipping fraction is always zero | With one fresh on-policy step, initial ratios are one; clipping is not expected to activate |
| Repeated jobs restart learning | Restore Adam moments/counters and the schedule cursor, not only LoRA weights |
| An early adapter is called "untrained" | It already includes the earlier browser SFT; document the complete initialization |
| Probe trajectories are reused for optimization | Collect new groups and record their behavior fingerprints and RNG streams |
| GPU memory fits but backward is slow | Exact CPU activation offload uses host bandwidth and memory |
| Higher training success is claimed as generalization | Compare the same development cases before and after, with identical decoding |
| Twenty requested rounds are reported as twenty updates | Report attempted rounds, actual steps, and the runtime stopping reason separately |
| Small GRPO gain is presented as beating SFT | Include the completed SFT comparison and its stronger supervision budget |

## Exercise

Write out advantages for `[1,0,0,0]` and `[0,0,0,0]`. Explain why only the first
steps Adam, why a near-zero surrogate value can still have a nonzero gradient,
and why twelve-action failures should not automatically receive twelve times
the weight of short successes. Then identify which reported evidence would
show checkpoint correctness and which would show improved behavior.

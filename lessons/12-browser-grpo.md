# Stage 12 — Two browser GRPO rounds with a real gradient scorer

Stage 11 found a useful training signal on `multi-orderings`: returns
`[1,0,0,1]` and `[0,1,0,0]` on two training seeds. Tab search and popup login
had only failures. This lesson makes at most two updates on shuffled forms,
using **fresh trajectories after every optimizer step**.

Execution status: completed and GPU-validated as
`stage12-20261019T083719Z-c83d1c0e` in
[the Modal app](https://modal.com/apps/kiendepzai3/main/ap-uOWiFOUK17q76uHMb8Ou4I).
All 79 CPU tests and all 34 saved-episode checks pass. Two actual optimizer
updates completed, with exact sampling/gradient-score agreement and verified
checkpoint reload. Held-out success remained **0/8 before and 0/8 after**.
This establishes correctness checks, with no observed success improvement.

## Run the lesson

```bash
# Configuration/source checks only; no local Torch or browser required.
uv run vlm-multiturn-rl --stage 12 --episodes 2 \
  --sft-run-id stage9-20261019T041931Z-88adc041
uv run python -m unittest discover -s tests -v

# One L40S, at most two optimizer updates, 30-minute GPU function timeout.
uv run modal run modal_app.py --task browser-grpo --episodes 2 --updates 2 \
  --sft-run-id stage9-20261019T041931Z-88adc041 \
  --run-id stage11-20261019T070558Z-5074f9bb

# Offline checksums, groups, checkpoints, and paired evaluation records.
uv run vlm-multiturn-rl --stage 12 --replay artifacts/STAGE12_RUN_ID
```

Use your own completed Stage 9 SFT adapter and matching Stage 11 CPU preflight.
The launcher can find the latest compatible completed local preflight when
`--run-id` is omitted. It does not select an adapter by evaluation reward.
`--updates 1` runs one round; values above two are rejected in this lesson.
Two rounds can produce fewer than two optimizer steps if rewards are constant.
There is no automatic retry that samples until a desirable group appears.

`stage12.py` owns collection, optimization, checkpoints, and reporting.
`BrowserTrainPolicy` in `browser_policy.py` owns differentiable cached replay.
`modal_app.py` handles resources, preflight, persistence, and artifact downloads.
Browser dynamics, rewards, parser, screenshot scale, and model revision remain
those validated in the previous lesson.

## 1. A behavior score is not yet a training scorer

Stage 11's `BrowserCachedPolicy.score` deliberately rejects gradient-enabled
use. Its ordinary Transformers generation call runs under `no_grad`. Numerically
matching recorded probabilities does not prove that a tensor has a gradient
path to trainable parameters.

In pinned Transformers 5.20.1, `generate` accepts a custom decoding callable.
The callable receives prepared inputs and caches. `gradient_decode` enters
`torch.enable_grad()` and invokes that version's same `_sample` loop. This
includes prompt prefill, multimodal position handling, and every cached decode
step. See the pinned [generation implementation](https://raw.githubusercontent.com/huggingface/transformers/v5.20.1/src/transformers/generation/utils.py)
and [PEFT wrapper](https://raw.githubusercontent.com/huggingface/peft/v0.21.2/src/peft/peft_model.py).

Both rollout sampling and gradient replay use this callback, with model eval
mode and zero dropout. This avoids comparing a no-grad sampling kernel with a
potentially different gradient-enabled kernel. Collection builds a graph for
one action, records detached scalar behavior probabilities, and releases the
graph without backward. Saved backward tensors are discarded during collection;
this retains gradient-enabled forward kernels without retaining their activations. Weights stay fixed through the whole trajectory group.
The numeric protocol is `grad-cache-v1`; old Stage 11 trajectories are not
reused for optimizer updates.

Replay traverses the exact generated IDs, including EOS. A logits processor
captures each ordinary policy log probability **before** masking scores to
force the recorded next ID. Forced logits are only a traversal mechanism; their
chosen-token log probability of zero is not the policy probability.

The default dynamic cache concatenates key/value tensors. We retain their
autograd history and assert every decoder value-cache tensor has a gradient
function. Detaching a cache may preserve forward values while removing gradient
paths through earlier hidden states. See the pinned
[dynamic-cache implementation](https://raw.githubusercontent.com/huggingface/transformers/v5.20.1/src/transformers/cache_utils.py).

The private `_sample` call is intentionally version-sensitive. The browser lock
is unchanged. A future dependency upgrade must repeat calibration and backward
checks before accepting optimizer results. This is a small reference experiment,
not an assertion that every model/cache/checkpointing combination supports
cached training.

## 2. Calibrate before any optimizer step

On training-only teacher histories with one, three, and seven observations:

1. Sample a fresh text action with the gradient-enabled forward path.
2. Reconstruct the exact prompt tokens, images, and generated IDs.
3. Re-score those IDs with autograd enabled.
4. Check finite probabilities and unchanged-weight ratios.
5. Backpropagate a probe negative log probability.
6. Require finite nonzero language-LoRA gradients, no frozen-weight gradients,
   and unchanged adapter parameter fingerprints; clear probe gradients.

The fixed ratio check is

\[
\max_k\left|\exp(\log\pi_\theta(y_k\mid h,y_{<k})-
\log\pi_{\mathrm{behavior}}(y_k\mid h,y_{<k}))-1\right|\le10^{-4}.
\]

This is stricter than the earlier inference diagnostic's 0.1 log-probability
tolerance. A failure stops the run before optimization. Every real sampled
training action repeats the ratio check; calibration is not permission to skip
checks on longer or malformed continuations.

The GPU also runs synthetic causal-alignment, masking, clipping, detached-old
probability, and detached-advantage checks on tiny CPU Torch tensors. Local
standard-library tests validate task splits, saved report reconstruction,
checkpoint step counts, stale policies, and evaluation pairing.

## 3. Collect groups and compute the loss

Training uses seeds 3000 and 3001 from the selected form curriculum. Each round
starts from its specified task/seed and collects four independent trajectories.
Every rollout has a fresh browser context, up to 12 actions, full changing-image
history, and its own deterministic Torch/CUDA stream. Rewards remain sparse
binary success; there is no formatting bonus or per-action penalty.

For a group of complete returns,

\[
A_i=\frac{R_i-\bar R}{\operatorname{std}_{\mathrm{population}}(R)+10^{-6}}.
\]

A constant-return group skips the optimizer. Otherwise use the clipped
assistant-token objective with epsilon 0.2:

\[
L=-\frac1G\sum_i\frac1{N_i}\sum_{k\in\mathrm{assistant}(\tau_i)}
\min\left(r_{ik}A_i,
\operatorname{clip}(r_{ik},0.8,1.2)A_i\right).
\]

Here `N_i` counts all generated assistant tokens across trajectory i, including
EOS. Prompt tokens, images, tool feedback, and previous assistant turns are
context for the current action, not additional supervised targets. Assistant
spans are recorded as `[prompt_length, prompt_length + generated_length)`.

We normalize each trajectory by its own generated-token count, then average
across four trajectories. Averaging every action independently instead would
give short actions disproportionate weight. Averaging all group tokens together
would give long trajectories greater weight.

The same episode-level advantage weights every current action token. This is
coarse credit assignment: a missed click penalizes correct text tokens too.
A token ratio is a practical local surrogate, not a ratio of full trajectory
probabilities or a differentiable browser simulator.

At unchanged weights, ratios should be one. Because there is exactly one
optimizer step after a group, clipping is normally inactive in that update.
The loss value can also be near zero because advantages average to zero while
the policy gradient remains nonzero. Neither fact means the update failed.

## 4. Accumulate gradients without changing the behavior policy

Score one action, form its contribution divided by `G*N_i`, call backward, and
release its graph before the next action. No gradients pass through browser
transitions or discrete token selection. They pass through the current policy's
representations of its fixed history and generated-token prefixes.

The first attempt, `stage12-20261019T082701Z-5551c2f9`, passed all three
calibration backward checks but exhausted GPU memory while re-scoring a longer
rollout, before any optimizer update. Seven-image calibration alone did not
bound the memory cost of twelve-image or longer-token actions. The failed run
retains incomplete diagnostics and is excluded from completed results.

`saved_activation_memory` now uses PyTorch saved-tensor hooks. Collection
and collection-only re-scoring discard saved backward values, and reject an
accidental backward call. Actual optimization stores large non-parameter
activations on CPU and restores their exact values on the original device for
backward. Parameter storages remain on GPU; copying the frozen model weights
for every decode step would waste host memory. Small saved tensors also stay
on GPU. The callback preserves KV graph connections throughout.

A CUDA toy check compares offloaded and ordinary gradients and confirms that
collection-only graphs cannot be backpropagated. Real model calibration repeats
with CPU offload. The GPU job requests 64 GiB of host memory for activation
storage, still uses one L40S, and keeps the same task, image resolution, history,
action budget, optimizer, and strict probability-ratio check. See
[PyTorch's saved-tensor hooks](https://docs.pytorch.org/docs/2.14/autograd.html#torch.autograd.graph.saved_tensors_hooks).

Accumulate across all turns and all four episodes, clip the total gradient norm
to one, and perform one AdamW step. Learning rate is `1e-5`, weight decay zero,
KL beta zero. Vision and base weights remain frozen. No gradient checkpointing,
vLLM, distributed training, or off-policy replay is added here.

A new group is collected only after the step. The next group's behavior
fingerprint must equal the previous group's after-update fingerprint. Fresh
trajectories, distinct RNG streams, exact images/IDs, and live browser
re-execution remain required even when a group will eventually be skipped.

Zero KL is a transparent simplification for two rounds. A later longer learning
experiment needs its own drift and stability design; this run does not validate
long training or demonstrate that clipping alone prevents drift.

## 5. Persist and reload the real training state

After every round, save under `checkpoints/round-001/` or `round-002/`:

- `adapter.zip`: language-side LoRA weights and adapter configuration.
- `training-state.pt`: Adam state and Python/NumPy/Torch/CUDA RNG state.
- `checkpoint.json`: configuration, parameter ordering, counters, parameter
  fingerprint, tensor-state digests, and file checksums.

The frozen base model is reused from the model cache, not duplicated. Browser
contexts are already closed when checkpoint persistence stops Playwright's sync
runtime and commits the Modal volume; subsequent episodes start a new runtime.
This avoids Modal async I/O conflicting with Playwright's active sync loop.

Before after-training evaluation, unload the in-memory adapter, load the saved
adapter, reconstruct Adam state, and restore RNG state. Check exact parameter,
optimizer, and RNG digests, Adam counters, and a fixed probe's policy scores.
Evaluation therefore uses the actual saved adapter, not just the live weights.

The command saves and verifies round-boundary state but does not yet expose a
CLI continuation workflow. `--resume-run-id` is rejected instead of silently
restarting Adam. Interrupted runs retain progress and committed checkpoints
without a completed root manifest.

## 6. Evaluate the predefined held-out pool

Open the eight `multi-orderings` instances reserved in Stage 11, seeds
30000–30007, for paired evaluation. They were first opened by the failed implementation
attempt, which measured the SFT baseline at 0/8 before running out of memory.
The retry keeps the same predefined pool and settings; memory handling changed
independently of those rewards. Their oracle outcomes validate reachability;
oracle actions never enter training loss, demonstrations, or group advantages.

Evaluate the original SFT adapter greedily on all eight cases before training,
then evaluate the reloaded GRPO adapter on exactly those cases with identical
decoding. Check each pair's initial goal and pixel hash. The remaining 16
reserved tab-search/popup instances are not opened by this lesson.

Report success, invalid actions, execution errors, episode length, groups,
gradient norms, runtime, and memory. A two-round result is a correctness
experiment with a small evaluation set, not convincing evidence of learning or
generalization. Do not tune the curriculum or run length after seeing these
evaluation rewards and still describe the same pool as untouched.

A complete two-round report contains 34 episodes: ten oracles, eight sampled
training trajectories, and sixteen paired model-evaluation episodes. If configured
for one round, it contains 30 episodes. `groups.json` links exact trajectories
to advantages and update proofs; `training.json` records calibration and reload
checks. `report.html` contains screenshots, actions, and the comparison.

## 7. Measured result

| Measurement | Result |
| --- | --- |
| Round 1 returns | `[1,0,0,1]` |
| Round 2 returns, collected after update 1 | `[0,1,0,0]` |
| Actual optimizer updates | **2**, no zero-variance skips |
| Pre-clipping gradient norms | **2.4500**, **2.1418**; both clipped to 1 |
| Scored assistant tokens | 698 in round 1; 922 in round 2 |
| Maximum sampling / gradient-scoring discrepancy | **0.0** |
| Maximum unchanged-weight ratio deviation | **0.0** |
| Clipped-token fraction before each step | 0.0, as expected for one step per fresh group |
| Adapter fingerprints | Changed after each optimizer step |
| Adapter / Adam / RNG reload | All exact; probe log-probability difference **0.0** |
| CUDA activation-offload gradient check | Maximum difference **0.0** |
| CPU tests / saved-episode checks | **79 / 34**, all passed |
| Source verification | Completed manifest matches Python source at completion |
| GPU function time | **1180.2 seconds / 19.7 minutes** |
| Peak allocated CUDA memory | **7.57 GiB** on one L40S |
| Host memory request | 64 GiB for offloaded activations; host peak was not recorded |
| Saved training checkpoints | **20.7 MiB each**, **41.4 MiB total** |

Each checkpoint contains about 6.5 MiB of compressed LoRA adapter data and
14.2 MiB of Adam/RNG state, plus metadata. The frozen base model is not copied
into those checkpoints. Runtime excludes container startup and the earlier
failed attempt.

| Paired held-out metric, same eight cases | SFT source | After two GRPO updates |
| --- | --- | --- |
| Successful episodes | **0/8** | **0/8** |
| Mean actions | 12.0 | 12.0 |
| Invalid actions | 6/96 (6.25%) | 1/96 (1.04%) |
| Execution errors | 0 | 0 |

Lower invalid-action frequency did not translate into task success. There are
only two selected training seeds, two optimizer steps, and eight held-out
instances. These results do not establish that GRPO cannot learn the task, or
that the model lacks capacity. They also do not establish improvement.

In held-out case `multi-orderings-hard-eval-000`, both policies generated the
requested year, genre, and director strings. However, the genre/director clicks
fell just left of the inputs, leaving those fields empty. Both then repeatedly
clicked `(121.75, 301.25)`, below the visible Submit button, until the action
limit. This is visible state the model could have checked after typing; correct
text content alone did not complete the form. The greedy action sequence on
this case was unchanged by the two updates.

Open the [complete replay report](../artifacts/stage12-20261019T083719Z-c83d1c0e/report.html),
[training checks](../artifacts/stage12-20261019T083719Z-c83d1c0e/training.json),
[group objectives](../artifacts/stage12-20261019T083719Z-c83d1c0e/groups.json),
and [checkpoint metadata](../artifacts/stage12-20261019T083719Z-c83d1c0e/checkpoints/round-002/checkpoint.json).

The next learning study needs more training-instance diversity and a practical
compute/resume plan. This readable reference already takes nearly 20 minutes
for two rounds plus evaluation, so a 20-round job must not be assumed to fit
its 30-minute timeout. Measure collection, scoring, and backward throughput
before scaling. Keep new-workflow SFT for tab search and popup recovery separate
from the present form-only RL result, and reserve an additional final test pool
if subsequent decisions use these evaluation outcomes.

## 8. Frequent problems

| Symptom | Check |
| --- | --- |
| Matching scores but no gradients | Ordinary generation is no-grad; inspect `requires_grad`, cache history, and a real backward probe |
| Ratio differs before a step | Match forward kernels, cache handling, image ordering, exact tokens, and unchanged weights; do not widen tolerance |
| Out of memory | Discard collection saved tensors, offload backward activations, keep parameter storages resident, and release each action graph after backward |
| All-zero advantages | Count constant-return skips; two rounds do not guarantee two optimizer updates |
| Loss near zero | Inspect gradient norm and changed adapter fingerprint; centered advantages can cancel scalar loss |
| Large improvement on training cases only | Check paired held-out evaluation; selected training seeds already influenced curriculum choice |
| An extra image or stale screenshot | Inspect each turn's ordered image paths, grids, and prompt IDs before accepting its scores |
| Checkpoint loads but Adam differs | Verify parameter ordering, moments, step counters, and RNG; adapter-only reload is not training-state restore |
| Browser runtime error during volume commit | Close Playwright's sync runtime before Modal I/O, then reinitialize it for later episodes |

## Exercise

For a mixed group, compute its advantages and each trajectory's `N_i`. Find one
failed click followed by correctly generated text. Explain which tokens receive
the negative advantage, why previous observations remain visible to attention,
and why the environment image itself never receives a policy-loss target.

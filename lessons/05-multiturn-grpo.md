# Stage 5 — Implement multi-turn GRPO explicitly

This stage replaces the solver's supervised next action with four sampled
trajectories and their episode returns. You can already understand a single-turn
GRPO loss; the new work is preserving the right observation history for every
action and grouping complete interactions from one initial board.

Read [grpo.py](../src/vlm_multiturn_rl/grpo.py) for the arithmetic and
[stage5.py](../src/vlm_multiturn_rl/stage5.py) for collection, scoring, and the
optimizer loop. Heavy imports remain inside the remote path.

## 1. Check the objective without a GPU

```bash
uv run vlm-multiturn-rl --stage 5
uv run python -m unittest discover -s tests -v
```

The first command needs neither local PyTorch nor Pillow. Its controlled group
has returns `[-0.08, -0.08, 0.98, -0.08]`, mean `0.185`, population standard
deviation about `0.459`, and advantages about
`[-0.57735, -0.57735, 1.73205, -0.57735]`.
It also checks both signs of clipping, finite zero-variance behavior, and that
excluded observation entries can contain arbitrary values without changing loss.

The remote function additionally runs tiny PyTorch checks before loading Qwen.
These verify the differentiable objective, clipped gradients, detached old
probabilities/advantages, and causal indexing. This separates mistakes in loss
arithmetic from VLM integration failures.

## 2. What changes from single-turn GRPO?

In single-turn GRPO, you sample several completions for one prompt. Here you
sample several complete episodes for one initial board:

```text
                         same initial board
                         /      |      |      \
                      τ₀       τ₁     τ₂       τ₃
                    o₀,a₀    o₀,a₀   ...      ...
                    o₁,a₁    o₁,a₁
                    ...      ...
                    R₀       R₁      R₂       R₃
```

The initial image is the same. Later images can differ because earlier actions
differ. All four environments reset independently. Each trajectory has a distinct
deterministic RNG seed; the generator is seeded once per trajectory, not reset
at every action. Sampling occurs sequentially on one GPU.

The behavior policy is held fixed throughout all four episodes and their old-score
passes. We fingerprint its LoRA tensors before and after collection and reject
changes. Only after collecting a complete group do we compute advantages and
accumulate its gradients. We then make at most one optimizer step, discard the
group for optimization, and collect fresh episodes for the next round.

Training boards follow a deterministic round-robin order. Evaluation boards stay
held out. The collector does not retry a group until it contains a success, choose
boards based on observed rewards, or replace failed trajectories with solver paths.

## 3. Derive the trajectory policy

Let \(s_t\) be private environment state, \(o_t\) the rendered observation, and
\(h_t\) the public conversation available before action \(a_t\). The raw action
is a sequence of assistant tokens; our parser maps that sequence to a movement
or an invalid action. For a fixed initial task:

\[
p_\theta(\tau)=\rho(s_0)\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid h_t)\,
P(s_{t+1},o_{t+1}\mid s_t,a_t).
\]

The environment and renderer do not have trainable policy parameters, so:

\[
p_\theta(\tau)\propto\prod_t\pi_\theta(a_t\mid h_t),\qquad
\log\pi_\theta(a_t\mid h_t)=
\sum_{k=1}^{K_t}\log\pi_\theta(a_{t,k}\mid h_t,a_{t,<k}).
\]

Consequently, the score-function gradient for expected episode return contains
\(R(\tau)\sum_{t,k}\nabla\log\pi_\theta(a_{t,k}\mid h_t,a_{t,<k})\).
Each factor must use the history that actually preceded that action. Scoring an
entire episode as one assistant response would include environment tokens as
decisions and would not represent this interaction policy.

Our rewards remain unchanged: `+1` at the goal and `−0.01` per attempted action.
We use undiscounted episode return \(R=\sum_t r_t\), with an eight-action collection
cap. There is no distance shaping, format bonus, or reward for a valid call.

## 4. Compute one group-relative advantage per episode

For the four returns:

\[
\bar R=\frac1G\sum_i R_i,\quad
\sigma_R=\sqrt{\frac1G\sum_i(R_i-\bar R)^2},\quad
A_i=\frac{R_i-\bar R}{\sigma_R+10^{-6}}.
\]

We explicitly choose population standard deviation. Library defaults may use
sample standard deviation, which changes advantage scale for a group this small.
The advantages are constants during differentiation.

GRPO uses group rewards to construct relative advantages rather than learning
a separate critic; see [DeepSeekMath](https://arxiv.org/abs/2402.03300). Our
implementation is a minimal token-clipped variant with explicit normalization
and **KL coefficient zero**. It does not claim to reproduce every trainer's
defaults or the full original paper objective.

Equal-return groups produce exactly zero advantages. We skip `optimizer.step()`
entirely, including Adam's state/momentum update. All failures return `−0.08`
under this reward contract, so they provide no relative signal even when their
actions differ. All successes with identical episode lengths also give zero
signal; successes of different lengths can still differ because of action cost.
We do not add noise or divide by an arbitrarily small variance to manufacture
learning.

Broadcasting \(A_i\) to every generated token is coarse credit assignment. A
successful episode's detours and malformed calls receive its positive advantage
alongside useful actions. A failed episode's useful early moves receive its
negative advantage. There is no per-turn value function or return-to-go estimate
in this lesson. Stage 8 will examine those alternatives.

## 5. Score exact action tokens under their actual histories

For each recorded action, the scorer:

1. Reprocesses that action's saved public messages with the ordered images.
2. Requires the processed prefix IDs, image paths, and image grids to match the
   generation record.
3. Appends the saved `generated_ids` directly, including EOS only if emitted.
4. Extends attention/type metadata for the appended assistant text.
5. Gathers only generated-token log probabilities from the model output.

It never decodes and re-tokenizes a sampled action, substitutes the solver's
direction, adds an EOS to a length-truncated answer, or scores the next board
as something the policy generated.

If the prefix has \(P\) tokens and the action has \(K\) tokens, score logits at
`P-1 ... P+K-2` against IDs at `P ... P+K-1`. The model's output at the final
input position predicts an unsupplied next token and is excluded. `selected_log_probs()`
does this shift once. There are no SFT labels passed to the model for this loss.

The processor's image tensors and multimodal type IDs are preserved. Image
features can affect the assistant logits through attention while contributing
zero direct policy-loss terms. The loss mask is separate from the attention
mask: history remains visible.

The sampler uses full-softmax multinomial sampling at temperature 1, no top-k,
no top-p filtering, and no repetition penalty. Those settings avoid scoring a
different distribution from the one that produced the tokens. For example,
sampling with a truncated top-p distribution and then using unmodified softmax
scores as exact behavior probabilities would be inconsistent. The relevant
generation settings are documented in the [Transformers generation reference](https://huggingface.co/docs/transformers/main_classes/text_generation).

This reference disables the KV cache during rollout generation and scoring to
reduce differences between token-by-token generation and full-action teacher
forcing. This costs throughput, especially with several images in history.
Different BF16 matrix shapes can still cause numerical differences. We retain
the sampling-time selected-token log probabilities and compare them with scores
from the fixed behavior model. A maximum absolute log-probability difference
over `0.1` fails the run; the observed maximum is saved for inspection.

The old scores used by the objective are computed under the unchanged behavior
model with exactly the later gradient-scoring code path, then detached and
stored as numbers. This makes the initial ratios reproducible. The tolerance
above concerns numerical sampling/scoring drift, not permission to change
temperature or filtering. We use `model.eval()` for scoring with autograd enabled:
evaluation mode disables dropout; it does not mean `no_grad()`.

## 6. Build the practical clipped surrogate

Define the per-token ratio using that token's actual history:

\[
r_{i,t,k}(\theta)=\exp\left(
\log\pi_\theta(a_{i,t,k}\mid h_{i,t},a_{i,t,<k})-
\log\pi_{\rm old}(a_{i,t,k}\mid h_{i,t},a_{i,t,<k})\right).
\]

Our minimized loss is:

\[
\mathcal L=-\frac1G\sum_i\frac1{N_i}\sum_{t,k}
\min\left(r_{i,t,k}A_i,
\operatorname{clip}(r_{i,t,k},0.8,1.2)A_i\right),
\qquad N_i=\sum_t K_{i,t}.
\]

This averages each trajectory's generated-token terms, then averages trajectories.
A long failed episode does not get a larger total weight just because it has
more tokens. Averaging all tokens across the batch instead would weight longer
episodes more heavily. Normalizing each turn independently would introduce yet
another weighting. Tests distinguish these choices.

The joint trajectory probability ratio would multiply all the token ratios.
We use a token-level clipped surrogate, not a clipped joint trajectory ratio.
The length normalization and group-relative standardization are practical
estimator choices; this is not an unbiased implementation of the raw
expected-return score-function formula above. The clipped construction follows
the idea introduced by [PPO](https://arxiv.org/abs/1707.06347).

Clipping is sign-sensitive. If \(A>0\), raising a ratio from 1.2 to 1.5 gives no
additional improvement. If \(A<0\), lowering it from 0.8 to 0.5 gives no additional
improvement. Changes in the harmful direction remain penalized. Clamping the
ratio unconditionally and omitting `minimum` would implement a different loss.

**With one optimizer step per fresh group, ratios start at 1 and clipping is
normally inactive.** We assert the pre-step ratios remain within `1e-4` of 1.
Synthetic cases exercise clipping away from 1; the real smoke run verifies the
on-policy update. Clipping does not constrain a large step taken afterward.
We use a small learning rate and gradient-norm clipping, but neither is a KL
trust-region guarantee.

At ratio 1, the group loss is approximately `−mean(A) = 0`. Its gradient need
not be zero: different sampled token sequences have different score gradients.
Check the gradient norm and adapter fingerprint rather than treating an initial
zero loss as a broken update.

## 7. Run the bounded GPU smoke test

```bash
# Uses the latest completed GPU Stage 4 report under artifacts/ by default.
uv run modal run modal_app.py --task grpo-grid --updates 2 --episodes 4

# Pin the starting checkpoint explicitly for a repeatable comparison.
uv run modal run modal_app.py --task grpo-grid --updates 2 --episodes 4 \
  --sft-run-id stage4-20261017T175637Z-7415e007
```

For your own Stage 4 run, substitute its ID. The checkpoint is read from the
persistent Modal run volume. A CPU job verifies its checksums, protocol, model
revision, dataset seed, and rank before the GPU job starts. GPU loading verifies
the source receipt again. Checkpoints are selected by completion time when the
flag is omitted, never by evaluation score.

The 30-minute GPU function loads the same BF16 base model and the SFT adapter
with training enabled. Only language-side LoRA weights receive gradients; the
base, visual encoder, and merger are frozen. AdamW uses learning rate `1e-5`,
zero weight decay, and gradient-norm clipping at 1. There is no critic, reference
model, KL penalty, entropy bonus, vLLM, or distributed execution.

Backward passes use one recorded action at a time. Each turn's loss is divided
by `G * trajectory_token_count`; gradients accumulate across every turn of all
four episodes. No optimizer step occurs inside those loops. This reference
keeps the model in evaluation mode during backward and does not rely on training-mode
gradient checkpointing. Reduce image resolution if a long-history scoring pass
exceeds memory; gradient accumulation alone cannot shrink that forward pass.

`--updates 2` means two collection/update rounds. An equal-return group is logged
as a skipped round. `training.json` records the actual optimizer-step count.
No retries are hidden inside the run.

```bash
# Verify checksums, replay transitions, and validate recorded groups on CPU.
uv run vlm-multiturn-rl --stage 5 --replay artifacts/RUN_ID
```

## 8. Inspect the artifacts

| File | Evidence |
| --- | --- |
| `groups.json` | Four member indices, same initial layout, seeds, returns, advantages, policy fingerprints, skips/updates |
| `episodes.json` | Actual public histories, raw action IDs, sampling and old log probabilities, outcomes |
| `mask-inspection.json` | Long-history prefix, assistant IDs, loss mask, shifted prediction indices |
| `training.json` | Actual optimizer steps and integration checks |
| `source-adapter.json` | Exact SFT run and archive/manifest hashes |
| `metrics.json`, `replay.html` | Paired greedy SFT/GRPO evaluation and training-rollout diagnostics |
| `images.zip` | Actual per-turn PNG observations |
| `adapter.zip` | Final adapter for inference or the next lesson |
| `environment.json`, `manifest.json` | Sampling/scoring settings, memory/runtime, versions, artifact/source hashes |

SFT and final GRPO evaluation use identical held-out boards and greedy decoding.
Training rollouts use sampling and training boards; their success rate is not a
paired estimate of held-out improvement. Stage 6 adds fuller comparisons,
checkpoint/resume, and learning curves. This stage saves an adapter checkpoint
without optimizer/RNG resume state.

After the smoke test passes, the same entry point permits `--updates 20` for a
bounded learning experiment. Measure collection/scoring time and zero-variance
fraction first. Raising the update count does not help if every group is skipped.
The timeout remains 30 minutes; no improvement is promised by the command.

### Results from the verified smoke run

Run `stage5-20261017T180901Z-cf26666d` started from the pinned 20-update SFT
checkpoint shown above, with seed 42 and 448² images. All 39 CPU tests passed,
as did the differentiable tensor checks and the GPU integration checks.

| Round | Episode returns | Advantages, approximately | Outcome |
| --- | --- | --- | --- |
| 1 | `[-.08, -.08, -.08, -.08]` | `[0, 0, 0, 0]` | Skipped; adapter unchanged |
| 2 | `[.93, .98, -.08, -.08]` | `[.9511, 1.0477, -.9994, -.9994]` | One real optimizer step |

The second round's gradient norm was `0.3437`; its adapter fingerprint changed.
Its initial loss was `1.86e-9`, ratios were exactly 1, and clipping fraction was
zero. This is the near-zero-loss/nonzero-gradient case derived above. Sampling
versus behavior-scoring log probabilities differed by at most `0.000719`, well
within the numerical guard. The eight trajectories had distinct per-group seeds
and complete, independently reset histories.

Both SFT and SFT → GRPO solved **1/4 held-out boards** with greedy decoding, with
identical aggregate return, efficiency, and invalid/blocked-move rates. The smoke
test demonstrates a working update; it establishes no evaluation improvement.
The sampled training rollouts solved 2/8 training tasks and had a 22.8% invalid
action rate. Stochastic sampling can expose malformed calls even when greedy
evaluation produces valid calls.

The seven-action success in round 2 includes a blocked move, yet that move's
tokens receive the same positive `0.9511` advantage as its useful moves. This is
a concrete episode-credit limitation, not evidence that collision rewards changed.
The two-action success gets a slightly larger advantage due to the action cost.

The recorded compute phase took `159.9` seconds and peaked at `18.24 GiB` of
allocated GPU memory. Collection/scoring/update took about 49 and 48 seconds for
the two rounds. These measurements exclude CPU preparation, GPU scheduling,
and final report packaging. The saved mask example has eight image observations,
2,626 prefix tokens with zero loss weight, and 14 generated tokens with unit
loss weight. All 16 trajectories (eight training rollouts and eight paired
evaluation episodes) passed CPU dynamics/history/group replay.

Open `artifacts/stage5-20261017T180901Z-cf26666d/replay.html` to examine these
episodes, and `groups.json` to reproduce the arithmetic. The final adapter is
saved in that run's `adapter.zip`.

## 9. Common failures

| Symptom | Interpretation or next check |
| --- | --- |
| Every group returns `−0.08` | Sparse-reward exploration failure. All advantages are zero; the optimizer should skip. |
| Four identical trajectories | Check seeds and sampling settings; greedy generation cannot provide useful stochastic groups. |
| Very different initial boards inside one group | Grouping bug: compare alternatives from the same initial task. |
| Initial ratios differ from 1 | Stale weights, dropout, mismatched history/IDs, or a different scoring path. The run rejects this. |
| Sampling log probabilities differ substantially from old scores | Check sampling transformations, processor metadata, precision, cache mode, and causal alignment. |
| Loss is near zero but adapter changes | Expected for centered advantages at initial ratio 1; inspect gradient norms. |
| Clipping fraction stays zero | Expected with one update per freshly collected group. Test clipping with controlled ratios. |
| Long failures dominate gradients | Check the trajectory denominator; do not normalize each action separately by accident. |
| Successful episodes contain bad moves | Episode-level credit rewards all their tokens. This is a limitation of this estimator. |
| Training reward rises but evaluation does not | Sampling noise, board memorization, or policy drift. Use paired held-out evaluation. |
| A malformed action appears | Keep it and spend a turn. Repairing it with a solver would change the collected policy. |
| Memory grows with turn count | Full history contains increasingly many visual tokens; inspect prefix lengths and lower resolution. |

## Exercise

Find one nonconstant group in `groups.json`. Recompute its returns and advantages
from the four referenced episodes. Identify an action in a successful trajectory
that did not help reach the goal: what advantage does it receive anyway?

Then inspect `mask-inspection.json`. If its prefix length is \(P\), identify the
first scored logit position and explain why the preceding tool response has no
policy-loss terms despite being visible. Finally, explain why the group's loss
can start near zero while its adapter fingerprint changes after the update.

If your run contains only equal-return groups, use the CPU synthetic example
for the arithmetic and explain why the real run correctly skipped optimization.
Do not manufacture a successful rollout to complete the exercise.

Continue with [Lesson 06: training, resume, and evaluation](06-training-evaluation.md).

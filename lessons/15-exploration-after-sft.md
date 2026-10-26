# Stage 15 — Find a useful RL signal after SFT

Stage 14 improved greedy form development success from 2/16 to 16/16. That is
evidence for SFT on this template. Before another GRPO experiment, we need to
check whether the policy can produce both successful and unsuccessful complete
trajectories from the same training instance. This stage collects evidence;
it performs **zero optimizer updates**.

Execution status: completed and GPU-validated in
[the Modal app](https://modal.com/apps/kiendepzai3/main/ap-w0qCqkCxXtdF9PPsoQVSLZ).
All 32 sampled trajectories, eight groups, and fresh-browser replays passed
verification. One group from the early checkpoint had mixed returns; all six
groups from the final checkpoint had zero variance. Final-test cases remain closed.

## Commands

```bash
# Standard-library source checks and the complete predeclared study plan.
uv run vlm-multiturn-rl --stage 15 \
  --sft-run-id stage14-20261019T102008Z-1255a893 \
  --warmup-run-id stage14-20261019T100154Z-0bf33123

uv run python -m unittest discover -s tests -v

# Six CPU oracle preflights, then eight sequential four-trajectory GPU groups.
uv run modal run modal_app.py --task browser-exploration \
  --sft-run-id stage14-20261019T102008Z-1255a893 \
  --warmup-run-id stage14-20261019T100154Z-0bf33123

# Validate a completed downloaded CPU or GPU bundle without local PyTorch.
uv run vlm-multiturn-rl --stage 15 --replay artifacts/STAGE15_RUN_ID

# Generate a local summary from a completed workflow and verified bundles.
uv run python lessons/tools/summarize_exploration.py artifacts/STAGE15_WORKFLOW.json
```

The local reader also needs the original Stage 9, Stage 13, and Stage 14 input
directories alongside the probe. It verifies their checksums and contracts.
Every remote shard reads the existing persistent cache; it saves observations,
exact actions/tokens/probabilities, groups, a standalone replay page, source
receipts, and environment details. No base weights or new adapter are saved.

For an infrastructure retry, `--run-id CPU_PREFLIGHT_ID --pilot-case N` reruns
one declared group, where N is 0–7. Configuration and core-source hashes must
still match the preflight. A retry is for failed execution, not for replacing
an unfavorable completed group. A partial workflow is labeled `partial-study`.
The summary helper requires each of the eight declared groups exactly once.

## 1. A good SFT policy can be a poor starting point for a particular RL task

For four returns from one initial state, we reuse population-standard-deviation
normalization:

\[
\bar R=\frac14\sum_i R_i,\qquad
\sigma_R=\sqrt{\frac14\sum_i(R_i-\bar R)^2},\qquad
A_i=\frac{R_i-\bar R}{\sigma_R+\epsilon}.
\]

`[1, 1, 1, 1]` and `[0, 0, 0, 0]` both give zero advantages. With the current
sparse reward and no separate regularization term, neither group supplies a
reward-driven policy gradient. Adding epsilon prevents numerical problems; it
does not manufacture information. Failures on one initial state and successes
on another do not create within-group variance.

If each rollout independently succeeds with probability p on a fixed initial
state, the probability of observing a mixed group of size four is

\[
P(\mathrm{mixed})=1-p^4-(1-p)^4.
\]

At p=0.5 this is 0.875; at p=0 or 1 it is zero. This is an illustrative
Bernoulli calculation, not an estimated success probability for our agent.
The readiness measurement itself is small and noisy: two training seeds per
task and four samples per seed. No mixed group observed does not prove mixed
groups are impossible.

Greedy evaluation and sampled exploration answer different questions. A model
can succeed on all sixteen greedy development cases but sometimes fail when
sampling at temperature one on new training cases. We measure this rather
than infer it from the development score.

## 2. Declare the study before looking at returns

| Candidate | Existing checkpoint | Tasks | Training seeds | Groups / trajectories |
| --- | --- | --- | --- | ---: |
| Final form SFT | Stage 14 update 214, all 853 targets once | `multi-orderings`, `click-tab-2-hard`, `login-user-popup` | 80000, 80001 per task | 6 / 24 |
| Early form SFT | Stage 14 update 2, eight optimized targets | `multi-orderings` only | 80000, 80001 | 2 / 8 |

The final candidate tests the improved adapter on fresh form instances and on
the tab-search and popup workflows that were difficult in Stage 11. The early
candidate asks whether a shorter SFT warmup leaves useful form exploration.
It is a saved ancestor of the final checkpoint, not a newly trained model or
a checkpoint selected after inspecting these probe returns.
Both candidates already contain Stage 9's 100 browser-SFT updates; “early”
refers only to the additional form curriculum.

The source validator checks the complete pass, update-two cursor, exact shared
configuration, training-row prefix, and actual parent chain. It also checks
model revision, processor settings, dependency lock, and action/history budgets.
Stage 14 trained only forms; loading it on the other two templates is an
explicit task transfer. An early/final comparison does not isolate an optimal
SFT duration, and these candidates use different declared sampling RNG streams.

All cases here are marked `train`. Seeds 80000–80001 are disjoint from the
form training pool 50000–50119, development pool 60000–60015, final pool
70000–70031, and the earlier harder-task pilot. Fresh form seeds add instance
diversity within the same familiar template; they are not new websites.
The 32 form final cases and the reserved tab/popup evaluation cases stay closed.

For task definitions see the official
[tab-search environment](https://miniwob.farama.org/environments/click-tab-2-hard/)
and [popup login environment](https://miniwob.farama.org/environments/login-user-popup/).
Runtime assets stay at the existing immutable MiniWoB commit and SHA-256.

## 3. Keep group collection on policy

Each group starts four fresh BrowserGym worlds with the same task seed. The
model gets only the goal, screenshot, and public feedback. The privileged oracle
first verifies all six tasks are solvable and replayable; its DOM data is never
passed to the policy. The action cap stays twelve and the context guard 8,192
tokens, with the existing image-resolution limits.

We assign a distinct reproducible Torch/CUDA seed to every trajectory. Initial
instructions and pixel hashes must agree within a group. Every member keeps
its own full ordered image history and a unique screenshot namespace. Reusing
an image filename across rollouts would silently replace an earlier observation
in the asset dictionary, even when saved message paths still looked correct.

Sampling uses temperature 1, top-p 1, top-k 0 and no repetition penalty. The
behavior adapter remains unchanged throughout collection and replay scoring.
Four trajectories from one checkpoint form one group; mixing early and final
policies in a group would invalidate the behavior-policy interpretation.

## 4. Use the future GRPO scoring path now

`BrowserTrainPolicy` uses Stage 12's `grad-cache-v1` path for both sampling and
replay. It keeps the same grad-enabled forward kernels and exact causal cache
prefixes. For collection-only scoring, backward saved tensors are discarded,
so no optimizer, backward pass, or parameter update occurs. This is distinct
from saying that all forward computation runs under `torch.no_grad()`.

For every generated token, including EOS when sampled, we require

\[
\left|\exp\left(\log\pi_{\rm replay}
 -\log\pi_{\rm sample}\right)-1\right|\le 10^{-4}.
\]

We store the actual sampled token IDs and log-probabilities, reconstruct the
same prompt IDs, image ordering/grids, and causal prefixes, then store replay
log-probabilities and the current-assistant loss span. Short oracle-history
probes, with up to three images, fail early if this interface is broken. Actual sampled
histories are checked too, including long failing episodes.

Stage 10's legacy group helper allows a looser absolute log-probability
tolerance. This stage adds the strict ratio check both online and in the CPU
reader; it never relies on that legacy tolerance. The older full-batch BF16
scorer remains unsuitable for the behavior denominator. We do not widen a
tolerance to get the study to pass.

These are collection/scoring checks. A later GRPO job must still perform its
own real backward and update/save/reload checks. This stage does not establish
that an optimization step will improve behavior.

## 5. Record complete evidence and decide conservatively

Each group runs in its own 30-minute function on one L40S, with two CPUs and
24 GiB requested host memory. The eight groups run sequentially. At most
32 × 12 = 384 model actions and 24,576 generated completion tokens are allowed;
there are also bounded calibration calls. Partial evidence is saved if a shard
fails, without a completion manifest. A completed bundle records zero optimizer
steps and identical before/after adapter fingerprints.

All sampled episodes are reexecuted in fresh browsers. The offline reader
recomputes groups, advantages, reward statistics, exact RNG assignments, spans,
and strict probability ratios. The summary combines only the declared shards,
checks source/preflight relationships and decoding agreement, and labels every
number as a training-only measurement.

| Outcome within a checkpoint/task | Interpretation and next action |
| --- | --- |
| At least one mixed group | A bounded GRPO correctness experiment has a possible relative-reward signal; still evaluate learning separately |
| Every member succeeds | This reward/case pool is saturated under the probe; consider greater training diversity or task difficulty |
| Every member fails | Inspect grounding, tool format, and workflow failures; targeted supervision may be needed before sparse-reward GRPO |
| Both all-success and all-failure groups, none mixed | Aggregate variation exists but every current group has zero relative advantage |
| Probability-ratio or replay check fails | Fix collection/scoring/environment consistency before interpreting reward |

Do not add a formatting reward just to make groups nonconstant: that changes
what the policy is optimized to do. Likewise, repeatedly sampling new seeds
until a desired result appears would conceal the exploration cost. This study
has a fixed budget and reports every declared group.

## Frequent problems

| Symptom | Check |
| --- | --- |
| High greedy success but uncertain GRPO readiness | Measure sampled complete returns from the same initial training state |
| Nonzero dataset reward variance but all advantages zero | Compute variance within each same-state group |
| Probe does not reproduce the future trainer | Use the same sampling/scoring kernels, temperature, masks, and token IDs |
| A replay graph is mistaken for a training graph | Collection mode discards saved backward tensors; actual backward uses Stage 12's separate memory path |
| Every form succeeds but tabs fail | Additional form SFT did not establish tab-search competence |
| A "hard" task sometimes ends in one action | Inspect actual oracle paths; a tab-search seed may start with the target already visible |
| Wrong screenshots appear in later analysis | Give every trajectory its own image-path namespace |
| A completed bundle still appears incomplete | Check its completion manifest and final progress status |
| A retry changes the claimed study | Retain completed outcomes; distinguish infrastructure retries from additional exploration |

## Measured result

These are temperature-one **training-only probes**, not held-out evaluation.
Each row includes both declared seeds, with four independent samples per seed.

| Candidate / task | Successful trajectories | Returns by group | Mixed groups | Invalid actions |
| --- | ---: | --- | ---: | ---: |
| Final SFT / forms | **8/8** | `[1,1,1,1]`, `[1,1,1,1]` | 0/2 | 0/56 |
| Final SFT / tab search | 4/8 | `[0,0,0,0]`, `[1,1,1,1]` | 0/2 | 11/30 |
| Final SFT / popup login | 4/8 | `[1,1,1,1]`, `[0,0,0,0]` | 0/2 | 0/28 |
| Early SFT / forms | **1/8** | `[1,0,0,0]`, `[0,0,0,0]` | **1/2** | 9/91 |

The final adapter has 50% success on each of the two harder task templates,
but every one of its groups has zero relative advantage. On tab seed 80001,
the target is already visible and all four attempts succeed in one action.
On seed 80000, the oracle switches tabs before clicking the target, while all
four model attempts fail. One attempt emits an unsupported `terminate` tool
eleven times; others repeat a click or select a wrong link.

Popup login has a similar branch distinction. Seed 80000 does not trigger a
popup; all four model episodes finish in five actions. Seed 80001 opens an
"Exit to home page?" dialog after focusing the username. Every model attempt
clicks the left **OK** button and fails on action two. The oracle clicks
**Cancel** at `(148.95, 249.96)`, focuses the input again, and completes the
login in seven actions. This is a workflow decision failure with a valid tool
call, not a browser execution error or evidence of a model capacity limit.

The early form checkpoint does give a nonconstant return group on seed 80000:
`[1,0,0,0]`, with mean 0.25 and population variance 0.1875. Its advantages are
approximately `[1.732, -0.577, -0.577, -0.577]`. The successful trajectory clicks
Submit at `(119.5, 260.5)` and terminates in seven actions. Failed trajectories
click below the button around y=272–274, repeat themselves, type `Submit`, or
emit unsupported tools. This provides a concrete action distinction for a
future reward-driven update. The other early-checkpoint seed is all failure.

| Validation / compute measurement | Result |
| --- | --- |
| CPU tests | **97 passed**, with no local PyTorch |
| CPU teacher preflights / oracle actions | **6 / 29**, all freshly replayed |
| Sampled trajectories / actions / groups | **32 / 205 / 8** |
| Fresh-browser model replays | **32 passed** |
| Sampled-action probability checks | **205 passed**, exact token/history replay |
| Maximum sampling/replay log-probability or ratio deviation | **0.0** |
| Adapter before/after fingerprints | Identical for every shard; source reload fingerprints match |
| Execution errors | **0** |
| Mixed / all-success / all-failure groups | **1 / 4 / 3** |
| Recorded probe work time | **1012.1 seconds / 16.9 minutes**, sequential one L40S |
| CPU preflight time | **51.7 seconds** |
| Peak allocated CUDA memory | **7.56 GiB** |
| New optimizer updates / new model checkpoints | **0 / 0** |
| Form final-test instances opened | **0/32** |

Recorded probe work includes model loading, calibration, collection, scoring,
and browser reexecution. It excludes container startup, preceding source/cache
validation, and final artifact serialization. The 7.56 GiB figure is for
collection and scoring without backward; actual GRPO training has a different
activation-memory requirement. The CPU teacher histories are calibration inputs,
not additional optimization targets in this stage.

All eight GPU manifests match the CPU preflight's core-source hashes, and the
current core source still matches those recorded inputs. The combined summary
also checks matching decoding settings, preflight receipts, distinct trajectory
IDs, and initial instruction/pixel equality against each teacher case.

The next controlled experiment should initialize GRPO from the **early form
checkpoint**, collect fresh same-case groups, and begin with two update attempts
plus real backward and checkpoint/reload checks. Measure that checkpoint before
and after training on the declared development pool; keep the final SFT adapter
as a stronger supervised comparison. Do not optimize stored probe episodes as a
substitute for fresh collection, or promise improvement over the already strong
final SFT policy. The observation of one mixed group is a readiness result, not
evidence that GRPO has learned or that this is the optimal warmup duration.

The final adapter's tab-switching and popup-dismissal branches need targeted
training demonstrations if we choose that curriculum next. Their current
all-failure groups do not provide a reward-driven gradient in this GRPO setup.
Choosing the early form policy is a practical next correctness experiment;
it does not establish long-horizon planning or memory improvements.

Open the [combined results and group replays](../artifacts/stage15-20261019T143704Z-7b6ed5ac-workflow.html),
[machine-readable summary](../artifacts/stage15-20261019T143704Z-7b6ed5ac-workflow-summary.json),
or [mixed-group trajectories](../artifacts/stage15-20261019T144956Z-cc0a6ba8/report.html).
The [workflow](../artifacts/stage15-20261019T143704Z-7b6ed5ac-workflow.json) lists
the CPU preflight and all eight completed run IDs. Each group has its own
checksummed bundle; the summary page is regenerated locally from those bundles.

## Exercise

For four trajectories from each initial state, compare returns `[1,1,1,1]` and
`[0,0,0,0]` with two groups of `[1,0,1,0]`. Both studies have the same aggregate
success rate. Compute their advantages and explain why only the second supplies
a reward-driven update in our implementation. Then calculate the mixed-group
probability for p=0.1 and explain why four failures do not prove p=0.

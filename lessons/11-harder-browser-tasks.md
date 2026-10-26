# Stage 11 — Find browser tasks with useful exploration

The 100-update adapter solved all 24 fresh evaluation instances and all 24
sampled training episodes in Stage 10. Its six groups therefore had zero
relative advantages. This lesson changes the task templates and measures
exploration **before making optimizer updates**.

Execution status: implemented; all 76 CPU tests pass. Remote CPU preflight
`stage11-20261019T070558Z-5074f9bb` solved and live-replayed all six training
instances: tab search took 6/3 actions, shuffled forms 7/7, and popup login 7/7.
Both login seeds triggered popups: password focus for seed 3000, username focus
for seed 3001. All six GPU shards completed in
[the Modal app](https://modal.com/apps/kiendepzai3/main/ap-YWIXfD6779poDuuysN8KF0).
The measured model results follow below.

## Run the pilot

```bash
# Standard-library source, split, and configuration checks.
uv run vlm-multiturn-rl --stage 11 --episodes 2 \
  --sft-run-id stage9-20261019T041931Z-88adc041
uv run python -m unittest discover -s tests -v

# CPU oracle and live browser re-execution, followed by sequential L40S jobs.
uv run modal run modal_app.py --task browser-hard-pilot --episodes 2 \
  --sft-run-id stage9-20261019T041931Z-88adc041

# Alternatively, inspect demonstrations before requesting the GPU.
uv run modal run modal_app.py --task browser-hard-check --episodes 2 \
  --sft-run-id stage9-20261019T041931Z-88adc041

# Reuse that CPU run; optionally select just one task for a bounded retry.
uv run modal run modal_app.py --task browser-hard-pilot --episodes 2 \
  --sft-run-id stage9-20261019T041931Z-88adc041 \
  --run-id CPU_RUN_ID --pilot-task click-tab-2-hard

# Offline checksums, histories, groups, and metrics; no local Torch/browser.
uv run vlm-multiturn-rl --stage 11 --replay artifacts/GPU_RUN_ID
```

The CPU preflight has a 15-minute timeout. Each task/seed group gets its own
30-minute GPU function, with sequential calls using one L40S at a time. There are two training
seeds and four independently sampled trajectories per seed: eight model
episodes per task, 24 across the full pilot. Total wall time can exceed 30
minutes because there are six separate GPU jobs. No new adapter or optimizer
checkpoint is written. Supply `--episodes 1` for a smaller pilot, with a matching
CPU preflight.

The launcher saves completed reports immediately. A later failed task does not
invalidate earlier completed shards. Retry only the failed group with
`--pilot-task TASK_NAME --pilot-case 0` (or `1`) and the same CPU run. Incomplete GPU evidence is retained
in the run volume without a completion manifest. It is not a finished result.

## 1. Three different challenges

| Task | Policy behavior needed | Teacher behavior |
| --- | --- | --- |
| `click-tab-2-hard` | Read links, open different tabs, and find the named target | Inspect only visible target links; otherwise visit the next tab in order |
| `multi-orderings` | Bind genre, director, and year values to shuffled labels | Locate each labeled field, click, type, then submit |
| `login-user-popup` | Notice an interruption, dismiss it correctly, and continue login | Cancel a visible popup before filling remaining fields |

The benchmark [tab source](https://raw.githubusercontent.com/Farama-Foundation/miniwob-plusplus/7fd85d71a4b60325c6585396ec4f48377d049838/miniwob/html/miniwob/click-tab-2-hard.html)
randomizes two through six tabs and terminates on a link click. The
[form source](https://raw.githubusercontent.com/Farama-Foundation/miniwob-plusplus/7fd85d71a4b60325c6585396ec4f48377d049838/miniwob/html/miniwob/multi-orderings.html)
shuffles three rows and checks all three submitted values. The
[popup source](https://raw.githubusercontent.com/Farama-Foundation/miniwob-plusplus/7fd85d71a4b60325c6585396ec4f48377d049838/miniwob/html/miniwob/login-user-popup.html)
triggers a popup on username focus, password focus, or neither. It is triggered
by interaction rather than an elapsed-time timer. Some seeds therefore have no
popup; inspect the replays before describing every login instance as harder.

These are candidate tasks, not an established difficulty ranking for this
adapter. Shuffled forms primarily test visual grounding; they do not by
themselves establish a memory requirement. Tab search makes observation history
useful for tracking visited panels, but this pilot contains no memory ablation.

## 2. Preserve the interface while changing the task

The existing `BrowserConfig` keeps its original tasks, eight actions, and
4096-token guard. `BrowserDifficultyConfig` explicitly selects the new suite,
12 actions, and an 8192-token guard. Its system instruction differs only in
stating twelve rather than eight actions. Image resolution, model revision,
strict JSON parser, coordinate scale, allowed tools, and sparse binary reward
remain unchanged. The probe uses the validated cached sampling/replay path
described below.

A longer cap lets the agent recover from an interruption or search several
panels. It also permits longer failure loops. The context guard rejects an
oversized history instead of silently discarding observations. Longer
histories cost GPU work even though we are not training.

The existing adapter is transferred unchanged. `source-adapter.json` records
its exact hashes and the intentional task/budget/context changes. Checks still
require the same model/processor configuration, MiniWoB assets, tool protocol,
and browser dependency lock. This is transfer to new templates, not an SFT
resume or a controlled comparison changing only one variable.

`oracle_next_action` is reactive. Computing a fixed action list at reset would
miss the popup and cannot locate links that appear only after opening a tab.
The teacher uses privileged DOM inspection to choose actions, but executes
exactly the same click/type API as the model. DOM text, selectors, hidden
panels, seeds, and teacher decisions never enter policy observations.

All selected oracle episodes must succeed and re-execute in fresh browsers
before requesting any GPU. This verifies reachability under the action budget,
coordinate scaling, binary reward, and action execution for the selected seeds.
It does not verify every possible seed of these templates.

## 3. Training selection is separate from evaluation

Training seeds are 3000 and 3001 per task. `reserved-eval.json` declares seeds
30000–30007 per task before the pilot, but this stage does not open, demonstrate,
or score them. Curriculum choices use only training groups.

`episodes.json` stores real training labels throughout. `metrics.json` is empty
because the shared evaluator intentionally accepts only evaluation episodes.
`difficulty.json` contains explicitly named **training_metrics**, aggregated
from a temporary view without changing the saved episodes. A training success
rate must not be presented as held-out generalization.

CPU reports contain six successful oracle episodes and training-only
`demonstrations.json`. Each GPU shard contains those six oracle records plus
four sampled episodes. All six shards together contain 24 unique model
episodes; the same six oracle records are copied into each of the six reports.

## 4. Select tasks using within-case reward variation

For each fixed task/seed, sample four complete trajectories with frozen adapter
weights and independent reproducible Torch/CUDA streams. Every trajectory starts
in a fresh browser context. Initial instructions and pixel hashes must match.
Preserve exact assistant IDs, image order, sampling scores, and cached replay behavior
scores. Re-execute all model actions in fresh browsers.

For each group, compute

\[
A_i = \frac{R_i-\bar R}{\operatorname{std}_{\mathrm{population}}(R)+10^{-6}}.
\]

| Observed group | Meaning for the next experiment |
| --- | --- |
| Both successes and failures | A nonzero GRPO signal exists for this case |
| Four successes | This case is saturated under sparse success reward |
| Four failures | No successful exploration was sampled; inspect behavior and consider a small SFT bridge |

A task can have one all-success case and one all-failure case. Its aggregate
success rate is 50%, but both groups still have zero advantages. Choose from
`mixed_return_groups`, not aggregate success alone. Four samples can also miss
rare successes or failures; zero observed variance is not proof that the true
return distribution is constant.

Mixed training returns support a bounded correctness update as the next
lesson; it is not evidence that GRPO improves generalization. Before that
update, verify unchanged-weight behavior scores against the actual
gradient-enabled scorer, and handle prefix graph memory explicitly.

## 5. Use KV caching without inventing behavior probabilities

The first non-cached 12-action tab rollout took several minutes. We stopped
that exploratory run (`stage11-20261019T070739Z-976ec182`) after retaining an
incomplete diagnostic trajectory. It is not included in the completed pilot.
This motivated smaller jobs and a faster inference implementation; it did not
motivate lowering the action cap or changing rewards.

`BrowserCachedPolicy` uses Transformers generation with a KV cache for both
sampling and replay. Replay starts from a fresh cache and traverses the exact
recorded token IDs, including EOS. A custom logits processor records the
ordinary policy log probability **before** replacing the scores with a mask
that forces the recorded next token. The forced distribution controls which
history is visited; it must never supply the behavior probabilities, since its
chosen-token log probability would always be zero.

`ReplayCursor` checks the actual generated prefix before every forced token.
Replay preserves Torch/CUDA RNG state. Model generation, cache preparation,
multimodal position handling, and forward shapes are shared with sampling.
See the official [generation interface](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
and [logits processor interface](https://huggingface.co/docs/transformers/main/en/internal/generation_utils).

Each job first tests sampling/replay agreement on training-only oracle histories
with one and three images (or the final turn for a shorter demonstration).
Every sampled action is subsequently re-scored, and each group must pass the
unchanged 0.1 maximum absolute log-probability tolerance. The old full-sequence
scorer remains a diagnostic, not the behavior policy. Saved metadata names the
new numeric protocol `cached-replay-v1`; identical seeds need not reproduce
trajectories from the previous non-cached protocol.

This API is **no-grad only**, because Transformers generation does not provide
the differentiable scorer needed for training. It explicitly rejects use with
gradients enabled. The slower `BrowserPrefixPolicy` remains available. A later
GRPO update must independently validate its actual gradient scorer at unchanged
weights; this inference pilot does not resolve that requirement.

## 6. Inspect artifacts and failures

Open `report.html` for screenshots and exact actions, `groups.json` for returns
and advantages, and `difficulty.json` for the per-task decision. The local
launcher also writes `artifacts/CPU_RUN_ID-pilot-summary.json` as each group
finishes. Manifests bind source, configuration, dependencies, and artifacts.

Common issues:

- **Repeated coordinates across changed screens:** inspect whether the model
  grounds the latest image instead of replaying the familiar login script.
- **Typing into disabled fields:** a popup may have appeared after the previous
  focus click; the next screenshot must affect the next decision.
- **Invalid JSON or repeated failed submissions:** these are exploration failures
  even if the task looks semantically easy. Inspect raw output before adding SFT.
- **All-zero groups:** teaching a few successful training demonstrations can
  initialize new workflows. Evaluation demonstrations remain excluded.
- **Action/context limit reached:** inspect actual histories. Do not shorten the
  budget to manufacture difficulty or silently drop old images to make a job fit.
- **Scrolling tasks later:** the current scroll tool points at screenshot center,
  which may fall outside MiniWoB content. Fix targeting before using such tasks.

## 7. Completed pilot results

| Task | Training successes | Same-case returns | Mixed groups | Next experiment |
| --- | --- | --- | --- | --- |
| `click-tab-2-hard` | 0/8 | `[0,0,0,0]`, `[0,0,0,0]` | 0/2 | Add tab-search SFT demonstrations before sparse-reward RL |
| `multi-orderings` | 3/8 | `[1,0,0,1]`, `[0,1,0,0]` | **2/2** | First candidate for a bounded browser GRPO correctness update |
| `login-user-popup` | 0/8 | `[0,0,0,0]`, `[0,0,0,0]` | 0/2 | Teach interruption recovery before sparse-reward RL |

These are training selection outcomes, not held-out evaluation scores. All 24
reserved evaluation instances remain unopened. Increasing training reward on
these six cases later would not establish generalization.

| Validation / compute | Measured result |
| --- | --- |
| CPU tests | 76 passed |
| Oracle validation | 6/6 successful cases; 37 training action demonstrations |
| Unique sampled model episodes | 24; all live-reexecuted |
| Saved GPU episode checks | 60 records, including repeated copies of the six oracle episodes |
| Sampling / cached replay maximum log-probability discrepancy | **0.0** |
| Old batched diagnostic maximum discrepancy | 0.713928; not used as behavior probabilities |
| Invalid model actions | Tab search 28/96 actions; forms 1/71; popup login 0/24 |
| Model execution errors | 0 |
| Frozen adapter | Same parameter fingerprint across all six jobs |
| Source verification | All completed GPU manifests matched the Python source at Stage 11 completion |
| Optimizer updates / new adapter checkpoints | **0 / 0** |
| Completed GPU function time | 976.3 seconds total, **16.3 minutes** |
| Peak allocated GPU memory | **8.15 GiB** on one L40S |

GPU function time excludes CPU preflight, container startup, and the earlier
intentionally stopped non-cached diagnostic job. Each completed group took
63.5–263.1 seconds, within its separate 30-minute timeout.

Read the [aggregate results](../artifacts/stage11-20261019T070558Z-5074f9bb-aggregate.json)
and [per-group summary](../artifacts/stage11-20261019T070558Z-5074f9bb-pilot-summary.json).
The primary replay reports are:

| Task | Seed 3000 | Seed 3001 |
| --- | --- | --- |
| Tab search | [Replay](../artifacts/stage11-20261019T071627Z-9f5ec679/report.html) | [Replay](../artifacts/stage11-20261019T072105Z-f87924c8/report.html) |
| Shuffled forms | [Replay](../artifacts/stage11-20261019T072452Z-65cc02db/report.html) | [Replay](../artifacts/stage11-20261019T072746Z-32c05267/report.html) |
| Popup login | [Replay](../artifacts/stage11-20261019T073102Z-6cd38efd/report.html) | [Replay](../artifacts/stage11-20261019T073245Z-378011ab/report.html) |

Use shuffled forms for the next small GRPO experiment. Its two observed groups
have useful reward variation already, so a new SFT bridge is not required to
obtain a first exploration signal there. Tabs and popups need new-workflow
initialization or a separate exploration intervention. Do not combine a
successful form episode with a failed popup episode to manufacture a group.

Before optimizer updates, implement and verify a differentiable scorer against
the cached behavior distribution at unchanged weights. The current
`BrowserCachedPolicy.score` intentionally rejects gradient-enabled use. Only
then collect fresh same-case trajectories and test at most two updates before
considering a longer training run. Keep the reserved evaluation contract fixed.

## 8. Failure diagnosis

The first shuffled-form group completed as `[1, 0, 0, 1]`. In one failed
trajectory the model generated the correct director, year, and genre strings,
but clicked just left of the genre input. After typing, the screenshot still
showed an empty genre field; the model then submitted and failed. This is a
visual action-grounding and state-checking failure, not evidence that the model
could not extract the requested values. Another trajectory missed the Submit
target and later emitted an unsupported `submit` tool call.

This illustrates coarse episode-level credit: the same negative advantage
applies to correct text tokens and the mistaken click. The pilot identifies a
learning signal; it does not establish that episode-relative GRPO will assign
that credit efficiently or improve held-out success.

In popup login, all eight samples chose **OK** on the modal asking whether to
exit to the home page. The environment correctly terminated with failure. The
teacher chose **Cancel**, re-focused the interrupted field, and completed login.
The model's tool calls were valid, so action-format accuracy alone would have
missed this workflow failure.

Tab search produced repeated clicks, poor target coordinates, and unsupported
`terminate` calls. Declaring success or including a model-written reward did
not change the environment outcome: the parser rejected those calls and they
consumed an action. This separates protocol compliance from successful search.

## Exercise

Pick two cases with the same aggregate task success but different group
compositions. Explain why only mixed groups produce relative advantages. Then
inspect a failure replay and classify it as visual grounding, workflow
selection, action formatting, or exploration. Suggest one targeted training-only
intervention and keep the reserved evaluation pool unchanged.

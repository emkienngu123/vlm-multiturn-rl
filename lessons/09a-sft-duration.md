# Stage 9A — Test whether the browser policy needs more SFT

The 20-update experiment was deliberately short. With accumulation two and 18
action examples, it processed 40 examples, or approximately 2.22 dataset passes:

\[
\text{passes}=\frac{\text{updates}\times\text{accumulation}}{\text{action examples}}.
\]

Undertraining is a reasonable hypothesis. More updates alone might also
memorize our six training browser instances. This experiment isolates training
duration before moving to browser GRPO.

**Execution status:** Completed on October 19, 2026, as
`stage9-20261019T041931Z-88adc041`. Local verification passed for all 43 saved
episodes, training masks, training/evaluation summaries, and artifact checksums.
See the measured comparison below.

## Run the controlled comparison

```bash
uv run python -m unittest discover -s tests -v
uv run modal run modal_app.py --task browser-sft --episodes 2 --updates 100 \
  --run-id stage8-20261019T034403Z-a72e8d1c \
  --sft-run-id stage9-20261019T035637Z-5b8a7079
uv run vlm-multiturn-rl --stage 9 --replay artifacts/NEW_STAGE9_RUN_ID
```

`--run-id` identifies the completed CPU preflight containing demonstrations.
**For `browser-sft`, `--sft-run-id` is an optional diagnostic reference adapter,
not a training-resume checkpoint.** The reference must use exactly the same
source demonstrations, browser settings, learning rate, accumulation, and LoRA
rank, with fewer updates.

The 100-update run starts fresh from the base model. Seed, action schedule,
rank-8 language LoRA, learning rate `2e-4`, processor, assistant-only loss, and
greedy evaluation remain unchanged. It processes 200 action examples, about
11.11 dataset passes. Browser SFT now permits up to 100 updates within the
existing 30-minute timeout; the default remains two. Grid lesson limits remain
unchanged. We choose 100 before observing its outcome and retain both results.

## Measure training fit and free-running behavior

The original probe loss measured one training action. It could not establish
fit across the entire dataset. The optional duration diagnostic adds:

1. **Mean teacher-forced loss on all 18 training actions**, with the same
   action-level normalization used in training. These inputs contain correct
   demonstration histories.
2. **Six free-running training-browser episodes per adapter.** The model acts
   from its own history and receives resulting screenshots, without oracle
   correction. This tests whether fitting demonstrations translates to execution.
3. **The usual six evaluation episodes**, kept separate from training diagnostics.
   Their seeds never enter optimization.

The code loads the saved reference, measures its training loss and rollouts,
unloads it without merging, and verifies restoration of the base probe outputs.
It restores Python, NumPy, CPU Torch, and CUDA RNG states before fresh training.
After training, the reloaded 100-update adapter receives the same diagnostics.

`training.json` contains `duration_diagnostics`: source-adapter hashes, both
full-training losses, per-task training success, dataset exposure, and the
maximum difference between the new run's first 20 update losses and the original
run's losses. The HTML report includes this section. All diagnostic episodes
and screenshots are saved, and their action sequences are re-executed in fresh
browsers. `metrics.json` still reports **evaluation instances only**. Diagnostic
actions never become additional oracle targets.

## Interpret the evidence

| Observation | Interpretation and next experiment |
| --- | --- |
| Training fit and held-out success improve | More SFT helped at this data scale; retain the measured adapter as a GRPO candidate |
| Training improves, held-out success stays flat | Duration improved fitting; expand training layout/instruction diversity |
| Teacher-forced loss is low, free-running training episodes fail | Inspect action precision, wrong focus, exposure bias, and missing recovery demonstrations |
| Training itself stays poor | Inspect all-target losses/masks, coordinate representation, adapter capacity, and learning-rate settings |
| Training succeeds, held-out behavior deteriorates | Longer training may overfit; preserve both outcomes rather than selecting duration using these same evaluation cases |

Undertraining and insufficient data can coexist. The 20-update policy used
valid calls but repeated `(100,80)` on randomized button layouts, while the
field-entry templates succeeded. Per-task training/evaluation comparisons help
separate these problems. Low loss alone does not establish visual grounding.

These six evaluation cases have now influenced development decisions. They
remain disjoint from optimization, but they are not an untouched final test.
Use fresh seeds/layouts for a stronger final claim.

## Measured 20-versus-100-update comparison

The user ran the prepared command. The recorded Python source hashes match the
implementation, and the new run's first 20 update losses match the original
20-update run exactly: maximum difference **0.0**. The reference adapter was
unloaded, the base probe restored, and RNG states restored before fresh training.

| Measurement | 20 updates | 100 updates |
| --- | --- | --- |
| Processed action examples | 40 | 200 |
| Dataset passes | 2.22 | 11.11 |
| Mean teacher-forced loss, all 18 training actions | 0.369407 | 0.014009 |
| Free-running training-browser success | 3/6 | **6/6** |
| Evaluation success | 4/6 | **6/6** |
| Invalid-action rate, training/evaluation | 0% | 0% |

At 20 updates, both training `click-test` instances and one training `enter-text`
instance failed. At 100 updates, all training and evaluation cases succeeded.
Each button case took one action, each text-entry case three, and each login case
five; evaluation mean episode length therefore fell from 5.33 to 3.00 actions.

The evaluation button actions now depended on their screenshots:

```text
click-test-eval-000: click(139.5, 219.75) -> success
click-test-eval-001: click(158.0, 220.5) -> success
```

Previously, both repeatedly clicked `(100,80)`. The successful evaluation
coordinates also differ from the two training button targets, `(109.5,253.5)`
and `(78.75,213.75)`. This is evidence of useful coordinate transfer on these
specific instances, although two evaluation button layouts are a very small test.

These observations support **insufficient SFT duration as a real limitation of
the earlier run**, rather than concluding that the remaining failures required
GRPO. They do not establish broad browser competence or a need to stop SFT at
exactly 100 updates.

The 100 optimizer updates took 104.7 seconds. The complete GPU function, including
reference diagnostics, loading, fresh training, save/reload, browser rollouts,
re-execution, and reporting, took 334.9 seconds (about 5.6 minutes). Peak allocated
CUDA memory was 10.56 GiB on one L40S. Adapter weights remained 7,393,888 bytes.

All adapter-update, frozen-parameter, assistant-mask, causal-alignment, and reload
checks passed. Both training diagnostic policies and the evaluated SFT actions
were re-executed in fresh browsers by the job; local verification checked their
saved records without launching another GPU job.

Open [the completed browser report](../artifacts/stage9-20261019T041931Z-88adc041/report.html)
and [the full training diagnostics](../artifacts/stage9-20261019T041931Z-88adc041/training.json).
The saved `adapter.zip` in this run is the 100-update candidate for subsequent
exploration tests.

## When to begin GRPO

SFT need not be perfect. We need valid actions and enough successful exploration
on training tasks to produce **mixed-return groups**. All-failure and all-success
groups both have zero relative advantage under the current sparse binary reward.
After this duration study, inspect stochastic training groups. If a task produces
only failures, add training demonstrations or improve exploration before a long
GRPO run. Keep evaluation cases out of sampling and curriculum decisions.

The current six training cases now all succeed under greedy decoding. Before
GRPO, measure whether stochastic rollouts produce mixed-return groups and test
more unseen evaluation seeds. If training groups are uniformly successful, add
harder training instances or tasks to provide a learning signal; perfect scores
on this tiny subset alone do not justify an RL update.

## Exercise

For each task, compare 20 versus 100 updates on training loss, training rollout
success, and evaluation success. Decide whether the evidence supports
undertraining, limited data diversity, exposure bias, or a combination. Support
your diagnosis with actual replayed actions.

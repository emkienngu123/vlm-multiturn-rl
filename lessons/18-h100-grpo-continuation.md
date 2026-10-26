# 18 — Continue real GRPO training on H100

Stage 17 measured speed using disposable updates on archived trajectories.
This lesson collects new screenshot → action → environment trajectories and
updates the actual learning checkpoint.

```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/modal run modal_app.py --task browser-h100-grpo
```

The origin is `stage16-20261019T155940Z-0927d223`: four attempted groups,
two real optimizer steps, and the saved LoRA adapter, Adam moments, step
counters, and RNG states. No Stage 17 benchmark weights enter training.
The origin and all source artifacts must exist locally and on the run volume.

We use one `H100!`, BF16, frozen vision/base weights, language-side LoRA,
sequential groups of four, a 40 GiB saved-activation retention budget, and
exact CPU offloading for overflow. Model, dependency lock, tool protocol,
rewards, sampling settings, learning rate, and objective stay fixed.
Each remote group has a 30-minute timeout, two requested CPU cores and
64 GiB host memory. H100 development evaluation uses the existing 24 GiB
host-memory request. Local exercises still require no Torch or browser stack.

The original remaining schedule is preserved: resume at round five, seed
80003, and continue toward twenty **total groups**, not twenty guaranteed
optimizer steps. Sampling seeds come from the original unused round indices.
Equal-return groups skip Adam and retain its step counter and weights.

This is a new bounded experiment with an additional 1,800 seconds of recorded
training work. The budget excludes the earlier L40S experiment, CPU preflight,
development evaluation, and container startup outside the recorded function
work. Before starting a group we reserve 1.25 times the slowest completed
H100 group plus 30 seconds; the first estimate is 450 seconds. A prediction
is not a hard wall-clock cancellation. The original training history remains
intact, so cumulative work across experiments can exceed thirty minutes.

An explicit migration receipt permits exactly one source/hardware boundary
at the origin checkpoint. It records the origin manifest and source hashes,
adapter identity, original counters, H100 request, and memory profile. Later
resumes require identical source hashes and execution profiles again. This
keeps source checking strict rather than turning it off to allow migration.
New CPU teacher preflight uses the current source and rechecks all ten tasks.

Adam moments carry information about earlier gradients, so restoring only
LoRA weights would change the optimization process. We restore and verify
the real optimizer and RNG state, perform gradient checks on the resumed
model, and verify every saved checkpoint by reloading adapter/Adam/RNG and
comparing scores. Collection never steps the optimizer; each four-member
group uses a fixed behavior adapter.

GPU architectures can produce different BF16 scores. No L40S old log
probabilities are used in an H100 objective. We sample new H100 trajectories,
retain their exact generated token IDs and actual image histories, and
require unchanged-policy probability ratios to match before backward.
The first migrated group also runs short/middle/long gradient probes and
the saved-activation gradient check before collecting training data.

Evaluate all sixteen development cases **on H100 before and after**, with
the same goals, initial pixels, and greedy decoding. This separates training
effects from hardware-dependent numerical differences. Evaluate again after
two new groups if weights changed. Reuse an earlier complete H100 evaluation
only when its adapter fingerprint equals the final checkpoint exactly;
reused episodes do not count as independent new evidence. The thirty-two
final-test cases remain closed. The older full-SFT result is historical
context, not a newly paired H100 comparator.

The workflow writes checkpoint bundles, trajectory replays, gradient and
ratio diagnostics, phase timing, an HTML comparison, and a JSON summary
under unique Stage 18 run IDs. Bundles retain the Stage 16 artifact schema so
existing replay tools can inspect them; the new H100 receipt identifies the
phase. Rebuild its summary without GPUs:

```bash
.venv/bin/python -c 'from vlm_multiturn_rl.stage18 import summarize; summarize("artifacts/STAGE18_WORKFLOW-workflow.json")'
.venv/bin/modal run lessons/tools/plot_h100_grpo.py \
  --workflow artifacts/STAGE18_WORKFLOW-workflow.json
```

Common issues: a faster GPU cannot create reward variance in an all-failure
group; training updates and attempted groups must be counted separately.
The retention budget is not a total VRAM cap. Image history still grows with
turns. A higher training success rate does not establish development gains,
and sixteen instances on one familiar template provide limited evidence.
Modal responses are normalized to plain JSON values to keep framework scalar
subclasses from triggering local Torch imports.

## Measured continuation

The [completed report](../artifacts/stage18-20261019T171720Z-b1512977-workflow.html)
contains learning diagnostics, action replays, checkpoint provenance, and
screenshots of the newly successful development case.

| Measurement | H100 before | After two new groups | H100 final |
| --- | ---: | ---: | ---: |
| Successful development episodes | 3/16 | 3/16 | 4/16 |
| Mean actions | 10.44 | 10.44 | 10.13 |
| Invalid actions / all actions | 8/167 | 6/167 | 2/162 |
| Execution errors | 0 | 0 | 0 |
| Total optimizer steps | 2 | 4 | 7 |

Seven new groups sampled 28 fresh trajectories and made five actual updates.
Two all-failure groups skipped Adam. Across both experiments the checkpoint
has eleven attempted groups and seven optimizer steps. The five new mixed
groups occurred on distinct training layouts. All ten training layouts have
now been visited across the experiments.

The new phase recorded 1,541.1 seconds (25.7 minutes) of training work. The
next conservative group reservation was 372.7 seconds, exceeding the remaining
budget, so round twelve was not started. The twenty-total-group target was
not completed. Six eight-case development shards added 1,194.9 seconds
(19.9 minutes); evaluation and preflight were outside the training-work cap.

All sampled/replayed behavior ratios matched exactly. All adapter/Adam/RNG
restores and checkpoint reloads passed, with zero score difference after
reload. The first migrated model passed one-, three-, and seven-image gradient
probes. Every timed update kept its saved activations on GPU with no CPU
overflow copies. Peak allocated GPU memory was 22.48 GiB and peak reserved
memory was 51.13 GiB. All 106 CPU tests passed.

The final checkpoint is `stage18-20261019T180025Z-b146d1dd`; its adapter,
Adam/RNG state, and metadata occupy 21,700,768 bytes (20.7 MiB), excluding the
cached base model and replay artifacts. The original checkpoint is preserved.
Source hashes match throughout this new phase. The old source boundary is
permitted only through the explicit origin receipt. No final-test cases were
opened and no development episodes were reused in this phase.

The one new development win is `forms13-dev-010`; the three previous successes
were retained. Both policies generated the right strings for the year,
director, and genre. In the before replay, clicks at x=72.75 fell left of the
director and genre fields, leaving them empty despite the generated typing
actions. The final policy clicked inside those fields at x=79.5 and clicked
Submit at (119.9, 251.9), succeeding in seven actions. Its before-action-seven
screenshot consequently differs from the earlier policy's screenshot.
This is evidence of improved visual action grounding on this case, rather
than a newly demonstrated memory skill.

The measured gain is one case: 18.75% → 25% development success. Illustrative
95% Wilson intervals are about 6.6–43.0% and 10.2–49.5%, respectively. They
overlap substantially, and fixed cases on one familiar template are a small
diagnostic. This is a bounded learning experiment, not a fully converged
GRPO model or established broad generalization. The earlier full-pass SFT
result remains a stronger historical reference; it was measured separately
on L40S and is not a paired H100 comparator here.

Exercise: compare the new mixed-group fraction with Stage 16. If all groups
fail, explain why increasing the learning rate cannot help an optimizer that
never receives a nonzero group-relative advantage.

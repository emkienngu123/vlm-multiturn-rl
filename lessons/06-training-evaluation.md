# Stage 6 — Train, resume, evaluate, and diagnose

The GRPO arithmetic is unchanged. This lesson makes an experiment recoverable
and its results interpretable: persist training state at round boundaries,
resume in another container, and compare base, SFT, and SFT → GRPO on identical
held-out boards.

Read [stage6.py](../src/vlm_multiturn_rl/stage6.py) for orchestration,
[checkpoints.py](../src/vlm_multiturn_rl/checkpoints.py) for saved state, and
[reporting.py](../src/vlm_multiturn_rl/reporting.py) for metrics and curves.

## 1. Begin with the CPU exercise

```bash
uv run vlm-multiturn-rl --stage 6
uv run python -m unittest discover -s tests -v
```

The exercise reports that `1/4 = 25%` success has a Wilson 95% interval of about
`4.6%–69.9%`. It also lists the state a training checkpoint needs. It uses only
the standard library; plotting, model execution, and checkpoint tensor loading
remain remote. Matplotlib is pinned in the Linux GPU environment.

This wide interval is a useful warning about our benchmark's scale. It does not
measure variability across training seeds, renderer styles, or other task
families. The fixed maze generator is a diagnostic distribution, not a claim
about general visual-agent capability.

Wilson intervals also avoid the zero-width uncertainty that a naive normal
approximation would report for zero successes. A result of `0/4` is weak evidence
about the task distribution, not proof that the true success probability is zero.

## 2. Run a bounded experiment

```bash
# Base/SFT evaluation, two GRPO rounds, checkpoints, final evaluation and plots.
uv run modal run modal_app.py --task train-grid --updates 2 --episodes 4 \
  --sft-run-id stage4-20261017T175637Z-7415e007
```

Substitute your own completed Stage 4 run ID if needed. Omitting the flag selects
the latest completed local GPU SFT report, as in Stage 5. Stage 6 starts from
that SFT adapter; a Stage 5 adapter archive does not contain Adam state and is
not an exact-resume checkpoint.

The defaults are unchanged: one L40S, BF16 base, FP32 language-side LoRA, four
sampled episodes per training board, eight actions per episode, full histories,
and one update per nonconstant group. AdamW uses `1e-5`, zero weight decay, and
norm clipping at 1. The reference keeps KL coefficient zero. Training can drift
away from the SFT policy; the held-out replays help reveal that drift.

`--updates` counts **collection rounds**, not successful optimizer steps.
Equal-return rounds are logged and skipped. Their position still matters for
the next board and RNG seeds. Restoring only an optimizer-step counter could
repeat a previously collected board/seed combination.

The job timeout is 30 minutes. Defaults save a checkpoint after every round,
evaluate every five rounds, and also evaluate at the final requested round.
`--checkpoint-every` and `--eval-every` set those intervals for a new experiment.
The final round always gets a checkpoint and evaluation.

After the smoke run passes, the bounded learning experiment is:

```bash
uv run modal run modal_app.py --task train-grid --updates 20 --episodes 4 \
  --sft-run-id stage4-20261017T175637Z-7415e007
```

Use the measured collection time to estimate cost before increasing the round
count. Long histories require repeated visual processing. Accumulating gradients
over actions limits retained graphs, but it does not shrink one long-history
forward pass. The saved throughput includes generation and behavior scoring,
so it is not a pure decoder-speed benchmark.

## 3. Why an adapter alone cannot resume Adam

Adam maintains running gradient moments:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
\]

Its step also depends on the counter through bias correction. Reloading adapter
weights while resetting these moments changes the next update, even with the
same next batch. PyTorch's [checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
therefore includes optimizer state when training will continue.

Our round checkpoint stores:

| State | Reason |
| --- | --- |
| Adapter weights/configuration | Reconstruct the policy on the pinned base model |
| Adam moments, counters, and parameter groups | Preserve the update rule |
| Ordered trainable parameter names | Prevent silently assigning moments to different parameters |
| Completed collection round and actual update count | Continue the board schedule and distinguish skipped groups |
| Python, NumPy, CPU Torch, and CUDA RNG states | Restore stochastic state after model/optimizer construction |
| Original SFT receipt, runtime, configuration, and source hashes | Reject incompatible continuation |
| Histories, image observations, group records, evaluations | Preserve the evidence behind the final comparison and curves |

There is no scheduler or gradient scaler in this BF16 reference. If you add
either later, their states become part of the resume contract too.

Checkpoints are written only after an entire group and update have finished.
No partially accumulated gradients, open environments, or stale behavior group
need to be restored. Every new round collects fresh trajectories under the
current policy. This simplifies recovery substantially compared with resuming
halfway through a rollout or optimizer update.

`training-state.pt` contains tensors and built-in Python containers, loaded with
`weights_only=True`. NumPy generator state is converted to plain lists before
saving. The restore path checks adapter identity, parameter ordering, every
Adam step counter, optimizer-state content, all RNG states, and the next random
draws. The original base weights remain in the model cache.

## 4. Persist a checkpoint before the job exits

Each checkpoint has its own checksummed manifest:

```text
/runs/STAGE6_RUN_ID/checkpoints/round-0002/
  checkpoint.json
  adapter.zip
  training-state.pt
  history.json
  images.zip
  manifest.json
```

The manifest is written last and the run volume is explicitly committed after
that boundary. A later container reloads the volume before reading it. This
follows [Modal Volume persistence semantics](https://modal.com/docs/guide/volumes).
The cache volume stores the shared base model; the run volume stores changing
experiment state.

An interrupted job may lack a final report manifest while still having usable
round checkpoints. Resume searches completed checkpoint manifests directly and
ignores directories whose writes never reached a manifest. If a completed
checkpoint's checksum is wrong, loading fails instead of silently accepting it.

Each snapshot includes the history so far. Saving every round is convenient
for these short jobs but duplicates storage. For longer exercises, choose a
larger interval before launching the experiment and balance storage against
the amount of work you are willing to repeat after interruption.

## 5. Verify a fresh-container resume

```bash
# Add a one-round continuation reference for this correctness test.
uv run modal run modal_app.py --task train-grid --updates 2 --episodes 4 \
  --sft-run-id stage4-20261017T175637Z-7415e007 --verify-resume

# Replace STAGE6_RUN_ID with the ID printed by that job.
# Resume from round 2 and extend the TOTAL target to four rounds.
uv run modal run modal_app.py --task train-grid --updates 4 --episodes 4 \
  --resume-run-id STAGE6_RUN_ID --resume-round 2
```

`--verify-resume` is an optional validation cost. The live model executes one
additional next round, records its generated token IDs, episode returns, and
resulting adapter/optimizer identities, then restores the original training
boundary. This validation branch supplies a reference for uninterrupted
continuation. Its rollouts are separate from the experiment's training metrics.

The second job uses a new single-use container. It restores the checkpoint,
collects the next real round, and compares that round with the saved live
reference. A match covers actual action sampling, outcomes, and the resulting
weights and optimizer state. Merely loading an adapter would not pass this
check after nonzero Adam updates.

RNG restoration happens **after** constructing the new model and optimizer,
because construction can consume random draws. The collector also derives
trajectory seeds from the global collection-round index. Resuming must continue
that index rather than resetting it to zero.

If `--resume-round` is omitted, the latest completed checkpoint is selected.
The original SFT ID is read from the checkpoint. The new run receives a unique
ID, preserving its parent. `--updates 4` means a total of four rounds, not four
additional rounds. The requested target must exceed the restored round.

This tutorial rejects changed training/evaluation settings, runtime versions,
GPU type, source hashes, or dependency pins during an exact resume. To change
the experiment, start a new run. Bitwise continuation observed with this pinned
stack does not promise equivalence across different hardware or kernels.

On resume, the saved base/SFT evaluation episodes and intermediate evaluations
are retained with their images. The same immutable evaluation contract applies
to the newly evaluated GRPO checkpoint. This avoids paying to regenerate those
baselines on every continuation.

## 6. Read the comparison report

Open `artifacts/STAGE6_RUN_ID/report.html`. It links to the complete replay and
has exportable Matplotlib SVG/PNG curves and CSV data.

Base, SFT, and SFT → GRPO share:

- The exact held-out layouts and eight-action limit.
- The full-history input protocol, image resolution, and 64-token action budget.
- The tool prompt, parser, greedy decoding, model revision, and processor.

The base evaluation disables the adapter on the same frozen backbone using
the [PEFT adapter context](https://huggingface.co/docs/peft/package_reference/peft_model).
It therefore uses the Stage 4/5 tool format. Stage 3's direction-word baseline
is not substituted into this paired comparison.

For each policy, the report includes success rate and Wilson interval, mean
return, path efficiency, episode length, invalid calls, blocked moves, and
episode-return variance. Per-board outcomes distinguish a new success from a
lost old success. Two policies can both score `1/4` while solving different
boards; their aggregate rates alone hide this change.

The paired success difference is

\[
\Delta=\frac{\#(\mathrm{GRPO\ success,\ SFT\ failure})
-\#(\mathrm{SFT\ success,\ GRPO\ failure})}{n}.
\]

Wilson intervals summarize each marginal success estimate; they are not a
paired-difference significance test. With four boards, conclusions are mostly
diagnostic. Repeatedly choosing settings using these boards also turns them
into a validation set; reserve additional untouched tasks for stronger claims.

To broaden a new experiment, `--episodes 16` extends the deterministic held-out
prefix and `--seed 7` changes training sampling. Policy seeds do not change the
dataset seed. Use several independent training seeds and clearly label changes
to the task set. Increasing `--train-layouts 16` mixes the original SFT mazes
with additional training mazes, which is a distinct data experiment.

## 7. Interpret curves and diagnose failures

The four plot panels show sampled training return, held-out greedy success,
within-group population reward variance, and gradient norm before clipping.
The initial held-out point is SFT at round zero. Subsequent points evaluate the
current GRPO adapter at fixed intervals and at each job's final round.

Each training point uses a group from one board; board difficulty can change
between points. A rising line can reflect easier sampled tasks rather than
policy improvement. Success uncertainty is wide in a small evaluation, and
per-update losses near zero remain normal for freshly collected groups.

| Pattern | Useful diagnosis |
| --- | --- |
| Return `−0.08`, variance zero, no optimizer steps | Sparse-reward exploration is stalled; more nominal rounds may add no learning. |
| High return, variance zero | All sampled paths have equal return; relative advantages still vanish. |
| Valid calls but repeated wall collisions | Protocol learning is stronger than visual navigation; inspect newest observations and action choices. |
| Invalid sampled calls despite valid greedy calls | Sampling explores low-probability syntax errors; compare decoding settings before blaming masking. |
| Training reward rises while held-out success falls | Check memorization, task difficulty, and drift from the SFT policy. |
| More turns with similar success | Look for detours/repetition; path efficiency and action cost reveal them. |
| Large gradient norms | Inspect variance and sampled tokens; the logged norm is before clipping. |
| Resume repeats an old board/seed | A round index was reset or confused with optimizer-step count. |
| Adapter matches but resumed update differs | Check Adam state, parameter ordering, RNG restoration, and code/runtime contract. |
| OOM or timeout | Use the last committed checkpoint. Reduce resolution in a new experiment; changing it is not exact resume. |

The environment reward never pays for invalid calls, collisions, or looping.
All unsuccessful eight-action trajectories still tie at `−0.08`. A successful
trajectory's inefficient steps can receive positive episode-level advantages,
so good metrics do not replace action-by-action inspection. Reward exploitation
is diagnosed against the actual task contract and replayed behavior.

For example, if malformed calls accidentally ended episodes with zero reward,
an agent could prefer that exit to a costly eight-step failure. Spending a turn
for every invalid call closes that particular loophole. Other environments may
have shortcuts that maximize their implemented reward without solving the
intended task; inspect successful episodes as carefully as failed ones.

Use `environment.json` for current-job runtime and peak allocated GPU memory.
`learning-curves.csv` records collection/scoring time, update time, generated-token
throughput, and per-round memory. Resumed curves retain earlier job rows; the
current job's wall-clock measurement is not a cumulative runtime estimate.

## 8. Replay and inspect saved artifacts

```bash
uv run vlm-multiturn-rl --stage 6 --replay artifacts/STAGE6_RUN_ID
```

CPU replay verifies report checksums, transitions, histories, group advantages,
round continuity, policy lineage, unique group membership, and the paired
comparison recomputed from recorded outcomes. Tensor/optimizer/RNG restoration
is checked remotely, where the GPU libraries are installed.

`checkpoint.zip` is a local copy of the final complete training checkpoint.
`adapter.zip` contains inference weights. Intermediate checkpoints remain in
the Modal run volume. `training.json` records actual optimizer steps, skipped
groups, parent-state restoration, and continuation-reference verification.

## Results from the verified runs

All **45 CPU tests** passed. The initial job and the fresh-container continuation
both completed on one L40S using the pinned 20-update SFT checkpoint.

| Run | Completed rounds | Actual Adam updates | Equal-return skips | Replay episodes |
| --- | --- | --- | --- | --- |
| Initial | 2 | 1 | 1 | 20 |
| Resumed | 4 total | 2 total | 2 total | 32 |

The new container restored all 144 trainable tensors' Adam states and the saved
step count of 1. Adapter identity, optimizer contents, RNG state, and next random
draws matched. The next real round reproduced the live continuation's exact
generated token IDs and returns `[-0.08, -0.08, 0.97, 0.97]`; its post-update
adapter and optimizer identities also matched. This exercised a nonzero resumed
Adam update, rather than only an all-failure skip.

| Policy after four rounds | Success | Wilson 95% interval | Mean return | Mean actions | Blocked moves |
| --- | --- | --- | --- | --- | --- |
| Base | 0/4 | 0%–49.0% | −0.080 | 8.0 | 0% |
| SFT | 1/4 | 4.6%–69.9% | 0.185 | 6.5 | 65.4% |
| SFT → GRPO | 1/4 | 4.6%–69.9% | 0.185 | 6.5 | 65.4% |

SFT and GRPO succeeded on the same board: zero GRPO-only wins and zero SFT-only
wins. Base calls were all invalid; SFT and final GRPO greedy calls were valid.
Training mean returns were `−0.08, 0.4375, 0.445, −0.08` across four different
initial training boards. Their changes did not produce a held-out gain. These
are correctness and resume experiments, not evidence of a successful learning
curve or broad generalization.

The initial recorded compute phase took 201.4 seconds, including its additional
continuation validation. The resumed phase took 129.7 seconds for two new rounds
and evaluation. Peak allocated GPU memory was about 18.38 GiB in both jobs.
Collection plus behavior scoring ran at about 9.6–11.2 generated tokens/second;
non-skipped gradient updates took about 8 seconds. These measurements exclude
image building, CPU preparation, scheduling, and final report packaging.

The final checkpoint archive is about 20 MiB. It is much smaller than the base
weights, but includes the optimizer and history needed for continuation.

Local reports:

- Initial run: `artifacts/stage6-20261017T182728Z-4540fc41/report.html`
- Resumed run: `artifacts/stage6-20261017T183150Z-adce0ae9/report.html`

Both plots were visually inspected, and the final report's 32 recorded
trajectories and paired comparison passed CPU replay.

## Exercise

Find a skipped group and a nonconstant group in `history.json`. Explain why
both advance the collection-round counter while only one advances Adam.
Inspect the restored Adam count in `training.json` and compare it with the
parent run. Then use `comparison.json` to count GRPO-only wins and SFT-only
wins yourself. Does a change in training return establish a change in held-out
success? Support your answer with the paired boards and replay.

The next lesson connects this tested environment and evaluation contract to
TRL, while checking that its multimodal tool responses and loss masks preserve
the same interaction semantics.

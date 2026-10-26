# Stage 14 — Train on the form and recovery demonstrations

Stage 13 established that the browser executed requested coordinates accurately,
while most saved model clicks missed the form controls. It produced 853 teacher
targets: 700 clean actions and 153 corrective actions. This stage continues the
existing browser SFT adapter on those targets and measures free-running
development performance. It does not perform GRPO.

Execution status: completed and GPU-validated in
[the Modal app](https://modal.com/apps/kiendepzai3/main/ap-E3QADRxBfpOc4LvtVWnzrM).
One pass improved paired development success from **2/16 to 16/16**.
The final saved checkpoint is `stage14-20261019T102008Z-1255a893`.
All 91 CPU tests, 853 target-mask inspections, and 32 evaluation-episode
replays passed. The final-test pool remains closed.

## Commands

```bash
# Validate source receipts and inspect the full-pass training budget locally.
uv run vlm-multiturn-rl --stage 14 \
  --dataset-run-id stage13-20261019T091807Z-60a38afd \
  --sft-run-id stage9-20261019T041931Z-88adc041 --epochs 1

uv run python -m unittest discover -s tests -v

# A two-update smoke checkpoint, continued training chunks, then paired dev eval.
uv run modal run modal_app.py --task browser-form-sft \
  --dataset-run-id stage13-20261019T091807Z-60a38afd \
  --sft-run-id stage9-20261019T041931Z-88adc041 --epochs 1

# Inspect a downloaded training checkpoint or development-evaluation shard.
uv run vlm-multiturn-rl --stage 14 --replay artifacts/STAGE14_RUN_ID

# Continue after an interrupted workflow using its last completed checkpoint.
# If the pass is already complete, this runs the paired development evaluation.
uv run modal run modal_app.py --task browser-form-sft \
  --dataset-run-id stage13-20261019T091807Z-60a38afd \
  --sft-run-id stage9-20261019T041931Z-88adc041 --epochs 1 \
  --resume-run-id COMPLETED_STAGE14_CHECKPOINT

# Optional: regenerate plots on CPU, then a local comparison page.
uv run modal run lessons/tools/plot_form_sft.py \
  --workflow artifacts/STAGE14_WORKFLOW.json
uv run python lessons/tools/summarize_form_sft.py \
  artifacts/STAGE14_WORKFLOW.json
```

The local reader requires the original dataset and Stage 9 adapter artifact
directories alongside the Stage 14 directory. It validates their receipts as
well as the new files. The Modal volume already holds those sources.

Each training chunk and evaluation shard gets its own run ID and manifest.
The launcher writes `artifacts/stage14-...-workflow.json`, listing checkpoints,
evaluation shards, and the final comparison. A failed chunk does not overwrite
its parent. Continue from the last completed checkpoint; partial accumulation
inside an interrupted chunk is not saved.

## 1. Continue the adapter, rather than initialize another LoRA

The source is the validated 100-update Stage 9 adapter. We load it with
`PeftModel.from_pretrained(..., is_trainable=True)`; inference-only loading is
the default, so omitting that flag can silently leave no trainable adapter.
See the [PEFT loading interface](https://huggingface.co/docs/peft/package_reference/peft_model#peft.PeftModel.from_pretrained).

The frozen base model and vision encoder remain BF16. Only language attention
`q_proj` and `v_proj` LoRA parameters train, in FP32, with rank 8, alpha 16,
and zero adapter dropout. We verify the trainable names and dtypes before an
optimizer step. This reuses the source adapter's configuration and weights;
it does not call `get_peft_model` to create a new adapter.

The first job starts a fresh AdamW optimizer for this new curriculum. Stage 9
saved an inference adapter, not the Adam state needed to resume its old run.
Later Stage 14 jobs restore the actual Stage 14 Adam moments and counters.
We use a constant learning rate of `5e-5`, zero weight decay, and gradient norm
clipping at 1. These are explicit experiment settings, not an established
optimum for this task. The learning rate is lower than Stage 9's `2e-4` because
we are continuing an already adapted policy; this is a conservative starting
choice, not a learning-rate ablation.

## 2. Define duration in data passes

For N=853 targets, accumulation B=4, and one pass, the update count is

\[
U=\lceil N/B\rceil=214.
\]

A deterministic local RNG shuffles all target indices for each epoch. The
schedule records the exact indices for every update and includes the last
one-example batch. There is no modulo cycling over a small prefix, and no
dropping the final example. The loss is divided by the actual number of
examples in the update, including that final partial batch.

The default is one pass, chosen before opening development instances. The
configuration also supports two passes as a separate predefined experiment.
Continuation requires identical configuration and source code; changing
`--epochs` during continuation is rejected. A longer experiment needs its own
configuration and comparison, rather than silently redefining this run.

## 3. Audit the current assistant target

We reuse the existing processor and `sft_batch` implementation. For every
example it renders the full multimodal history, checks image placeholder/grid
alignment, tokenizes the current teacher action, and appends EOS explicitly.
All 853 examples receive a saved mask inspection, including their ordered
image paths, grids, input IDs, labels, and supervised span.

Only the current teacher action plus EOS has labels other than `-100`. This
matters especially for recovery examples: earlier assistant messages include
the injected wrong actions. Those messages remain context but are not targets.
The data reader checks each target against Stage 13's exact verified example;
supervising all assistant-role messages would be incorrect here.
Tokenizing oracle text defines an SFT target. This differs from RL scoring,
where we must preserve the actual IDs sampled by the policy rather than
re-tokenize its decoded output.

Let A be the current action token positions and h the fixed public history:

\[
\mathcal L_{\rm action}
=-\frac{1}{|A|}\sum_{k\in A}\log\pi_\theta(a_k\mid h,a_{<k}).
\]

We average these action losses within an optimizer update. This gives each
action equal weight within its batch; it is not an episode-normalized loss.
Long recovery episodes therefore contribute more examples than short episodes.
Observations influence hidden states through attention without becoming
supervised output tokens.

Before real training, explicit cross-entropy checks verify the model's causal
label shift on a short and the longest target. A longest-history backward
probe checks finite nonzero LoRA gradients without stepping the optimizer.
Its gradients are cleared, RNG is restored, and adapter weights must remain
unchanged. The first two real optimizer updates are a saved smoke checkpoint,
not a second training run that gets discarded before starting over.

## 4. Bound memory and job duration

Each microbatch is one multimodal action history. Four microbatches accumulate
before AdamW steps. Non-reentrant gradient checkpointing recomputes activations
during backward instead of retaining all of them, and training sets
`use_cache=False`. This is ordinary teacher-forced forward/backward, not the
cached generated-token GRPO scorer from Stage 12. See
[PyTorch checkpointing](https://docs.pytorch.org/docs/2.14/checkpoint.html).

Each job uses one L40S, two CPUs, 24 GiB host memory, and a 30-minute timeout.
After the two-update smoke job, the launcher requests at most 64 updates per
job and reduces that cap using measured update/backward time. This limits
exposure to the timeout; it is not a guarantee against every infrastructure
interruption. Model loading and the initial full mask audit are additional
overhead. Training and browser evaluation use separate jobs.

## 5. Save enough state to continue

Every completed chunk contains:

- `adapter.zip`: the updated adapter; frozen base weights remain in the cache.
- `training-state.pt`: Adam state and Python/NumPy/Torch/CUDA RNG state.
- `checkpoint.json`: parameter ordering, cumulative update cursor, fingerprints,
  state digests, and checksums.
- `training.json`: cumulative exact example schedule, losses, gradient norms,
  dataset exposure, and restore proofs.
- `mask-inspection.json`, source receipts, configuration, and environment metrics.

Before the next chunk trains, we verify the dataset and initial-adapter receipts,
configuration, source code, adapter fingerprint, Adam ordering/moments/counters,
and RNG state. Each chunk also unloads and reloads its newly saved adapter and
reconstructs Adam to verify the saved state. The next remote job repeats that
check in a new container, providing actual cross-job continuation evidence.

Loading an adapter alone would restart Adam and would not resume this experiment.
Similarly, restarting the shuffle at index zero would repeat examples. The
cumulative cursor points into the saved deterministic schedule.

## 6. Evaluate the completed saved adapter

Only after the pass finishes do evaluation jobs open the 16 development seeds
60000–60015 declared in Stage 13. They evaluate the unchanged source adapter and
the completed-pass adapter with identical greedy settings and a 12-action cap.
Each job covers eight cases. No oracle demonstrations or development actions
enter training loss. The saved trained adapter is loaded from its completed
checkpoint, rather than taken from an in-memory training object.

The aggregator requires all 16 cases exactly once per policy, matching initial
instructions and pixel hashes. It also checks that decoding settings agree
and each policy's fingerprint is the same across its two shards. Actual actions
are replayed in fresh browsers. Metrics include episode success, mean actions,
invalid actions, execution errors, runtime, and peak allocated CUDA memory.

This is development evaluation on one familiar template. It does not establish
general web competence or prove that recovery improved separately from ordinary
form filling. The 32 final-test seeds 70000–70031 stay unopened. If development
results guide another change, they remain development results.

## Frequent problems

| Symptom | Check |
| --- | --- |
| No gradients after loading an adapter | Set `is_trainable=True`; inspect LoRA names/dtypes |
| Training teaches injected mistakes | Mask historical assistant turns; target only the current verified teacher action |
| Images appear reordered | Compare message image paths against the actual image tensors/grids |
| "100 updates" processes little data | Report exact examples processed and dataset passes |
| Loss improves but browser success does not | Inspect free-running development replays, including missed focus and repeated clicks |
| Continuation starts over | Restore Adam/RNG and the cumulative example cursor, not only adapter weights |
| GPU memory grows | Disable generation cache during training; release each microbatch graph; check gradient checkpointing |
| A job reaches its timeout | Continue from the last completed chunk with identical configuration/code |
| A standalone remote helper cannot import the local entry point | Keep its runtime self-contained or explicitly include its dependencies; the plotting helper uses a small separate CPU image |

## Measured result

| Paired development metric | Stage 9 source adapter | After one form-data pass |
| --- | ---: | ---: |
| Successful episodes | **2/16 (12.5%)** | **16/16 (100%)** |
| Mean actions | 10.1875 | **7.0** |
| Invalid actions | 9/163 (5.52%) | **0/112** |
| Execution errors | 0 | 0 |
| Success-rate 95% Wilson interval | 3.5%–36.0% | 80.6%–100% |

Fourteen cases changed from failure to success; the two source successes stayed
successful. Every trained-policy episode finished in seven actions. There were
no exact initial instruction-and-image pairs shared between the training and
development instances, although both pools use the same familiar form template
and its six field orders.

| Training/validation measurement | Result |
| --- | --- |
| Optimizer updates / optimized examples | **214 / 853**, exactly one pass |
| Checkpoint update cursors | 2, 66, 130, 194, 214 |
| Cross-job adapter / Adam / RNG restores | All four passed |
| Saved-state unload/reload verification | All five checkpoints passed |
| Longest audited history | Ten images; 2,762 prompt tokens |
| Longest-history backward probe | Finite nonzero LoRA gradients; weights unchanged |
| Trainable parameters | 1,843,200, all language-side LoRA |
| CPU tests / mask inspections / evaluation replays | **91 / 853 / 32**, all passed |
| Training GPU-function time | **1103.5 seconds / 18.4 minutes** |
| Training plus paired evaluation GPU-function time | **1913.6 seconds / 31.9 minutes**, one L40S sequentially |
| Peak allocated CUDA memory | **13.68 GiB** |
| Final adapter ZIP | **6.49 MiB** |
| Adapter + Adam/RNG + checkpoint metadata | **20.70 MiB** per checkpoint |
| Additional full mask audit | **27.81 MiB** per checkpoint directory |
| Recorded core-source hashes | All training and evaluation manifests match |
| Final-test instances opened | **0/32** |

GPU-function time excludes container startup, initial cache preparation, and
the CPU-only reporting work. The base weights remain in the shared cache.
The duplicated full mask audit makes each downloaded training directory larger
than its 20.7 MiB training-state core; it is validation metadata, not extra
model weights. Host memory was requested at 24 GiB; host peak was not measured.

In `forms13-dev-000`, the request was for paranoid movies directed by Turner
from 1977. The source policy filled all three fields correctly, then repeatedly
clicked `(117.8, 272.1)`, below Submit, until the action cap. The trained adapter
filled the fields and clicked `(120.0, 257.25)`, completing the task. This is a
concrete correction in generated coordinates under the same environment.
It does not establish a separate improvement in recovery after a new mistake.

This experiment supports learning the form workflow with relevant supervision.
It changes both demonstration coverage and subsequent training exposure, so it
does not isolate longer duration as the sole cause. It uses one training seed,
16 development instances, and one UI template. These results demonstrate SFT
improvement; GRPO learning on these harder tasks remains a separate experiment.

The next RL decision should use training-only exploration groups. A policy that
already succeeds on every rollout has zero group-relative reward variance.
Check for useful mixed groups on broader training instances or harder workflows
before spending compute on additional GRPO updates. Keep the 32 final cases
closed while choosing that procedure.

Open the [comparison page and episode replays](../artifacts/stage14-20261019T100154Z-4b297337-workflow.html),
[training curves](../artifacts/stage14-20261019T100154Z-4b297337-workflow-training.png),
[machine-readable comparison](../artifacts/stage14-20261019T100154Z-4b297337-workflow-comparison.json),
or [final checkpoint metadata](../artifacts/stage14-20261019T102008Z-1255a893/checkpoint.json).
The workflow JSON records all run IDs; local helpers regenerate the convenience
comparison page and plots from the verified core bundles. Training curves show
different teacher-action batches and are not held-out success measurements.

## Exercise

Inspect the final one-example update. Explain why its loss is divided by one,
which target would be omitted by dropping the tail, and why adapter-only loading
plus restarting the shuffle would not reproduce the continued experiment.

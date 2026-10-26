# Stage 9 — Teach the browser action protocol with LoRA SFT

The Stage 8 base model solved **0/6 held-out browser instances**, with **100% invalid
actions**. It emitted Python-style calls such as `click(x=100, y=80)` instead of
the literal JSON tool call our environment accepts. Before sparse-reward RL, we
need a policy that can express an executable action.

This lesson trains a **fresh browser adapter on the pinned base model**, using the
successful training demonstrations from Stage 8. The grid adapter is not used.
We reuse the audited language-side LoRA trainer from Stage 4, while supplying
browser histories, browser screenshots, and browser targets.

## 1. Run the lesson

```bash
# Standard library only: validate the latest matching completed local CPU preflight.
uv run vlm-multiturn-rl --stage 9 --episodes 2
uv run python -m unittest discover -s tests -v

# Two-update correctness experiment on one L40S.
# Omit --run-id to select the latest matching local Stage 8 CPU preflight.
uv run modal run modal_app.py --task browser-sft --episodes 2 --updates 2 \
  --run-id stage8-20261019T034403Z-a72e8d1c

# Predeclared, bounded learning exercise; starts fresh from the base model.
uv run modal run modal_app.py --task browser-sft --episodes 2 --updates 20 \
  --run-id stage8-20261019T034403Z-a72e8d1c

# Inspect a completed run without PyTorch, Pillow, Chromium, or model inference.
uv run vlm-multiturn-rl --stage 9 --replay artifacts/STAGE9_RUN_ID
```

For a controlled 100-update comparison with full training-fit diagnostics, see
[Stage 9A: SFT duration](09a-sft-duration.md). Its optional `--sft-run-id` loads
an earlier browser adapter for diagnosis; the new training run still starts
from the base model.

On another workspace, first run `browser-check --episodes 2` from Lesson 8 and
substitute its CPU run ID. `--run-id` here selects the **source preflight**; the
SFT command always generates a new Stage 9 run ID. The source must use the same
seed, pixel limits, generation budget, and number of instances. Pass `--episodes 2`
explicitly: the shared Modal command's default is four instances per task/split.

The command validates the source on a remote CPU before requesting the GPU. It
checks checksums, replayed transition records, train/evaluation separation, the
benchmark revision, prompt, and dependency lock. GPU execution has a 30-minute
timeout. Expect printed update losses and gradient norms, then a local
`artifacts/STAGE9_RUN_ID/report.html` with every browser screenshot/action.

## 2. What is the training dataset?

There are three familiar task types: `click-test`, `enter-text`, and `login-user`.
With two training seeds per task, their successful oracle episodes contain
respectively one, three, and five actions:

\[
N=2(1+3+5)=18\quad\text{supervised action examples}.
\]

Each example consists of the **history before one action** and that action's
literal target text. The last login example includes five ordered screenshots,
four previous actions, and four public feedback messages. It does not include
the result of the current target action.

The privileged oracle used DOM geometry to produce click coordinates. The policy
receives only screenshots, task instructions, and public feedback. Case IDs,
seed values, DOM selectors, and oracle state are bookkeeping; they do not enter
the model prompt. A requested username/password is intentionally part of the
public task instruction, since the policy needs to type it.

Training seeds are 0 and 1; held-out seeds are 10000 and 10001 for each task.
The six evaluation instances test new instances of **familiar task templates**.
They do not establish transfer to unseen websites or task types.

## 3. Multi-turn teacher forcing and the loss

Let \(h_t=(o_0,a_0,\ldots,o_t)\) be the available history and let the teacher's
current action contain token IDs \(y_{t,1},\ldots,y_{t,L_t}\), including EOS.
The per-example loss is

\[
\ell_t(\theta)=-\frac{1}{L_t}\sum_{j=1}^{L_t}
\log\pi_\theta(y_{t,j}\mid h_t,y_{t,<j}).
\]

We average these action-level losses over two accumulated microbatches per
optimizer update. This gives each sampled action equal weight; a longer JSON
action does not automatically receive more weight because it has more tokens.
The deterministic shuffled schedule cycles through the 18 examples. Two updates
process four examples; 20 updates process 40. Neither is a large dataset.

The input includes the complete history, but labels are:

```text
input:  [system, instruction, images, prior actions/feedback, assistant prefix | target action, EOS]
labels: [-100,   -100,        -100,   -100,                  -100             | target IDs,   EOS]
```

`-100` excludes a position from cross-entropy. **Prior assistant actions are also
masked**: each becomes a target once in its own example. Attention masks and loss
masks have different jobs. The model may attend to an observation without being
trained to predict it.

For the first target token at input position \(P\), the predicting logit is at
position \(P-1\). The trainer compares the model's internal shifted loss with an
explicit cross-entropy calculation over these positions. This catches an easy
off-by-one mistake that can produce a finite but incorrect training loss.

Teacher forcing conditions on successful oracle histories. At deployment, the
model conditions on its own actions. A misplaced click can yield a screenshot
absent from these demonstrations, and subsequent mistakes compound. SFT can
establish protocol competence while still failing at visual grounding, planning,
or recovery. This distribution shift is a reason to add online RL later.

## 4. How the code enforces this

Read these functions in order:

1. `stage9.validate_source`: verifies a completed CPU preflight and the unchanged
   browser contract. It reconstructs the training targets from successful
   training-oracle episodes, excluding evaluation and model-generated actions.
2. `stage9.BrowserSFTActor.sft_batch`: reuses `HistoryVLM`'s processor, ordered
   multimodal history, exact teacher target tokenization, and assistant-only mask.
   Its small metadata translation maps `case_id` into the shared trainer's old
   inspection interface; it does not add that ID to policy messages.
3. `stage9.validate_mask`: checks every example, including the five-image login
   history. `mask-inspection.json` saves input IDs, readable tokens, labels, image
   order, image grids, and the exclusive supervised span `[start, end)`.
4. `stage4.train_adapter`: freezes the entire base model, including vision;
   targets only language attention `q_proj` and `v_proj` with rank-8 LoRA; uses
   BF16 base weights, FP32 adapter parameters, gradient checkpointing, AdamW at
   `2e-4`, microbatch one, accumulation two, and gradient clipping at norm one.
5. The same trainer verifies zero initial LoRA effect, finite losses/gradients,
   absence of frozen-parameter gradients, actual adapter changes, and the causal
   shift. It saves, unloads, and reloads the adapter on the same frozen base,
   requiring exact saved parameter equality and matching probe logits.
6. `stage9.train_remote`: evaluates the fresh base first, trains, then evaluates
   the **reloaded** adapter on the same held-out cases with the same greedy
   decoding settings. Initial instructions and raw screenshot hashes must match
   between policies. SFT action sequences are also re-executed in fresh browsers
   without model generation to verify their recorded outcomes.

Image processing and the 4096-token context guard are unchanged from Stage 8.
We preserve the original screenshot coordinate system when the processor resizes
images internally. Loss inspection does not silently discard older screenshots.

The initial/final/reloaded probe losses use **one fixed training action**, which
makes save/reload validation cheap. They are not held-out validation loss or a
generalization metric.

An actual inspected example, `login-user-train-001` turn 5, has five image grids
of `[1, 22, 36]`. With the model's 2×2 spatial merge, these contribute
\(5\times22\times36/4=990\) visual tokens. Its prompt has 1455 tokens and its
current action plus EOS has 27. Only positions `[1455, 1482)` are supervised;
the logit at position 1454 predicts the first target token. Earlier actions and
all five images remain available for attention.

## 5. Artifacts and checkpoint size

The run volume and downloaded local directory contain:

| Artifact | Purpose |
| --- | --- |
| `cpu-source.json` | Source manifest/lock hashes and exact training case IDs |
| `demonstrations.json`, `cases.json` | Training targets and the fixed task split |
| `training.json` | Losses, gradient norms, update times, adapter sizes, reload checks |
| `mask-inspection.json` | Every training token mask and ordered image list |
| `adapter.zip` | PEFT adapter configuration and LoRA weights |
| `episodes.json`, `images.zip`, `report.html` | Oracle, random, base, and SFT browser replays |
| `metrics.json`, `comparison.json` | Held-out per-task metrics and paired outcomes |
| `environment.json`, `manifest.json` | Runtime, decoding, memory/time, source and artifact hashes |

The rank-8 adapter has about **1.84 million trainable parameters**, approximately
**7.4 MB** of FP32 weights before ZIP compression. The frozen base weights stay
in the persistent model cache, roughly **7.5 GB**. Screenshots and verbose token
inspection also occupy space but are not model parameters.

This stage saves an **inference adapter**, without Adam moments or RNG state.
It validates adapter reload; it does not resume the optimizer trajectory. The
20-update command starts fresh rather than continuing the two-update run.
Lesson 6 shows the additional state needed for exact training continuation.

## 6. Read the measurements correctly

Compare success counts, invalid-action rate, execution errors, and episode
length for base versus SFT. Inspect `comparison.json` for which individual cases
changed. Oracle/random metrics use only their held-out episodes; training
demonstrations and invalid-action controls never enter evaluation averages.

A reduction in malformed JSON is progress in protocol compliance. A valid click
on the wrong pixel is a grounding error. Correctly typing into the wrong focused
field is a state-tracking error. Repeating a valid click until the cap is reached
is a behavior failure even though the invalid-action rate is zero.

Report all six cases and both run lengths. A short training run can lower training
loss without improving task success. Do not select the best checkpoint using
these same six evaluation cases and then present them as untouched test data.

### Observed two-update correctness run

Run `stage9-20261019T035306Z-3386e822` trained on the 18 targets from
`stage8-20261019T034403Z-a72e8d1c` and verified 31 saved episodes. Base and SFT
both solved **0/6** held-out instances and both had **100% invalid actions**.
The SFT model still emitted Python-style `click(x=100,y=110)` calls. All mask,
causal-loss, frozen-parameter, adapter-change, save/reload, and action-replay
checks passed. Correct training execution has not yet produced executable policy
behavior in this short run.

The fixed training probe loss changed from 2.20285 to 2.03132; reloading reproduced
2.03132 exactly. The GPU function took 149.5 seconds and peaked at 10.55 GiB of
allocated CUDA memory, excluding image build/CPU preparation. LoRA weights were
7,393,888 bytes; the complete compressed adapter ZIP was 6,513,911 bytes.
Open its local `report.html` and `mask-inspection.json` for the recorded evidence.

### Observed 20-update learning run

Run `stage9-20261019T035637Z-5b8a7079` used the same source, training settings,
held-out instances, and decoding. It started from the base model, as declared
before evaluation. Both runs are retained rather than selecting a checkpoint
using the evaluation scores.

| Policy | Held-out success | Invalid actions | Mean actions per episode |
| --- | --- | --- | --- |
| Base | 0/6 | 100% | 8.00 |
| SFT, 2 updates | 0/6 | 100% | 8.00 |
| SFT, 20 updates | **4/6** | **0%** | 5.33 |
| Privileged oracle | 6/6 | 0% | 3.00 |
| Random pixel clicks | 1/6 | 0% | 5.17 |

The 20-update adapter solved both `enter-text` cases in three actions and both
`login-user` cases in five. The requested held-out names/passwords were copied
correctly into the focused fields. Both `click-test` cases failed: the model
repeated the valid action `click(100,80)` eight times, with no correction. The
public JSON protocol was preserved throughout; raw outputs were not repaired.

This separates **protocol learning** from **reliable visual grounding/recovery**.
Success on text entry and login does not establish arbitrary coordinate grounding:
these are familiar templates, and the explicit randomized-button control still
fails. There are only two held-out instances per task. Treat 4/6 as a small
diagnostic result rather than a precise estimate of general web-agent performance.

The fixed training probe loss changed from 2.20285 to 0.46702 and reproduced
0.46702 after reload. The 20 optimizer updates took 20.6 seconds in total; the
whole GPU function, including loading, paired browser evaluation, replay checks,
and report generation, took 141.1 seconds. Peak allocated CUDA memory was
10.56 GiB. Frozen base weights remained cached; LoRA weights were 7,393,888 bytes
and the compressed adapter ZIP was 6,808,676 bytes. Wall time includes additional
Modal initialization/CPU preparation outside that GPU measurement.

Both runs verified all 31 saved episodes, all 18 training masks, exact saved
adapter parameter reload, probe-logit agreement, and fresh-browser SFT action
re-execution. Their recorded source hashes matched the Python code at completion,
before the subsequent Stage 9A duration-study extension.

Open [the 20-update replay report](../artifacts/stage9-20261019T035637Z-5b8a7079/report.html)
and [its paired comparison](../artifacts/stage9-20261019T035637Z-5b8a7079/comparison.json).
The next browser RL lesson can start from this adapter while retaining the same
strict action and held-out evaluation contracts.

The subsequent [100-update duration study](09a-sft-duration.md) improved training
success from 3/6 to 6/6 and evaluation success from 4/6 to 6/6. That run,
`stage9-20261019T041931Z-88adc041`, is the newer candidate for browser exploration
tests; the 20-update results above remain preserved for comparison.

## 7. Frequent problems

| Symptom | Likely cause and next inspection |
| --- | --- |
| Source rejected before GPU allocation | Wrong `--episodes`, changed lock/protocol/assets, incomplete report, or a superseded Stage 8 report; run a matching CPU preflight |
| Lower loss, still malformed calls | Too little protocol exposure; inspect exact raw output and EOS/truncation rather than repairing model text in the evaluator |
| Valid JSON but unsuccessful task | Open screenshot replays; separate coordinate errors, focus errors, copied text errors, and repeated actions |
| Only the final turn appears supervised | Expected within each example; verify every previous action also has its own example |
| Training memory grows with turn number | Each observation contributes visual tokens and history; inspect image grids/context length, reduce bounded pixel cost consistently across source and training |
| Adapter reload differs | Base revision, vocabulary/processor, precision, adapter configuration, or incomplete files changed; the job fails rather than reporting an unverified adapter |
| Fresh-browser action replay differs | Browser/assets/timing or reset behavior changed; inspect the exact saved action sequence and pinned runtime |
| Cleanup reports a running event loop | Synchronous Playwright's global runtime must be stopped before Modal volume commit/container cleanup; the remote entry point does this in `finally` |

## 8. Exercise

Open the final `login-user` training mask. Find the preceding assistant action,
the newest image tokens, and the current target's first token. Confirm that only
the current action and EOS have labels other than `-100`, and identify the logit
position that predicts the first target token.

Then classify each held-out SFT failure as protocol, grounding, focus/text entry,
or recovery failure using its replay. Propose additional **training** trajectories
for one failure class without adding the held-out instance to training. The next
stage will use online browser interaction to expose the policy to its own states.

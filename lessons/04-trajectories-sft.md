# Stage 4 — Full trajectories and an assistant-only SFT bridge

By the end of this lesson, you can inspect a complete changing-image conversation,
replay its actions, identify every supervised token, and save/reload a language-side
LoRA adapter. We now train a policy, using solver actions as labels. Rewards are
recorded for the next lesson; they do not enter this SFT objective.

Start with the CPU exercise, then read the masking section before running the GPU
job. The default is a two-update correctness experiment. Success on held-out boards
is measured, rather than assumed from a falling training loss.

## 1. Run the CPU trajectory exercise

```bash
uv run vlm-multiturn-rl --stage 4
uv run python -m unittest discover -s tests -v
```

With the default dataset, expect eight successful solver demonstrations containing
28 action examples. The command prints a unique `artifacts/stage4-...` directory.
Open its `replay.html`, `demonstrations.json`, and `examples.json` together.

```bash
# Replace RUN_ID with the directory printed above (or a completed GPU run).
uv run vlm-multiturn-rl --stage 4 --replay artifacts/RUN_ID
```

This verifies checksums, deterministic transitions, stopping conditions, message
histories, observation references, and the correspondence between examples and
solver actions. CPU reports contain SVG inspection images in `images.zip`. GPU
reports contain the actual PNG observations sent to the processor. CPU execution
uses neither PyTorch nor Pillow and performs no tokenization or model inference.

The first eight training mazes and the held-out evaluation mazes are unchanged
from Stage 3. Only training layouts generate demonstrations. Increasing
`--train-layouts` preserves the existing prefix of that training set.

## 2. One turn is an action followed by a real transition

From now on, an action is one explicit tool call:

```text
<tool_call>{"name":"move","arguments":{"direction":"left"}}</tool_call>
```

The accepted directions are `up`, `down`, `left`, and `right`. Whitespace around
the call or inside JSON is allowed. Extra text, duplicate JSON keys, extra
arguments, two calls, and unknown directions are invalid. The parser returns
`None` for invalid output; the environment spends a turn and keeps its position.
There is no hidden repair, retry, or solver fallback.

The conversation grows as follows:

```text
system: board legend, rules, and tool-call contract
user:   image o₀ + "Choose your next move."
assistant: move call a₀
tool:   feedback r₀, blocked/valid/end flags + image o₁
assistant: move call a₁
tool:   feedback r₁, blocked/valid/end flags + image o₂
...
```

`collect_trajectory()` in [trajectories.py](../src/vlm_multiturn_rl/trajectories.py)
owns the environment. The learned actor receives only a copy of the public
messages and image objects, in placeholder order. Private coordinates and solver
state go into diagnostics, outside the model messages. A new conversation is
created for every episode; neither messages nor environment state are shared
between episodes.

The pinned Qwen template renders role names and image placeholders but does not
implement a native tool schema here. We explicitly teach this text protocol in
the system instruction. We do not pretend that passing a Python callable enables
tool execution: our collector parses the output and executes `GridWorld.step()`.
The trainer integration in Stage 7 will handle a separate native tool interface.

Each recorded transition contains:

| Field | Purpose |
| --- | --- |
| `messages` | Exact public history supplied before this action |
| `generation.prompt_ids` | Actual processed input IDs for model-generated actions |
| `generation.generated_ids` | Exact continuation IDs, including a terminal EOS when emitted |
| `text`, `action` | Raw decoded call and the parser's result |
| `generation.image_paths`, `image_grid_thw` | Image order and processor metadata |
| `before`, `after` | Private state for replay and diagnosis |
| `reward`, `valid_action`, `blocked`, `terminated`, `truncated` | Transition outcome |

The episode also stores the full final conversation, all observation references,
return, success, efficiency, and end reason. The final tool image is saved even
when the episode ends immediately after that action.

Solver demonstrations have text labels and `generated_ids: null`: a solver did
not generate language-model tokens. SFT tokenizes those labels once. Model
rollouts retain their actual token IDs; later policy scoring must use those IDs,
because decoding and re-encoding text need not recover the sampled sequence.

## 3. Why take a short SFT step before sparse-reward RL?

Behavioral cloning fits the solver's action distribution on solver histories:

\[
\mathcal L_{\rm BC}(\theta)
=\mathbb E_{(h_t,a_t^*)\sim D}
\left[-\frac{1}{|a_t^*|}\sum_{k=1}^{|a_t^*|}
\log\pi_\theta(a_{t,k}^*\mid h_t,a_{t,<k}^*)\right].
\]

Here an action is a sequence of tool-call tokens, including its end-of-turn
token. The history contains all earlier observations, actions, and feedback.
The solver supplies an optimal next direction; it never supplies an explanation
or its private path to the policy. This bridge can teach reliable syntax and
some visual action behavior before GRPO needs successful exploration.

There is a distribution mismatch: demonstrations visit solver states, while
the trained agent visits states induced by its own mistakes. One wrong move can
lead to histories absent from the demonstrations. Consequently, low teacher-forced
loss does not establish closed-loop success. This is the imitation-learning
distribution-shift problem studied in the [DAgger paper](https://arxiv.org/abs/1011.0686).
We use ordinary behavioral cloning here; we do not implement DAgger in this stage.

The board is fully observable. Supplying history exercises the multi-image
pipeline but does not make this task a memory benchmark. Old images also add
compute: each new forward pass reprocesses the full history. Stage 8 will make
memory useful through partial observability.

## 4. Inspect the loss mask and causal alignment

Qwen's pinned chat template does not contain generation markers for automatic
assistant masking. Instead, `HistoryVLM.sft_batch()` builds each example explicitly:

1. Render the public history with an assistant generation prefix.
2. Process that prefix with its ordered images, preserving all processor outputs.
3. Append the known target action IDs and the tokenizer's assistant EOS ID.
4. Give every prefix token label `-100`, and every target token its own ID.
5. Append attention-mask ones and multimodal-type zeros for the text target.

For a tiny sequence:

```text
input:   [system, image, old-action, tool-result, assistant-prefix, LEFT, EOS]
labels:  [ -100,  -100,    -100,        -100,          -100,      LEFT, EOS]
```

Only the current action is supervised. Previous assistant actions remain useful
context and become targets in their own examples. Every demonstration action is
therefore supervised once, without repeatedly scoring earlier actions in each
longer history. Environment text, image tokens, role headers, and the system
instruction are never prediction targets.

For input IDs \(x_0,\ldots,x_{N-1}\), a causal model's logits at position \(j-1\)
predict \(x_j\). If the prefix length is \(P\), the first supervised token is
label \(P\), predicted by logit \(P-1\). Transformers shifts the unshifted labels
internally. Shifting them again would train the wrong predictions. The GPU probe
compares the returned loss against cross-entropy computed explicitly from
`logits[P-1:-1]` and `labels[P:]`.

The mask defines:

\[
\mathcal L=-\frac{\sum_{j=1}^{N-1}m_j\log p_\theta(x_j\mid x_{<j})}
{\sum_{j=1}^{N-1}m_j},\qquad m_j=\mathbf 1[j\ge P].
\]

Attention masks and loss masks serve different purposes. The model attends to
the observations even though we do not train it to predict them. This stage uses
unpacked examples with microbatch size one, so there is no padding to supervise.

Open `mask-inspection.json` after a GPU run. It contains the longest solver
history's input IDs, token strings, labels, ordered image references, grids,
and half-open supervised span `[start, end)`. Check that all preceding labels
are `-100`, image tokens are excluded, and the last target is EOS.

## 5. Train and reload a small LoRA adapter

```bash
# One L40S; base evaluation, two optimizer updates, save/reload, SFT evaluation.
uv run modal run modal_app.py --task sft-grid --updates 2 --episodes 4
```

The CPU cache-preparation job still precedes GPU allocation. The remote training
function has a 30-minute timeout and makes no model downloads. PEFT is pinned to
`0.21.2`; the complete Linux dependency lock remains in `requirements-gpu.txt`.

LoRA replaces a selected linear map's effective weight with

\[
W_{\rm effective}=W_0+\frac{\alpha}{r}BA,
\qquad A\in\mathbb R^{r\times d_{in}},\ B\in\mathbb R^{d_{out}\times r}.
\]

Only the small matrices are optimized. We use rank 8, alpha 16, zero adapter
dropout, and explicit language-model `q_proj`/`v_proj` module names. The base
model, vision encoder, and visual merger stay frozen. Matching names under
`language_model` prevents a broad attention-module selector from adapting the
vision transformer. See the [PEFT LoRA reference](https://huggingface.co/docs/peft/package_reference/lora)
for configuration and initialization behavior.

The implementation verifies the initial adapter is a no-op, then uses AdamW at
`2e-4`, no weight decay, gradient clipping at norm 1, BF16 base weights, FP32
adapter weights, and non-reentrant gradient checkpointing. Two sequential
microbatches accumulate into each update. Each microbatch loss averages its
current action's tokens; the accumulated loss averages those action losses.
Longer demonstration episodes contribute more action examples.

The sampler shuffles the 28 examples deterministically and cycles through that
order. Two updates consume only four examples. They check gradients, parameter
changes, and artifact correctness; they are not a full pass over this tiny dataset.
Frozen weights still participate in forward/backward computation needed to
train the language adapters, so LoRA does not eliminate activation memory costs.

The adapter is saved, unloaded without merging, and reloaded from disk on the
same base model. State tensors must match exactly and fixed-probe logits must
match within tolerance. The SFT evaluation uses this reloaded adapter. Saving
and reloading follow the [PEFT model API](https://huggingface.co/docs/peft/package_reference/peft_model).
`adapter.zip` includes its configuration and safetensors weights; it needs the
same pinned base model and processor. It is an inference checkpoint. Optimizer
state and exact training resume are reserved for Stage 6.

## 6. Read the evidence before increasing the run

The job evaluates base and SFT on identical held-out layouts, full-history input,
greedy decoding, image resolution, tool prompt, and 64-token action budget. The
`oracle_train` row describes the demonstration source and is labeled separately.
The Stage 3 VLM used a direction-word response and current-image input, so its
success rate is not a paired estimate of this adapter's effect.

Inspect these artifacts:

| Artifact | What it answers |
| --- | --- |
| `training.json` | Did gradients exist, adapters change, and save/reload agree? |
| `metrics.json` | Did base/SFT succeed, waste moves, or produce invalid calls? |
| `episodes.json`, `replay.html` | What did the agent actually do after each observation? |
| `mask-inspection.json` | Which exact tokens contributed to the loss? |
| `images.zip`, `examples.json` | Which ordered observations conditioned each target? |
| `environment.json`, `manifest.json` | GPU/runtime/memory, versions, hashes, and source provenance |

The fixed training-probe loss before/after measures one demonstrated action.
Per-update losses use different examples and need not decrease monotonically.
Neither measure proves visual planning or held-out improvement. Four evaluation
boards are a diagnostic sample; inspect each episode and use more layouts and
seeds in Stage 6 before drawing generalization conclusions.

### Results from the verified runs

These runs used seed 42, the default eight training mazes, four held-out mazes,
448² images, and the pinned model revision. Both passed causal-loss checks,
frozen-parameter checks, exact adapter-state reload, probe-logit agreement, and
CPU replay of all 16 trajectories (eight demonstrations plus eight evaluations).
All 30 CPU tests passed.

| Policy/checkpoint | Held-out success | Invalid calls | Blocked moves | Fixed training-probe loss |
| --- | --- | --- | --- | --- |
| Base | 0/4 | 100% | 0% | 0.3798 |
| SFT, 2 updates | 1/4 | 53.8% | 19.2% | 0.1741 |
| SFT, 20 updates | 1/4 | 0% | 65.4% | 0.0867 |

Rates are per attempted action. The two SFT checkpoints succeeded on different
boards. The base repeatedly emitted `move up`, which violates the tool format.
After two updates, many calls were missing `</tool_call>` despite ending with
EOS. After 20 updates, syntax was valid on all 26 attempted actions, but the agent
often repeated blocked moves. This establishes working protocol training and
exposes weak navigation; it does not establish robust visual planning.

The two-update run took 50.5 seconds inside the GPU function and peaked at
9.58 GiB of allocated GPU memory. The 20-update run took 80.6 seconds and peaked
at 11.90 GiB; its optimizer loop took 35.7 seconds. These exclude image building,
CPU cache preparation, GPU scheduling, and final report packaging. LoRA trained
1,843,200 parameters. The longest inspected example contained six distinct board
images: 1,988 prefix tokens were ignored and exactly 14 action/EOS tokens were
supervised.

Local results:

- Two updates: `artifacts/stage4-20261017T175412Z-4a120b09/`
- Twenty updates: `artifacts/stage4-20261017T175637Z-7415e007/`

Each directory has its own configuration, source hashes, adapter, and replay.
The 20-update adapter is a useful starting checkpoint for the next lesson, with
its remaining failures visible in the saved evaluation.

Once the two-update run passes, a bounded learning exercise is:

```bash
uv run modal run modal_app.py --task sft-grid --updates 20 --episodes 4
```

Keep the evaluation contract fixed. Increasing `--train-layouts 16` is a separate
experiment because it changes the data. If memory is tight, try
`--max-pixels 50176` (224²), noting that this changes both training and evaluation
resolution. Gradient accumulation increases the effective batch; it does not
reduce the memory of one long-history forward pass. The 4096-token guard rejects
oversized inputs rather than silently truncating images or their placeholders.

## 7. Common failures and what they mean

| Symptom | Likely cause and useful check |
| --- | --- |
| Valid JSON, wrong movement | Syntax competence does not establish visual competence; inspect board/action pairs. |
| All actions repeat the prompt's example direction | Prompt bias or weak task learning; compare target directions and closed-loop behavior. |
| Calls end halfway through JSON | Generation budget too small; check `finish_reason`, which distinguishes EOS from length. |
| Tool delimiters disappear | Blanket special-token stripping; we remove only a terminal EOS for parsing and retain raw IDs. |
| New board looks unchanged | Could be a blocked/invalid move; compare flags before diagnosing stale observations. |
| Movement changes but the model uses an old board | Check ordered image paths and the newest tool message; histories are replay-validated. |
| Loss falls but rollout reward does not improve | Teacher-forcing shift, syntax dominating token loss, or overfitting; inspect success and invalid actions separately. |
| Image count/grid mismatch | Placeholder/image ordering or dropped multimodal metadata; encoding fails explicitly. |
| Zero trainable parameters or vision parameters trainable | Target-module paths changed; inspect names before changing the strict language-only selector. |
| Adapter loads but behavior differs | Base revision, processor, decoding, or adapter state mismatch; use the saved metadata and probe checks. |
| PEFT warns about a missing base config while offline | Automatic embedding-export detection may try a Hub lookup. We explicitly disable embedding export because embeddings are frozen and vocabulary is unchanged. |
| A job stops before writing a report | This stage writes a completed report at the end; use Modal logs to find the exception. Resume is a later stage. |

An eight-action truncation means collection ran out of turns; reaching the goal
is termination. These flags remain distinct in tool feedback and saved records.
Malformed actions are retained verbatim in the next history and consume budget.
We do not silently train on environment feedback to make the loss look better.

## Exercise

Open a successful solver episode with at least three actions. For its third SFT
example, identify all three image references, the two previous assistant calls,
and the current target. Find their token spans in a GPU mask inspection (use the
longest episode if necessary). Explain why the previous calls are visible but
have ignored labels. Then compare one failed base rollout with its SFT counterpart:
did SFT change call syntax, the selected direction, or both? Support your answer
with the saved actions rather than training loss alone.

The next lesson replaces supervised targets with freshly sampled trajectory
groups and complete-episode returns. The same history/observation contract and
exact generated-token records will make the GRPO loss auditable. Continue with
[Lesson 05](05-multiturn-grpo.md).

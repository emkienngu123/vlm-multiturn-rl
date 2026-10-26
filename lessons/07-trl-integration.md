# Stage 7 — Move the working environment into TRL

We now give TRL responsibility for grouping, tool execution, token masks, reward
normalization, the clipped loss, gradient accumulation, and optimization. The
grid dynamics, reward, strict action protocol, images, and evaluator stay ours.
Read Stage 5 first: a trainer is easier to debug when you know what its loss
should compute.

This is a **one-update integration experiment**, on one L40S, starting from a
completed Stage 4 adapter. It includes a scripted changing-image check before
sampling learned actions. That scripted check establishes plumbing correctness;
it is not evidence that the VLM can solve the task.

## Run it

```bash
# No local Torch, Pillow, datasets, or TRL required.
uv run vlm-multiturn-rl --stage 7
uv run python -m unittest discover -s tests -v

# Latest completed local GPU SFT run, or specify --sft-run-id STAGE4_RUN_ID.
uv run modal run modal_app.py --task trl-grid

# CPU replay/checksums after the GPU job finishes.
uv run vlm-multiturn-rl --stage 7 --replay artifacts/STAGE7_RUN_ID
```

The CPU exercise follows a two-action solver path and prints `return: 0.98`,
`steps: 2`, `observations: 3`. It exercises the wrapper and replay contract, not
native TRL. The remote job uses a separate dependency image with **TRL 1.16.0,
Transformers 5.20.1, PEFT 0.21.2, and datasets 5.1.0**. The full Linux resolution
is in `requirements-trl.txt`; earlier lessons keep `requirements-gpu.txt`.
Model weights still load offline from the shared cache volume. The GPU timeout
is 30 minutes. Stage 7 always takes one optimizer step; `--updates` belongs to
the earlier training exercises and does not extend this integration experiment.

Inspect `report.html`, `replay.html`, `integration.json`, and `training.json` in
the unique run directory. The final `adapter.zip` can initialize another
experiment, but this stage does not save Adam/RNG state for exact continuation;
Stage 6 remains the resume tutorial.

## 1. Read the boundary before the trainer

Start in `src/vlm_multiturn_rl/trl_environment.py`. The only public methods on
`GridToolEnvironment` are:

| Method | Responsibility |
| --- | --- |
| `reset(layout, **kwargs)` | Construct a fresh grid from private dataset metadata; return the initial image and instruction |
| `move(direction)` | Apply exactly one action and return feedback plus a PIL image |
| `get_reward()` | Return the sum of all action rewards, exactly once as a reward source |

TRL discovers public methods as tools, except the reserved `reset` and
`get_reward` hooks. Helpers therefore start with `_`. Factories produce separate
instances for concurrent rollouts. TRL **pools and resets** instances across
batches; a fresh factory call is not guaranteed for every episode. Reset clears
pending decisions, history, transitions, images, terminal flags, and image-owner
lookup entries. Shared artifact storage is safe because episode keys are unique;
the mutable `GridWorld` is never shared.

The dataset row contains a conversational prompt and a serialized private layout.
The layout goes to `reset`; it is never inserted into the chat template. The
returned content blocks are appended to the empty final user message, preserving
our original image-before-text order. Subsequent tool results contain public
reward/validity/termination feedback and the next image. Coordinates, maze IDs,
and shortest paths remain outside the model prompt.

The reward is still

\[
R(\tau)=\sum_{t=1}^{T}[-0.01+\mathbb{1}(\text{goal reached at }t)],\qquad T\leq8.
\]

We supply no second reward function. The environment's `get_reward` already
becomes a trainer reward source; adding the same reward again would double it.

## 2. A tool API does not define episode semantics

The pinned native loop continues only when its parser reports a tool call. It
does not ask our environment whether the episode has ended. Directly plugging
in a grid would therefore change the benchmark: malformed text could end an
episode early, and reaching the goal could lead to further calls.

`stage7.install_grid_protocol` and `make_trainer_class` provide three explicit
adaptations:

1. The tokenizer response parser delegates to the same strict `parse_move` used
   in Stages 4–6. Every generated response dispatches exactly one environment
   turn. Malformed output dispatches the internal `__invalid__` event, which
   applies `env.step(None)`: no movement, a turn consumed, and reward −0.01.
   The raw malformed output stays in history and the policy loss. This does
   not supply the agent with a valid action or a solver correction.
2. `_generate_single_turn` invokes native generation one history at a time,
   with 64 generated tokens per action, full-softmax sampling, no KV cache,
   and the exact ordered images. It verifies the native packed prompt IDs
   against the preceding actual prompt, actual sampled IDs, and an independent
   encoding of the new public tool response.
3. After a terminal tool result, the generation hook returns an empty
   continuation for that row. The native loop has already appended and masked
   the final image; it then exits without another action. The final unused
   assistant header belongs to the masked tool suffix.

The Jinja template preserves our role/image formatting and can also render
structured `tool_calls`. Qwen's checkpoint template silently omits structured
tool calls, so using it unchanged fails TRL's capability check. Our template is
prefix preserving: appending a tool result cannot rewrite preceding assistant
tokens. We attach the strict parser only inside the remote process. This is an
explicit protocol adapter, rather than relying on the trainer's permissive
default parser.

The subclass does **not** replace `_tool_call_loop`, `_compute_loss`, or reward
normalization. It overrides private generation/audit hooks, so the installed
release is pinned and hashes of relevant native methods are saved. An upgrade
requires rerunning these checks, not merely changing a version number.

An observed integration bug makes this distinction concrete: an action reached
64 tokens without generating EOS. Re-rendering its conversation through a chat
template inserted an end-of-turn token that the policy had never generated.
Native TRL instead appends its tool suffix directly to the actual sampled IDs.
Our check initially rejected that valid native history. The corrected check
preserves the exact IDs and verifies the appended suffix independently. A remote
CPU regression checks both EOS-ended and length-cutoff continuations. Image
features come from the same processor, while input IDs and multimodal token-type
IDs follow the actual packed stream. Consequently, native and reference rollout
formatting can differ after an action cutoff; the controlled loss comparison
does not claim their complete sampling procedures are identical.

## 3. Grouping and the masked objective

One dataset row is repeated four times. Its layout is training index 1, chosen
before sampling as a small integration task. Four independently sampled
trajectories share the initial board, but own separate environment state. Native
generation consumes one seeded random stream sequentially; it is not the same
per-trajectory RNG schedule used by the reference trainer. We verify that LoRA
weights remain unchanged throughout collection.

Our configuration sets `num_generations=4`, training microbatch size 1,
gradient accumulation 4, `num_iterations=1`, and one optimizer step. Thus the
group is collected together, then each trajectory is scored once before the
single accumulated update. There is no buffer of older groups reused after
that update.

TRL normally suppresses generated image-placeholder tokens for VLMs. We override
that generation setting with `suppress_tokens=None`, keeping full-softmax
sampling consistent with full-softmax scoring. If a visual placeholder is
actually sampled, the job fails explicitly because an action cannot provide new
image features. This avoids silently renormalizing the sampling distribution.

TRL packs the whole trajectory into a completion containing assistant tokens
and tool responses. Let \(c_k\) be its attention-valid completion mask and
\(u_k\) the native tool mask. The policy loss uses

\[
m_k=c_ku_k,\qquad
L=-\frac1G\sum_i\frac{1}{\sum_k m_{ik}}
\sum_k m_{ik}\min(r_{ik}A_i,
\operatorname{clip}(r_{ik},1-\epsilon,1+\epsilon)A_i).
\]

We explicitly select `loss_type="grpo"`, token-level importance ratios,
\(\epsilon=0.2\), and `beta=0`. The pinned release defaults to a different
length normalization (`dapo`), so copying defaults would not reproduce Stage 5's
mean of per-trajectory token means.

`Stage7Config` inherits the earlier model/environment settings for checkpoint
compatibility. Its inherited `advantage_epsilon` and `sampling_logp_tolerance`
are reference-trainer fields, not native TRL controls. Use the effective trainer
arguments in `integration.json["args"]` and the table below to interpret this run.

Tool feedback, images, formatting, and the next assistant prefix have mask zero.
They remain visible through attention: excluding an observation from the loss
does not exclude it from the history. Each generated action, including EOS,
has mask one. Audits recover the assistant tokens from the native packed stream
and require exact equality with concatenated sampled token IDs. They also
verify ordered image bytes, image-token counts, scoring image grids, terminal
states, and padding masks. A silent image drop or early context cutoff fails
the run.

With one iteration and no intervening update, TRL can use current log
probabilities detached as the old policy. Numerically the ratios start at one;
their derivatives are nonzero. A near-zero scalar loss does not imply no
learning. Check gradient norms and the adapter fingerprint.

## 4. Compare matching mathematics, not library names

`objective_comparison()` calls the installed **actual** `GRPOTrainer._compute_loss`
on tiny controlled log probabilities, substituting only the expensive model
forward. It compares scalar loss and gradients against `clipped_token_terms`
and the standard-library `reference_loss`. The rows have unequal supervised
lengths, positive and negative advantages, clipped and unclipped ratios,
masked tool positions, and padding. It also checks zero-advantage loss and
gradients. This verifies the surrogate and mask under matching settings; it
does not claim the entire trainers are interchangeable.

There is one intentional remaining numerical difference:

| Quantity | Reference Stages 5–6 | Native pinned TRL |
| --- | --- | --- |
| Group standard deviation | Population, divisor \(G\) | Sample, divisor \(G-1\) |
| Advantage denominator | \(\sigma+10^{-6}\) | \(s+10^{-4}\) |
| Equal-return group | Skip optimizer step | Zero advantages; trainer still counts an optimizer step |

For four nonconstant returns, \(s=\sqrt{4/3}\,\sigma\). Advantage magnitudes
therefore differ even with the same trajectories. The objective comparison
supplies identical advantages so this distinction cannot hide a masking or
clipping error. Reports retain actual native advantages and raw returns. Weight
decay is zero; a fresh all-equal group produces no policy update. With a
previously populated Adam optimizer, zero gradients need not erase momentum,
which is another reason not to equate an optimizer counter with learning.

## 5. Interpret the integration checkpoint

The GPU run first sends four copies of a known two-move training board through
the **native tool loop**, using scripted solver tokens. Each row must retain
three different board images in order, and all tool/image tokens must be masked.
These episodes are labeled `oracle_integration_check` and excluded from learned
policy evaluation.

Next it samples four model trajectories, computes native group advantages, and
performs exactly one accumulated update. `training.json` records reward
variance, gradient norm, and whether adapter bytes changed. Base, SFT, and the
post-TRL adapter use the same existing evaluator, held-out layouts, resolution,
full-history contract, and greedy decoding. This small check can establish
correct execution; four evaluation boards and one update cannot establish a
learning trend or equivalence to the reference training run.

### Observed run: seed 42

Run `stage7-20261017T185759Z-cb6adabe` passed the native two-turn image/mask
checks, surrogate/gradient comparison, and CPU replay of all 20 episodes. Its
four learned rollouts all returned −0.08. Native advantages and gradient norm
were zero, and the adapter fingerprint did not change despite one counted
optimizer step. Base solved 0/4 held-out boards; SFT and post-TRL both solved
1/4. The GPU function took about 118 seconds and peaked at 8.98 GiB allocated
memory, excluding CPU preparation and final report packaging.

This is a completed integration check with no learning signal. Preserve it when
trying another seed; discarding zero-variance groups from the report would hide
the exploration problem. A second seed is an additional correctness check, not
an estimate of training improvement.

### Observed run: seed 43

The additional run `stage7-20261017T190115Z-79c14c34` used the same SFT source,
training board, reward, and evaluator, with only `--seed 43` changed. Its group
returns were `[0.98, -0.08, 0.97, -0.08]`. Native advantages were approximately
`[0.8741, -0.8659, 0.8577, -0.8659]`; gradient norm was 0.17975 and the saved
adapter fingerprint changed after one optimizer step. The surrogate loss was
near zero initially, illustrating why loss magnitude alone is a poor update
diagnostic. All 20 trajectories replayed successfully.

Base/SFT/post-TRL success remained 0/4, 1/4, 1/4. The GPU function took about
91 seconds and peaked at 8.98 GiB allocated memory. This verifies a real native
TRL adapter update, without evidence of improved held-out performance. Both
the zero-gradient seed and the nonzero-gradient seed are part of the result.

```bash
uv run modal run modal_app.py --task trl-grid --seed 43
```

## Frequent failures

| Symptom | Cause and check |
| --- | --- |
| Template rejected or tool calls disappear | A text-only template can ignore `tool_calls`; capability and prefix checks must pass with an image |
| Second turn still sees the first board | Return actual image content blocks; inspect per-turn prompt IDs, image hashes, and `image_grid_thw` |
| Rollouts influence each other | Shared grid object or incomplete reset; use one instance per concurrent rollout and unique artifact keys |
| Invalid outputs earn a shorter, cheaper episode | Default text fallback ends the native tool loop; retain the explicit invalid-turn dispatcher |
| Native tool failures are zero despite malformed actions | The dispatcher successfully applied invalid turns; inspect each episode's `invalid_actions`, not just tool execution exceptions |
| Agent keeps acting after success | Native loop lacks a grid done hook; terminal continuation must be empty |
| Group all succeeds/fails; no changed adapter | Inspect raw returns and advantages; do not manufacture reward variance |
| Completion budget exhausted after a few actions | `max_completion_length` includes feedback **and image tokens**, unlike per-action `max_new_tokens` |
| Context check differs after a 64-token action | Re-rendering can invent EOS after a length cutoff; condition and score on the actual packed IDs |
| Model receives no images during scoring | Some trainer overflow fallbacks discard multimodal fields; strict image-count checks reject this |
| OOM despite sequential inference | Training still scores full packed histories; use microbatch 1 and non-reentrant gradient checkpointing; lower image resolution only in an explicitly new, paired experiment |
| Most rows reported as non-EOS completions | Terminal rows end on a masked tool suffix/unused assistant header; use environment termination metrics, not that text-generation diagnostic |
| Old Stage 6 exact resume rejected after this lesson | The source receipt deliberately changed; resume with the original code/dependency snapshot or start a new experiment |

## Exercise

Open `integration.json`. For one model rollout, recover assistant IDs by selecting
`completion_ids[k]` where `tool_mask[k] == 1`. Locate a turn in `history.json` and
verify its exact generated sequence appears there. Count observations, turns,
assistant tokens, and masked tool tokens separately. Then compute the native
and reference advantages from the same four saved returns. Explain why their
magnitudes differ while the controlled surrogate comparison still passes.

Do not increase the run length until you can explain an equal-return group's
gradient and an invalid action's effect on both episode return and loss mask.
The grid remains fully observable and does not establish a memory benefit.
Following the selected browser-agent direction, Stage 8 moves to MiniWoB++;
the planned partial-observability/credit-assignment ablation is deferred.

## Primary references

- [TRL GRPO: tools, environments, and VLMs](https://huggingface.co/docs/trl/grpo_trainer)
  describes the public interfaces. APIs are version sensitive.
- [Pinned TRL 1.16.0 release](https://pypi.org/project/trl/1.16.0/) supplies the
  installed implementation; `integration.json` hashes the native methods we check.
- [TRL implementation](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)
  is useful for comparison, but `main` may differ from the pinned wheel.

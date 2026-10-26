# Lesson 02 — From an image to a VLM action

In Stage 1 we established software compatibility and persistent storage. Now we
load `Qwen/Qwen2.5-VL-3B-Instruct` and inspect a complete inference call. We keep
the same model revision, dependency lock, and Modal volumes.

The goal is to understand the interface we will eventually optimize with RL:
**image + instruction → multimodal inputs → generated action tokens**.
There is no reward, environment transition, or parameter update in this lesson.

## 1. Start with an experiment we can interpret

The experiment uses two 4×4 boards. A blue circle is the agent and a green square
is the goal. In one board the goal is immediately to the agent's left; in the other
it is immediately to its right. The agent stays in the same cell.

Both calls use exactly the same instruction:

> Move exactly one cell to reach the goal. Reply with one direction word.

The complete prompt also defines the colors and image-relative directions. See
`PROMPT` in `src/vlm_multiturn_rl/vision.py`; the abbreviated text above is only
for explanation. The model never receives fixture names, coordinates, or expected
answers as text. Those labels are saved afterward for our inspection.

This control matters: a changed text prompt could explain a changed answer even
if the image were ignored. Here we verify that the rendered prompts and input
token IDs match, while the processed pixel tensors differ.

We use greedy decoding and fresh generation calls. This keeps sampling variation
out of the image comparison. The model may still answer incorrectly; that is a
behavior result, separate from whether the input pipeline works.

## 2. Run it and follow the code

From the project root:

```bash
uv run vlm-multiturn-rl --stage 2
uv run python -m unittest discover -s tests -v
uv run modal run modal_app.py --task inspect-vlm
```

The first two commands use the local CPU and need neither local PyTorch nor
Pillow. Rendering and model computation happen remotely.

The Modal entry point first calls `prepare_stage2_model`, a CPU job that downloads
the pinned snapshot, including its safetensors weights. After that job commits
the cache, `inspect_stage2_vlm` requests an L40S and loads from that cache with
Hub offline mode enabled. Both jobs have a 15-minute timeout. Later runs reuse
the downloaded weights.

We explicitly load the whole model on `cuda:0` in BF16, use PyTorch SDPA attention,
call `model.eval()`, and generate inside `torch.inference_mode()`. This avoids
autograd storage and training behavior during this diagnostic. We are not using
LoRA yet: first establish that the unmodified model's inference path is correct.

Read the code in this order:

1. `vision.py`: board fixtures, common prompt, renderer, action parser, token slicing.
2. `stage2.py`: snapshot preparation, input inspection, and generation.
3. `modal_app.py`: the CPU/GPU resource boundary and downloading the artifacts.

## 3. What changes from an ordinary LLM input?

A useful schematic is

\[
z_v = M_\phi(E_\phi(R(I))),
\]

where \(R\) resizes/normalizes the image, \(E_\phi\) encodes visual patches, and
\(M_\phi\) merges/projects the visual features for the language decoder.
These operations are already pretrained; this lesson does not train them.

```text
Messages ── chat template ── text + visual placeholder
                                    │
Image ── resize/normalize/patch ──────┤ processor
                                    v
       input_ids + pixel_values + image_grid_thw + masks
                                    │
                       visual encoder and merging
                                    │
                 decoder consumes visual + text embeddings
                                    │
                        next-token generation → action
```

An image placeholder is not a textual description of the pixels. `input_ids`
identifies visual positions, while `pixel_values` supplies image data and
`image_grid_thw` describes its patch layout. This explains how identical token
IDs can yield different outputs when the images differ. The processor owns the
chat/image preparation contract. See the
[multimodal template guide](https://huggingface.co/docs/transformers/chat_templating_multimodal).

Qwen2.5-VL also uses multimodal rotary position encoding to represent visual
spatial/temporal structure, rather than treating every visual location like an
ordinary text position. We preserve the processor's grid metadata and let the
model construct the required positions. Its dynamic-resolution visual encoder
is described in the [Qwen2.5-VL report](https://arxiv.org/abs/2502.13923).

Do not reconstruct a model input from the decoded text alone: it loses the
image payload and may lose information about visual positions.

## 4. Why the template and processor are separate steps here

The implementation deliberately makes both steps visible:

```python
prompt = processor.apply_chat_template(
    messages(), tokenize=False, add_generation_prompt=True,
)
inputs = processor(
    text=[prompt], images=[image], return_tensors="pt",
    add_special_tokens=False,
)
```

The messages include one image placeholder; the image list contains its actual
image, in the same order. `add_generation_prompt=True` opens the assistant turn.
`add_special_tokens=False` prevents the second step from adding another set of
special tokens to already formatted text. The saved prompt shows the role and
vision markers before image placeholders are expanded to their required length.

A single `processor.apply_chat_template(..., tokenize=True, ...)` call can also
prepare a supported multimodal conversation. We expose the two steps so you can
inspect where text formatting ends and image processing begins.

Keep `input_ids`, attention masks, and image grid metadata as integer tensors.
The GPU transfer casts only floating-point tensors to BF16; casting every input
to BF16 would corrupt the token/index interface.

## 5. Count the visual tokens

The inspection saves the actual tensor shapes; do not assume `pixel_values` must
have a conventional `[batch, channels, height, width]` shape. A patch-based
processor may return flattened patch data instead.

For the configured model, let `(T, H, W)` denote the patch grid and `s` the spatial
merge size. The number of visual positions consumed by the decoder is

\[
N_{\mathrm{visual}} = \frac{T H W}{s^2}.
\]

The code reads `s` from the loaded model configuration and checks that this count
matches the number of image placeholder IDs in the processed sequence. Start/end
markers and ordinary prompt tokens are additional positions.

The same 448×448 board is processed twice, with these pixel limits:

| Configuration | Expected processed size for this square fixture | Expected visual tokens |
| --- | --- | --- |
| Low: `224*224` | 224×224 | 64 |
| Default: `448*448` | 448×448 | 256 |

These counts follow from 14-pixel patches and a spatial merge size of 2 for this
model. The processor applies its own resizing rules; the JSON report, not the
nominal input size, is the evidence for what was processed. Pixel budgets are a
model-specific control; see the
[Qwen2.5-VL usage guide](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl).

Our pinned Transformers version uses the current `size` interface, passed
explicitly on each processor call:

```python
images_kwargs={
    "size": {"shortest_edge": config.min_pixels, "longest_edge": config.max_pixels}
}
```

For this Qwen image backend these two values are pixel-area budgets, despite the
edge-oriented names. This is model-specific behavior. Passing the older
`min_pixels`/`max_pixels` keyword interface produces a deprecation warning in the
pinned version. See its
[resize implementation](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py).

Checkpoint configuration can override settings passed only during construction.
We therefore check the actual processed dimensions against the requested area
limits. This check caught a silent override while building this lesson.

Higher resolution can preserve details but increases visual encoding and decoder
context cost. Dense decoder prefill attention has a quadratic sequence-length
term, and KV-cache storage grows with context length. Do not infer a fixed runtime
multiplier from the token ratio: text overhead, vision work, kernels, and decoding
also contribute.

This becomes especially important in multi-turn RL: retaining several screenshots
adds many visual positions before the agent generates its next action.

## 6. Keep only the generated tokens

For this decoder-only, batch-size-one call, `generate()` returns the input prefix
followed by new IDs:

```python
completion_ids = generated_suffix(prompt_ids, output_ids)
text = processor.tokenizer.decode(completion_ids, skip_special_tokens=True)
```

The helper verifies the prefix, then slices by its full token length. We save the
exact completion IDs, token strings, and both versions of the decoded answer:
with and without special tokens. Searching for the first EOS marker would be
wrong because the prompt itself can contain completed chat turns.

Later, GRPO will score precisely the sampled action tokens. Decoding and
re-tokenizing a response is not a reliable way to recover those exact IDs.
For padded batches, prompt-width handling needs additional care; this lesson uses
one example per call to keep that boundary explicit.

The strict parser accepts one direction word, allowing case/whitespace variation.
It does not turn an explanation, a list of moves, or `left.` into an action.
Formatting failures are preserved in the report instead of silently repaired.

## 7. Connect this to the RL you know

For an action expressed as tokens \(a=(y_1,\ldots,y_k)\),

\[
\pi_\theta(a\mid I,p)
=\prod_{j=1}^{k}\pi_\theta(y_j\mid I,p,y_{<j}).
\]

The autoregressive policy idea is familiar. What is new is that the conditioning
input includes visual features and their layout. In later stages, \(I,p\) expands
into a history of images, instructions, actions, and environment responses.

Greedy inference is useful for this control but provides no exploration on a
fixed input. Reusing greedy settings for every member of a GRPO group would
produce identical trajectories in a deterministic setup, leaving no relative
reward signal. Stage 5 will explicitly introduce sampling and behavior-policy
log-probabilities.

The report also saves raw first-token logits converted to probabilities: the
five most likely tokens and the probabilities for the four direction tokens.
The direction words are verified to be single tokens for this pinned tokenizer.
These probabilities are normalized across the **whole vocabulary**, not just the
four directions, so the four values need not sum to one.

An image can change this distribution without changing its maximum. Thus an
unchanged greedy action does not by itself prove that the image was ignored.
Conversely, different probabilities do not establish correct perception or
planning. The observed probabilities are also not calibrated confidence scores.

## 8. Inspect your results

The command prints the observed answer and expected direction for each board,
then saves checksum-verified local copies under `artifacts/STAGE2_RUN_ID/`:

- `goal-left.png` / `goal-right.png`: the exact images supplied to the processor.
- `goal-left-prompt.txt` / `goal-right-prompt.txt`: rendered chat prompts.
- `chat-template.json`: the processor's underlying template.
- `inspection.json`: tensor metadata, patch grids, complete input IDs, generated
  IDs, action parsing, first-token probabilities, and the resolution comparison.
- `config.json`, `environment.json`, `model-cache.json`, `manifest.json`: settings,
  runtime/memory information, cache receipt, and checksums.
- `inspect-vlm.json`: a convenient combined local result.

Open the PNGs before interpreting the model. Then compare `checks` with
`behavior`: the former describes enforced pipeline invariants; the latter reports
whether the actions changed and whether both were correct. Two easy fixtures are
not a navigation benchmark or proof that a trained agent will generalize.

In the initial default run, the model answered `left` for both boards. The left
case was correct and the right case was wrong; all input checks passed. Its
direction-token probabilities changed with the image while the highest-probability
direction stayed `left`. We preserve that negative result rather than changing
the fixtures until the model succeeds. These controls do not yet identify whether
the error comes from visual recognition, instruction following, or action selection.

The report's generation timers exclude model loading. Peak GPU memory describes
the inspection after loading, including resident model memory, but excludes
transient loading peaks. CPU download time is outside the GPU function timer.

## 9. Common issues

| Symptom | What to inspect |
| --- | --- |
| Output contains your instruction | Decode the generated suffix, not the entire returned sequence. |
| Image features/token count mismatch | Keep the processor, model revision, image order, grid metadata, and resize settings aligned. |
| A lower resolution setting does not reduce the grid | Check argument precedence and actual processed dimensions; use explicit per-call image kwargs. |
| Same token IDs for different images | Expected for equal-sized image placeholders; compare pixel tensors and images too. |
| Plausible output with wrong goal direction | First inspect the board, legend, coordinate convention, and processed resolution. Separate visual recognition from action formatting and planning. |
| Invalid output despite understanding the image | Inspect raw generated tokens and `finish_reason`; strict output formatting is its own capability. |
| Answer ends halfway through | Check `max_new_tokens` and the EOS IDs. Keep the raw truncated result. |
| CUDA out of memory | Check image-token count, context size, BF16 dtype, and device placement. Start with one example and bounded output. |
| GPU appears to download weights | Preparation must commit the cache before GPU loading. The inspection job intentionally loads offline. |
| Image changes are ignored | Verify `different_pixel_values`, identical text, fresh generation calls, and the actual PNGs before interpreting model behavior. |
| Left/right is reversed | Renderer coordinates are `(row, column)` but image drawing uses `(x=column, y=row)`. |

## Exercise — Predict the representation

1. Before opening the report, predict which fields differ between the two
   controls: token IDs, pixel values, patch grid, generated action.
2. Compare `resolution_comparison.low` and `.configured`. Compute their visual-token
   ratio and identify the non-image positions in the input token list.
3. Repeat with a lower maximum pixel budget:

   ```bash
   uv run modal run modal_app.py --task inspect-vlm --max-pixels 50176
   ```

   Predict the grid and token count. Check behavior separately from representation
   changes; lower resolution need not make these simple boards fail.
4. Explain why changing the seed need not change greedy outputs, and why the
   processor's image tensors must be available when recomputing action log-probabilities.

You are ready for Stage 3 when you can trace an image to visual positions, identify
the generated suffix, and explain why matching text tokens do not mean matching
VLM inputs. Next we will turn the static boards into a deterministic environment
with actions, transitions, rewards, and trustworthy baselines.

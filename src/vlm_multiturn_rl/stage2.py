"""Stage 2: inspect a pinned VLM on two controlled image observations."""

import base64
from dataclasses import asdict
import importlib.metadata
import os
from pathlib import Path
import platform
import time

from .stage1 import seed_everything
from .vision import (
    LOW_PIXELS, Stage2Config, control_boards, expected_action, generated_suffix,
    messages, parse_action, render_board,
)
from .workspace import RUNS_PATH, run_path, sha256_file, write_json, write_manifest


ARTIFACT_FILES = (
    "config.json", "environment.json", "model-cache.json", "inspection.json",
    "chat-template.json", "goal-left.png", "goal-right.png",
    "goal-left-prompt.txt", "goal-right-prompt.txt",
)
SNAPSHOT_PATTERNS = ["*.json", "*.txt", "*.jinja", "*.model", "*.safetensors"]


def prepare_model(raw_config: dict) -> dict:
    """CPU job: populate the persistent cache before allocating a GPU."""
    from huggingface_hub import snapshot_download

    config = Stage2Config(**raw_config)
    snapshot = Path(snapshot_download(
        config.model_id, revision=config.model_revision,
        allow_patterns=SNAPSHOT_PATTERNS, max_workers=2,
    ))
    weights = sorted(snapshot.glob("*.safetensors"))
    if snapshot.name != config.model_revision or not weights:
        raise RuntimeError("Expected safetensors weights at the pinned model snapshot")
    return {
        "model_id": config.model_id,
        "revision": snapshot.name,
        "weight_files": [path.name for path in weights],
        "weight_bytes": sum(path.stat().st_size for path in weights),
        "downloader_task_id": os.environ["MODAL_TASK_ID"],
    }


def tensor_summary(inputs: dict) -> dict:
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
        for key, value in inputs.items()
    }


def inspect_inputs(inputs, model_config, processor) -> dict:
    ids = inputs["input_ids"][0].tolist()
    grids = inputs["image_grid_thw"].tolist()
    merge = model_config.vision_config.spatial_merge_size
    image_token_count = ids.count(model_config.image_token_id)
    expected_tokens = sum(t * h * w // merge**2 for t, h, w in grids)
    if image_token_count != expected_tokens:
        raise RuntimeError("Image placeholders and spatially merged image features do not align")
    return {
        "tensors": tensor_summary(inputs),
        "image_grid_thw": grids,
        "processed_image_sizes": [
            [h * model_config.vision_config.patch_size, w * model_config.vision_config.patch_size]
            for _, h, w in grids
        ],
        "vision_patch_size": model_config.vision_config.patch_size,
        "spatial_merge_size": merge,
        "image_token_id": model_config.image_token_id,
        "image_token_count": image_token_count,
        "prompt_token_count": len(ids),
        "input_ids": ids,
        "input_tokens": processor.tokenizer.convert_ids_to_tokens(ids),
        "expanded_prompt": processor.tokenizer.decode(ids, skip_special_tokens=False),
    }


def inspect_vlm(run_id: str, raw_config: dict, source_sha256: dict, cache_receipt: dict) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, GenerationConfig, Qwen2_5_VLForConditionalGeneration

    started = time.monotonic()
    config = Stage2Config(**raw_config)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage 2 requires a CUDA GPU with BF16 support")
    seed_everything(config.seed)
    snapshot = snapshot_download(
        config.model_id, revision=config.model_revision,
        allow_patterns=SNAPSHOT_PATTERNS, local_files_only=True,
    )
    if Path(snapshot).name != config.model_revision or cache_receipt["revision"] != config.model_revision:
        raise RuntimeError("The cache receipt and loaded snapshot must match the configured revision")

    print("Loading pinned Qwen2.5-VL weights from the persistent cache...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        snapshot, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa",
    ).eval()
    torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    generation = GenerationConfig(
        do_sample=False, max_new_tokens=config.max_new_tokens, use_cache=True,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        temperature=1.0, top_p=1.0, top_k=50, repetition_penalty=1.0, num_beams=1,
        return_dict_in_generate=True, output_logits=True,
    )
    direction_token_ids = {
        action: processor.tokenizer.encode(action, add_special_tokens=False)
        for action in ("up", "down", "left", "right")
    }
    if any(len(ids) != 1 for ids in direction_token_ids.values()):
        raise RuntimeError("The Stage 2 direction-probability diagnostic expects one token per direction")
    directory = run_path(RUNS_PATH, run_id)
    directory.mkdir(parents=True, exist_ok=False)
    cases = {}
    image_inputs = []
    rendered_prompts = []

    for name, board in control_boards().items():
        image = render_board(board)
        image.save(directory / f"{name}.png")
        # Make the template step explicit: render role/image markers, then process
        # the text together with the actual image in the same placeholder order.
        prompt = processor.apply_chat_template(messages(), tokenize=False, add_generation_prompt=True)
        (directory / f"{name}-prompt.txt").write_text(prompt)
        rendered_prompts.append(prompt)
        inputs = processor(
            text=[prompt], images=[image], return_tensors="pt", add_special_tokens=False,
            images_kwargs={"size": {"shortest_edge": config.min_pixels, "longest_edge": config.max_pixels}},
        )
        input_report = inspect_inputs(inputs, model.config, processor)
        image_inputs.append(inputs["pixel_values"].clone())
        # Integer token IDs/grid metadata remain integers; only floats become BF16.
        gpu_inputs = {
            key: value.to(device="cuda:0", dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
            for key, value in inputs.items()
        }
        torch.cuda.synchronize()
        generation_started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**gpu_inputs, generation_config=generation)
        torch.cuda.synchronize()
        generation_seconds = time.monotonic() - generation_started
        completion_ids = generated_suffix(input_report["input_ids"], output.sequences[0].cpu().tolist())
        text = processor.tokenizer.decode(completion_ids, skip_special_tokens=True)
        action = parse_action(text)
        first_token_probs = output.logits[0][0].float().softmax(dim=-1)
        top_probs, top_ids = first_token_probs.topk(5)
        direction_probs = {
            direction: float(first_token_probs[ids[0]])
            for direction, ids in direction_token_ids.items()
        }
        eos_ids = generation.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
        cases[name] = {
            "image_file": f"{name}.png",
            "board": asdict(board),  # Diagnostic label, not part of the model prompt.
            "expected_action": expected_action(board),
            "input": input_report,
            "generation_input_tensors": tensor_summary(gpu_inputs),
            "generation": {
                "text": text, "action": action, "format_valid": action is not None,
                "correct": action == expected_action(board),
                "generated_ids": completion_ids,
                "generated_tokens": processor.tokenizer.convert_ids_to_tokens(completion_ids),
                "text_with_special_tokens": processor.tokenizer.decode(completion_ids, skip_special_tokens=False),
                "finish_reason": "eos" if completion_ids and completion_ids[-1] in eos_ids else "length",
                "seconds": generation_seconds,
                "first_token_direction_probabilities": direction_probs,
                "first_token_top5": [
                    {"id": token_id, "token": processor.tokenizer.convert_ids_to_tokens(token_id), "probability": probability}
                    for token_id, probability in zip(top_ids.tolist(), top_probs.tolist())
                ],
            },
        }
        if name == "goal-left":
            low_inputs = processor(
                text=[prompt], images=[image], return_tensors="pt", add_special_tokens=False,
                images_kwargs={"size": {"shortest_edge": LOW_PIXELS, "longest_edge": LOW_PIXELS}},
            )
            resolution = {
                "low_max_pixels": LOW_PIXELS,
                "configured_max_pixels": config.max_pixels,
                "low": inspect_inputs(low_inputs, model.config, processor),
                "configured": input_report,
            }
        print(f"{name}: {text!r}; expected {expected_action(board)!r}", flush=True)

    checks = {
        "same_text_prompt": rendered_prompts[0] == rendered_prompts[1],
        "same_input_ids": cases["goal-left"]["input"]["input_ids"] == cases["goal-right"]["input"]["input_ids"],
        "different_pixel_values": not torch.equal(image_inputs[0], image_inputs[1]),
        "image_tokens_match_grid": True,  # Enforced separately for each processor call above.
        "exact_generated_suffix_saved": True,
        "pixel_budgets_respected": (
            all(h * w == LOW_PIXELS for h, w in resolution["low"]["processed_image_sizes"])
            and all(config.min_pixels <= h * w <= config.max_pixels
                    for h, w in resolution["configured"]["processed_image_sizes"])
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Controlled-image input checks failed: {checks}")
    # Behavior is observed, not asserted: untrained policies may make mistakes.
    behavior = {
        "actions_changed": cases["goal-left"]["generation"]["action"] != cases["goal-right"]["generation"]["action"],
        "both_actions_correct": all(case["generation"]["correct"] for case in cases.values()),
        "max_direction_probability_change": max(
            abs(cases["goal-left"]["generation"]["first_token_direction_probabilities"][direction]
                - cases["goal-right"]["generation"]["first_token_direction_probabilities"][direction])
            for direction in direction_token_ids
        ),
    }
    inspection = {"cases": cases, "resolution_comparison": resolution, "checks": checks, "behavior": behavior}
    environment = {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "transformers": importlib.metadata.version("transformers"),
        "gpu": torch.cuda.get_device_name(0), "cuda_runtime": str(torch.version.cuda),
        "dtype": str(model.dtype), "attention_implementation": "sdpa",
        "processor_class": type(processor).__name__,
        "model_revision": Path(snapshot).name,
        "producer_task_id": os.environ["MODAL_TASK_ID"],
        "modal_image_id": os.environ["MODAL_IMAGE_ID"],
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    write_json(directory / "config.json", asdict(config))
    write_json(directory / "environment.json", environment)
    write_json(directory / "model-cache.json", cache_receipt)
    write_json(directory / "inspection.json", inspection)
    write_json(directory / "chat-template.json", {"chat_template": processor.chat_template})
    manifest = write_manifest(directory, {
        "source_sha256": source_sha256,
        "gpu_requirements_sha256": sha256_file(Path("/root/requirements-gpu.txt")),
        "elapsed_seconds": time.monotonic() - started,
        "generation": generation.to_dict(),
    }, stage=2, artifact_files=ARTIFACT_FILES)
    # Small inspectable artifacts travel back as bytes encoded in plain strings.
    files = {
        name: base64.b64encode((directory / name).read_bytes()).decode("ascii")
        for name in (*ARTIFACT_FILES, "manifest.json")
    }
    return {
        "manifest": manifest, "config": asdict(config), "environment": environment,
        "inspection": inspection, "verification": checks, "files": files,
    }

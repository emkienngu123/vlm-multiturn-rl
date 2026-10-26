"""Multi-turn behavior cloning. GPU imports stay inside the remote path."""

import base64
from dataclasses import asdict, dataclass
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import zipfile

from .gridworld import Layout, make_layouts, render_observation, render_svg, summarize
from .stage3 import GridVLM, Stage3Config, replay_html
from .trajectories import (
    TOOL_PROMPT, assistant_labels, collect_trajectory, demonstration_examples,
    image_paths, replay_trajectory,
)
from .vision import generated_suffix
from .workspace import RUNS_PATH, run_path, verify_artifacts, write_json, write_manifest


ARTIFACT_FILES = (
    "config.json", "environment.json", "layouts.json", "demonstrations.json", "examples.json",
    "episodes.json", "metrics.json", "replay.html", "policy-prompt.txt", "images.zip",
)
GPU_ARTIFACT_FILES = (*ARTIFACT_FILES, "training.json", "mask-inspection.json", "adapter.zip",
                      "model-cache.json", "chat-template.txt")


@dataclass(frozen=True)
class Stage4Config(Stage3Config):
    max_new_tokens: int = 64
    train_layouts: int = 8
    updates: int = 2
    gradient_accumulation: int = 2
    learning_rate: float = 2e-4
    lora_rank: int = 8
    max_sequence_tokens: int = 4096

    def __post_init__(self):
        super().__post_init__()
        for name, lower, upper in (("train_layouts", 1, 32), ("updates", 1, 20),
                                   ("gradient_accumulation", 1, 4)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        if self.max_new_tokens < 32:
            raise ValueError("Tool calls need a larger generation budget: use 32 to 64 tokens")
        if not math.isfinite(self.learning_rate) or not 1e-6 <= self.learning_rate <= 1e-3:
            raise ValueError("learning_rate must be finite and in [1e-6, 1e-3]")
        if self.lora_rank != 8 or self.max_sequence_tokens != 4096:
            raise ValueError("Stage 4 fixes LoRA rank 8 and a 4096-token context guard")


def build_demonstrations(config: Stage4Config, assets: dict, *, raster: bool = False) -> dict:
    train = make_layouts("train", config.train_layouts, config.dataset_seed)
    evaluation = make_layouts("eval", config.episodes, config.dataset_seed)
    if {x.maze_id for x in train} & {x.maze_id for x in evaluation}:
        raise RuntimeError("Training and evaluation mazes overlap")
    demos = [collect_trajectory(
        layout, "oracle", key=f"demo-{index:03d}", assets=assets,
        renderer=render_observation if raster else render_svg, extension="png" if raster else "svg",
        seed=config.seed, max_steps=config.max_steps,
    ) for index, layout in enumerate(train)]
    return {"layouts": {"train": [x.to_dict() for x in train], "eval": [x.to_dict() for x in evaluation]},
            "demonstrations": demos, "examples": demonstration_examples(demos),
            "episodes": [], "metrics": {"oracle_train": summarize(demos)},
            "verification": {"train_eval_mazes_disjoint": True, "all_episodes_replayed": True,
                             "protocol": "move-json-v1", "observation_mode": "full_history"}}


def _images_zip(assets: dict) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, image in assets.items():
            if isinstance(image, str):
                content = image.encode()
            else:
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                content = buffer.getvalue()
            archive.writestr(name, content)
    return output.getvalue()


def save_report(directory: Path, config: Stage4Config, data: dict, environment: dict,
                source_sha256: dict, assets: dict, extra_files: dict | None = None) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    values = {"config.json": asdict(config), "environment.json": environment, "layouts.json": data["layouts"],
              "demonstrations.json": {"episodes": data["demonstrations"]},
              "examples.json": {"examples": data["examples"]}, "episodes.json": {"episodes": data["episodes"]},
              "metrics.json": {"metrics": data["metrics"], "verification": data["verification"]}}
    for name, value in values.items():
        write_json(directory / name, value)
    display = {**data, "episodes": [*data["demonstrations"], *data["episodes"]]}
    (directory / "replay.html").write_text(replay_html(display, "Stage 4: multi-turn SFT bridge"))
    (directory / "policy-prompt.txt").write_text(TOOL_PROMPT + "\n")
    (directory / "images.zip").write_bytes(_images_zip(assets))
    for name, content in (extra_files or {}).items():
        (directory / name).write_bytes(content)
    files = GPU_ARTIFACT_FILES if extra_files is not None else ARTIFACT_FILES
    manifest = write_manifest(directory, {"source_sha256": source_sha256, "environment_version": "grid-v1",
                                         "protocol": "move-json-v1", "mode": environment["mode"]},
                              stage=4, artifact_files=files)
    verify_artifacts(directory, stage=4, artifact_files=files)
    return manifest


def replay_saved_run(directory: Path) -> int:
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest.get("environment_version") != "grid-v1" or manifest.get("protocol") != "move-json-v1"
            or manifest.get("mode") not in {"cpu", "gpu"}):
        raise ValueError("Unknown Stage 4 report contract")
    files = GPU_ARTIFACT_FILES if manifest["mode"] == "gpu" else ARTIFACT_FILES
    verify_artifacts(directory, stage=4, artifact_files=files)
    layouts = json.loads((directory / "layouts.json").read_text())
    lookup = {x["layout_id"]: Layout.from_dict(x) for group in layouts.values() for x in group}
    episodes = []
    for name in ("demonstrations.json", "episodes.json"):
        episodes.extend(json.loads((directory / name).read_text())["episodes"])
    with zipfile.ZipFile(directory / "images.zip") as archive:
        names = set(archive.namelist())
        for episode in episodes:
            replay_trajectory(lookup[episode["layout_id"]], episode)
            if any(o["image"] not in names for o in episode["observations"]):
                raise ValueError("Missing observation asset")
    expected = demonstration_examples([e for e in episodes if e["policy"] == "oracle"])
    if json.loads((directory / "examples.json").read_text())["examples"] != expected:
        raise ValueError("SFT examples differ from recorded solver actions")
    return len(episodes)


class HistoryVLM(GridVLM):
    """Reuse the pinned loader; replace the Stage 3 current-image interface."""

    def encode_history(self, messages: list[dict], images: list):
        if len(image_paths(messages)) != len(images):
            raise ValueError("Image objects must match message placeholder order")
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=images, return_tensors="pt", add_special_tokens=False,
            images_kwargs={"size": {"shortest_edge": self.config.min_pixels, "longest_edge": self.config.max_pixels}},
        )
        ids = inputs["input_ids"][0].tolist()
        grids = inputs["image_grid_thw"].tolist()
        expected = sum(t*h*w // self.model.config.vision_config.spatial_merge_size**2 for t, h, w in grids)
        if len(grids) != len(images) or ids.count(self.model.config.image_token_id) != expected:
            raise RuntimeError("Image/token alignment failed")
        if len(ids) + self.config.max_new_tokens > self.config.max_sequence_tokens:
            raise RuntimeError("History exceeds the context guard; reduce image resolution")
        return inputs

    @staticmethod
    def to_gpu(inputs):
        import torch

        return {k: v.to("cuda:0", dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                for k, v in inputs.items()}

    def act(self, messages: list[dict], images: list) -> dict:
        import torch

        inputs = self.encode_history(messages, images)
        prompt_ids = inputs["input_ids"][0].tolist()
        gpu_inputs = self.to_gpu(inputs)
        torch.cuda.synchronize()
        started = time.monotonic()
        self.model.eval()
        with torch.inference_mode():
            outputs = self.model.generate(**gpu_inputs, generation_config=self.generation)
        torch.cuda.synchronize()
        ids = generated_suffix(prompt_ids, outputs[0].cpu().tolist())
        eos = self.generation.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        ended = bool(ids and ids[-1] in eos)
        # Keep tool delimiters, even if registered as special tokens. Remove ONLY
        # a terminal EOS for parsing; save the exact generated sequence including EOS.
        text_ids = ids[:-1] if ended else ids
        return {"text": self.processor.tokenizer.decode(text_ids, skip_special_tokens=False),
                "generated_ids": ids, "prompt_ids": prompt_ids, "token_source": "model_generation",
                "image_paths": image_paths(messages), "image_grid_thw": inputs["image_grid_thw"].tolist(),
                "finish_reason": "eos" if ended else "length", "seconds": time.monotonic() - started}

    def sft_batch(self, example: dict, assets: dict):
        import torch

        paths = image_paths(example["messages"])
        inputs = self.encode_history(example["messages"], [assets[p] for p in paths])
        prompt_ids = inputs["input_ids"][0].tolist()
        tokenizer = self.processor.tokenizer
        # Solver text has no generated token sequence. Define its continuation
        # once, explicitly, including the assistant end-of-turn token.
        target_ids = tokenizer.encode(example["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
        ids, labels = assistant_labels(prompt_ids, target_ids, self.model.config.image_token_id)
        if len(ids) > self.config.max_sequence_tokens:
            raise RuntimeError("SFT target exceeds the context guard")
        inputs["input_ids"] = torch.tensor([ids], dtype=inputs["input_ids"].dtype)
        for key in ("attention_mask", "mm_token_type_ids"):
            if key in inputs:
                suffix = torch.full((1, len(target_ids)), 1 if key == "attention_mask" else 0, dtype=inputs[key].dtype)
                inputs[key] = torch.cat([inputs[key], suffix], dim=1)
        inputs["labels"] = torch.tensor([labels], dtype=torch.long)
        inspection = {"layout_id": example["layout_id"], "turn": example["turn"], "image_paths": paths,
                      "image_grid_thw": inputs["image_grid_thw"].tolist(), "prompt_tokens": len(prompt_ids),
                      "target_tokens": len(target_ids), "input_ids": ids, "labels": labels,
                      "tokens": tokenizer.convert_ids_to_tokens(ids),
                      "supervised_span": [len(prompt_ids), len(ids)], "target": example["target"],
                      "eos_token_id": tokenizer.eos_token_id}
        return self.to_gpu(inputs), inspection


def _probe(actor: HistoryVLM, example: dict, assets: dict) -> tuple[float, object]:
    """Verify the model's internal causal shift against explicit target-token CE."""
    import torch
    import torch.nn.functional as F

    batch, inspection = actor.sft_batch(example, assets)
    actor.model.eval()
    with torch.no_grad():
        outputs = actor.model(**batch, use_cache=False)
        start = inspection["prompt_tokens"]
        manual = F.cross_entropy(outputs.logits[0, start-1:-1].float(), batch["labels"][0, start:])
    if not torch.allclose(outputs.loss.float(), manual, rtol=1e-4, atol=1e-4):
        raise RuntimeError("Internal loss and explicitly shifted assistant-only loss differ")
    return float(outputs.loss), outputs.logits[0, start-1].float().cpu()


def train_adapter(actor: HistoryVLM, examples: list[dict], assets: dict, adapter_directory: Path) -> dict:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict

    config = actor.config
    base_loss, base_logits = _probe(actor, examples[0], assets)
    # Full names avoid accidentally adapting vision attention's q_proj/v_proj.
    targets = [name for name, module in actor.model.named_modules()
               if ".language_model." in name and name.endswith((".q_proj", ".v_proj"))
               and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise RuntimeError("No language-side attention targets found; inspect model module names")
    actor.model.requires_grad_(False)
    actor.model = get_peft_model(actor.model, LoraConfig(
        task_type="CAUSAL_LM", r=config.lora_rank, lora_alpha=16, lora_dropout=0.0,
        target_modules=targets, bias="none",
    ))
    trainable = {name: p for name, p in actor.model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in name or ".language_model." not in name for name in trainable):
        raise RuntimeError("Unexpected trainable parameters outside language-side LoRA")
    initial = {name: p.detach().cpu().clone() for name, p in trainable.items()}
    initial_loss, initial_logits = _probe(actor, examples[0], assets)
    if not torch.allclose(base_logits, initial_logits, rtol=1e-4, atol=1e-4):
        raise RuntimeError("Default LoRA initialization changed the base model")
    actor.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    actor.model.config.use_cache = False
    optimizer = torch.optim.AdamW(trainable.values(), lr=config.learning_rate, weight_decay=0.0)
    order = list(range(len(examples)))
    random.Random(config.seed).shuffle(order)
    rows = []
    for update in range(config.updates):
        started = time.monotonic()
        actor.model.train()
        optimizer.zero_grad(set_to_none=True)
        losses, counts = [], []
        for microbatch in range(config.gradient_accumulation):
            index = order[(update * config.gradient_accumulation + microbatch) % len(order)]
            batch, inspection = actor.sft_batch(examples[index], assets)
            outputs = actor.model(**batch, use_cache=False)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite SFT loss")
            (loss / config.gradient_accumulation).backward()
            losses.append(float(loss.detach()))
            counts.append(inspection["target_tokens"])
            del batch, outputs, loss
        gradient_norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0, error_if_nonfinite=True)
        if any(p.grad is not None for name, p in actor.model.named_parameters() if name not in trainable):
            raise RuntimeError("Frozen base/vision parameter received a gradient")
        optimizer.step()
        torch.cuda.synchronize()
        row = {"update": update+1, "mean_action_loss": sum(losses)/len(losses),
               "target_tokens": sum(counts), "gradient_norm_before_clip": float(gradient_norm),
               "seconds": time.monotonic()-started}
        rows.append(row)
        print(f"SFT update {update+1}/{config.updates}: {row}", flush=True)
    optimizer.zero_grad(set_to_none=True)
    actor.model.gradient_checkpointing_disable()
    actor.model.config.use_cache = True
    final_loss, final_logits = _probe(actor, examples[0], assets)
    changed = {name: float((p.detach().cpu()-initial[name]).abs().max()) for name, p in trainable.items()}
    if not any(value > 0 for value in changed.values()):
        raise RuntimeError("Optimizer did not change any adapter parameters")
    actor.model.peft_config["default"].base_model_name_or_path = config.model_id
    actor.model.peft_config["default"].revision = config.model_revision
    # Embeddings are frozen and vocabulary is unchanged; avoid an unnecessary
    # online config lookup from PEFT's automatic embedding-save detection.
    actor.model.save_pretrained(adapter_directory, safe_serialization=True, save_embedding_layers=False)
    expected_state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(
        actor.model, save_embedding_layers=False).items()}
    # Unload (do not merge) and reload from disk on the same frozen base model.
    # This tests the saved adapter without allocating a second 3B base model.
    base = actor.model.unload()
    del optimizer, trainable, initial
    actor.model = PeftModel.from_pretrained(base, adapter_directory, is_trainable=False).eval()
    actual_state = get_peft_model_state_dict(actor.model, save_embedding_layers=False)
    if set(expected_state) != set(actual_state) or any(
        not torch.equal(value, actual_state[key].detach().cpu()) for key, value in expected_state.items()
    ):
        raise RuntimeError("Reloaded adapter differs from the saved state")
    reloaded_loss, reloaded_logits = _probe(actor, examples[0], assets)
    if not torch.allclose(final_logits, reloaded_logits, rtol=1e-4, atol=1e-4):
        raise RuntimeError("Reloaded adapter changed probe logits")
    return {"updates": rows, "target_modules": targets,
            "trainable_parameters": sum(v.numel() for v in expected_state.values()),
            "max_parameter_change": max(changed.values()), "base_probe_loss": base_loss,
            "initial_adapter_probe_loss": initial_loss, "final_probe_loss": final_loss,
            "reloaded_probe_loss": reloaded_loss,
            "verification": {"language_lora_only": True, "frozen_parameters_no_gradients": True,
                             "lora_initialization_is_noop": True, "adapter_parameters_updated": True,
                             "causal_shift_matches_manual_loss": True, "saved_adapter_reload_matches": True}}


def _evaluate(actor: HistoryVLM, data: dict, assets: dict, policy: str):
    rows = []
    for index, raw in enumerate(data["layouts"]["eval"]):
        episode = collect_trajectory(
            Layout.from_dict(raw), policy, key=f"{policy}-{index:03d}", assets=assets,
            renderer=render_observation, extension="png", actor=actor.act,
            seed=actor.config.seed, max_steps=actor.config.max_steps,
        )
        rows.append(episode)
        print(f"{policy} {index+1}: success={episode['success']} steps={episode['steps']} "
              f"invalid={episode['invalid_actions']}", flush=True)
    data["episodes"].extend(rows)
    data["metrics"][policy] = summarize(rows)


def train_remote(run_id: str, raw_config: dict, source_sha256: dict, cache_receipt: dict) -> dict:
    import torch

    config = Stage4Config(**raw_config)
    if cache_receipt["revision"] != config.model_revision:
        raise RuntimeError("Model cache revision mismatch")
    started = time.monotonic()
    actor = HistoryVLM(config)
    torch.cuda.reset_peak_memory_stats()
    assets = {}
    data = build_demonstrations(config, assets, raster=True)
    print(f"Prepared {len(data['examples'])} solver action examples", flush=True)
    _evaluate(actor, data, assets, "base")
    # Inspect the longest history to verify that multiple tool images remain ordered.
    longest = max(data["examples"], key=lambda e: len(image_paths(e["messages"])))
    batch, mask = actor.sft_batch(longest, assets)
    del batch
    temporary_adapter = Path("/tmp") / run_id / "adapter"
    training = train_adapter(actor, data["examples"], assets, temporary_adapter)
    # Evaluate the reloaded adapter, not the in-memory pre-save wrapper.
    _evaluate(actor, data, assets, "sft")
    data["verification"].update(training["verification"])
    data["verification"].update({"paired_evaluation_layouts_and_decoding": True,
                                 "sft_evaluation_uses_reloaded_adapter": True,
                                 "longest_demo_history_images": len(mask["image_paths"])})
    environment = {
        "mode": "gpu", "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "pillow")},
        "model_revision": config.model_revision, "observation_mode": "full_history", "decoding": "greedy",
        "dtype": "bfloat16 base; float32 LoRA", "attention_implementation": "sdpa",
        "producer_task_id": os.environ["MODAL_TASK_ID"], "modal_image_id": os.environ["MODAL_IMAGE_ID"],
        "total_seconds": time.monotonic()-started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "generation_config": actor.generation.to_dict(),
    }
    adapter_zip = io.BytesIO()
    with zipfile.ZipFile(adapter_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(temporary_adapter.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(Path("adapter") / path.relative_to(temporary_adapter)))
    json_bytes = lambda value: (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    directory = run_path(RUNS_PATH, run_id)
    manifest = save_report(directory, config, data, environment, source_sha256, assets, {
        "training.json": json_bytes(training), "mask-inspection.json": json_bytes(mask),
        "adapter.zip": adapter_zip.getvalue(), "model-cache.json": json_bytes(cache_receipt),
        "chat-template.txt": actor.processor.chat_template.encode(),
    })
    replay_saved_run(directory)
    return {"manifest": manifest, "config": asdict(config), "environment": environment,
            "metrics": data["metrics"], "verification": data["verification"], "training": training,
            "files": {name: base64.b64encode((directory/name).read_bytes()).decode("ascii")
                      for name in (*GPU_ARTIFACT_FILES, "manifest.json")}}

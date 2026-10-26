"""Four same-board rollouts, frozen behavior scores, and one GRPO update per group."""

import base64
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import time
import zipfile

from .grpo import clipped_token_terms, relative_advantages, selected_log_probs, tensor_checks, validate_group
from .gridworld import Layout, episode_seed, make_layouts, render_observation, summarize
from .stage3 import Stage3Config, replay_html
from .stage4 import GPU_ARTIFACT_FILES as SFT_FILES, HistoryVLM, _evaluate, _images_zip
from .trajectories import TOOL_PROMPT, collect_trajectory, image_paths, replay_trajectory
from .vision import generated_suffix
from .workspace import RUNS_PATH, run_path, sha256_file, verify_artifacts, write_json, write_manifest


ARTIFACT_FILES = (
    "config.json", "environment.json", "layouts.json", "episodes.json", "groups.json", "training.json",
    "metrics.json", "replay.html", "images.zip", "policy-prompt.txt", "source-adapter.json",
    "adapter.zip", "mask-inspection.json", "model-cache.json", "chat-template.txt",
)


@dataclass(frozen=True)
class Stage5Config(Stage3Config):
    max_new_tokens: int = 64
    train_layouts: int = 8
    updates: int = 2  # Collection/update rounds; equal-return groups skip optimizer.step.
    learning_rate: float = 1e-5
    lora_rank: int = 8
    max_sequence_tokens: int = 4096
    group_size: int = 4
    clip_epsilon: float = 0.2
    advantage_epsilon: float = 1e-6
    sampling_logp_tolerance: float = 0.1
    sft_run_id: str = ""

    def __post_init__(self):
        super().__post_init__()
        for name, lo, hi in (("updates", 1, 20), ("train_layouts", 1, 32)):
            if type(getattr(self, name)) is not int or not lo <= getattr(self, name) <= hi:
                raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
        if self.max_new_tokens < 32:
            raise ValueError("Tool-call generation requires 32 to 64 tokens")
        if not math.isfinite(self.learning_rate) or not 1e-6 <= self.learning_rate <= 1e-4:
            raise ValueError("learning_rate must be finite and in [1e-6, 1e-4]")
        if (self.group_size != 4 or self.lora_rank != 8 or self.max_sequence_tokens != 4096
                or self.clip_epsilon != 0.2 or self.advantage_epsilon != 1e-6
                or self.sampling_logp_tolerance != 0.1):
            raise ValueError("Stage 5 fixes group=4, rank=8, context=4096, clip=.2, eps=1e-6, scoring tolerance=.1")
        if self.sft_run_id:
            run_path(Path("."), self.sft_run_id)


def latest_local_sft(root: Path) -> str:
    candidates = []
    for path in root.glob("stage4-*/manifest.json"):
        manifest = json.loads(path.read_text())
        if manifest.get("stage") == 4 and manifest.get("mode") == "gpu":
            candidates.append((manifest["created_at_utc"], path.parent.name))
    if not candidates:
        raise ValueError("No local GPU SFT report found. Run Stage 4 or supply --sft-run-id from the Modal volume")
    return max(candidates)[1]  # Latest completed run, never selected by evaluation reward.


def source_receipt(config: Stage5Config, root: Path = RUNS_PATH) -> dict:
    if not config.sft_run_id:
        raise ValueError("Stage 5 requires a completed GPU SFT run")
    directory = run_path(root, config.sft_run_id)
    manifest = verify_artifacts(directory, stage=4, artifact_files=SFT_FILES)
    previous = json.loads((directory / "config.json").read_text())
    if manifest.get("mode") != "gpu" or manifest.get("protocol") != "move-json-v1":
        raise ValueError("Source must be a GPU Stage 4 tool-policy checkpoint")
    for key in ("model_id", "model_revision", "dataset_seed", "lora_rank"):
        if previous[key] != getattr(config, key):
            raise ValueError(f"Source adapter uses a different {key}")
    return {"run_id": config.sft_run_id, "adapter_sha256": sha256_file(directory / "adapter.zip"),
            "manifest_sha256": sha256_file(directory / "manifest.json"), "sft_updates": previous["updates"],
            "model_revision": previous["model_revision"]}


def unpack_adapter(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    allowed = {"adapter/adapter_config.json", "adapter/adapter_model.safetensors", "adapter/README.md"}
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        if names - allowed or not {"adapter/adapter_config.json", "adapter/adapter_model.safetensors"} <= names:
            raise ValueError("Unexpected adapter archive contents")
        for name in names:
            (destination / Path(name).name).write_bytes(archive.read(name))


def adapter_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def extend_inputs(inputs, token_ids: list[int]):
    """Append actual generated IDs; never reconstruct them from decoded text."""
    import torch

    result = dict(inputs)
    suffix = torch.tensor([token_ids], dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device)
    result["input_ids"] = torch.cat([inputs["input_ids"], suffix], dim=1)
    for key in ("attention_mask", "mm_token_type_ids"):
        if key in inputs:
            extra = torch.full_like(suffix, 1 if key == "attention_mask" else 0, dtype=inputs[key].dtype)
            result[key] = torch.cat([inputs[key], extra], dim=1)
    return result


class GRPOPolicy(HistoryVLM):
    def __init__(self, config: Stage5Config, adapter_directory: Path):
        import torch
        from peft import PeftModel
        from transformers import GenerationConfig

        super().__init__(config)
        self.model = PeftModel.from_pretrained(self.model, adapter_directory, is_trainable=True).eval()
        trainable = [name for name, p in self.model.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in name or ".language_model." not in name for name in trainable):
            raise RuntimeError("Only language-side LoRA parameters may train")
        if any(module.p != 0 for module in self.model.modules() if isinstance(module, torch.nn.Dropout)):
            raise RuntimeError("The reference policy requires zero dropout")
        # No warpers or penalties: sampled probabilities and scored softmax refer
        # to the same policy. Disabling KV cache reduces sampling/scoring drift.
        self.rollout_generation = GenerationConfig(
            do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, num_beams=1,
            use_cache=False, max_new_tokens=config.max_new_tokens,
            eos_token_id=self.generation.eos_token_id, pad_token_id=self.generation.pad_token_id,
            return_dict_in_generate=True, output_scores=True,
        )

    def sample(self, messages: list[dict], images: list) -> dict:
        import torch

        inputs = self.encode_history(messages, images)
        prompt_ids = inputs["input_ids"][0].tolist()
        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(**self.to_gpu(inputs), generation_config=self.rollout_generation)
        ids = generated_suffix(prompt_ids, output.sequences[0].tolist())
        if not ids or len(output.scores) != len(ids):
            raise RuntimeError("Generation scores do not match sampled tokens")
        scores = torch.stack([row[0].float() for row in output.scores])
        chosen = torch.tensor(ids, device=scores.device)
        sampled_logps = scores.log_softmax(-1).gather(1, chosen[:, None]).squeeze(1).tolist()
        eos = self.rollout_generation.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        ended = ids[-1] in eos
        return {"text": self.processor.tokenizer.decode(ids[:-1] if ended else ids, skip_special_tokens=False),
                "generated_ids": ids, "prompt_ids": prompt_ids, "sampled_log_probs": sampled_logps,
                "token_source": "model_generation", "image_paths": image_paths(messages),
                "image_grid_thw": inputs["image_grid_thw"].tolist(),
                "finish_reason": "eos" if ended else "length"}

    def score(self, record: dict, assets: dict):
        generation = record["generation"]
        paths = image_paths(record["messages"])
        inputs = self.encode_history(record["messages"], [assets[name] for name in paths])
        prompt_ids = inputs["input_ids"][0].tolist()
        if (prompt_ids != generation["prompt_ids"] or paths != generation["image_paths"]
                or inputs["image_grid_thw"].tolist() != generation["image_grid_thw"]):
            raise RuntimeError("Scoring changed the sampled observation history or image order")
        ids = generation["generated_ids"]
        if not ids or self.model.config.image_token_id in ids:
            raise RuntimeError("Need a nonempty text action; generated image placeholders are unsupported")
        batch = self.to_gpu(extend_inputs(inputs, ids))
        self.model.eval()  # eval disables dropout; it does NOT disable autograd.
        outputs = self.model(**batch, use_cache=False)
        return selected_log_probs(outputs.logits, batch["input_ids"], len(prompt_ids))


def collect_group(actor: GRPOPolicy, layout: Layout, round_index: int, assets: dict) -> tuple[list, dict]:
    import torch

    fingerprint = adapter_fingerprint(actor.model)
    episodes = []
    for sample_index in range(actor.config.group_size):
        # A distinct deterministic RNG stream per whole trajectory. Never reset
        # it at each turn (which could accidentally repeat the same random draw).
        seed = (actor.config.seed + 1009*round_index + 9176*sample_index) % 2**32
        actual_seed = episode_seed(seed, layout)
        torch.manual_seed(actual_seed)
        torch.cuda.manual_seed_all(actual_seed)
        episode = collect_trajectory(
            layout, "rollout", key=f"round-{round_index:03d}-sample-{sample_index}", assets=assets,
            renderer=render_observation, extension="png", actor=actor.sample,
            seed=seed, max_steps=actor.config.max_steps,
        )
        episodes.append(episode)
        print(f"Round {round_index+1} sample {sample_index+1}/4: "
              f"return={episode['return']:.2f} success={episode['success']}", flush=True)
    validate_group(episodes, actor.config.group_size)
    max_sampling_delta = 0.0
    for episode in episodes:
        for record in episode["transitions"]:
            # Use exactly the later gradient-scoring path, then detach. No graph
            # is retained across actions and no optimizer step occurs here.
            logps = actor.score(record, assets).detach().cpu().tolist()
            generation = record["generation"]
            delta = max(abs(a-b) for a, b in zip(logps, generation["sampled_log_probs"]))
            max_sampling_delta = max(max_sampling_delta, delta)
            generation["old_log_probs"] = logps
            generation["behavior_fingerprint"] = fingerprint
    if adapter_fingerprint(actor.model) != fingerprint:
        raise RuntimeError("Policy changed during group collection/scoring")
    if max_sampling_delta > actor.config.sampling_logp_tolerance:
        raise RuntimeError(f"Sampling/scoring mismatch: max |delta logp|={max_sampling_delta:.6f}")
    stats = relative_advantages([e["return"] for e in episodes], actor.config.advantage_epsilon)
    return episodes, {**stats, "layout_id": layout.layout_id, "behavior_fingerprint": fingerprint,
                      "round": round_index+1, "max_sampling_scoring_logp_delta": max_sampling_delta,
                      "trajectory_seeds": [e["seed"] for e in episodes]}


def update_group(actor: GRPOPolicy, optimizer, episodes: list, stats: dict, assets: dict) -> dict:
    import torch

    if adapter_fingerprint(actor.model) != stats["behavior_fingerprint"]:
        raise RuntimeError("Stale behavior group: policy changed before the update")
    validate_group(episodes, actor.config.group_size)
    optimizer.zero_grad(set_to_none=True)
    if stats["zero_variance"]:
        return {"optimizer_step": False, "reason": "zero_return_variance", "gradient_norm": 0.0,
                "loss": 0.0, "max_ratio_deviation": 0.0, "clipped_fraction": 0.0,
                "after_fingerprint": stats["behavior_fingerprint"]}
    loss_value, scored_tokens, clipped, max_deviation = 0.0, 0, 0, 0.0
    for episode, advantage in zip(episodes, stats["advantages"]):
        trajectory_tokens = sum(len(t["generation"]["generated_ids"]) for t in episode["transitions"])
        for record in episode["transitions"]:
            if record["generation"]["behavior_fingerprint"] != stats["behavior_fingerprint"]:
                raise RuntimeError("Mixed behavior policies within a group")
            current = actor.score(record, assets)
            previous = torch.tensor(record["generation"]["old_log_probs"], device=current.device)
            ratios = (current.detach()-previous).exp()
            deviation = float((ratios-1).abs().max())
            if deviation > 1e-4:
                raise RuntimeError("Behavior replay differs before the first optimizer step")
            max_deviation = max(max_deviation, deviation)
            clipped += int(((ratios-1).abs() > actor.config.clip_epsilon).sum())
            scored_tokens += current.numel()
            terms = clipped_token_terms(current, previous, advantage, actor.config.clip_epsilon)
            # Sum turn losses into a generated-token mean per trajectory, then
            # average those means over G. Never step between turns or samples.
            loss = terms.sum() / (len(episodes)*trajectory_tokens)
            loss_value += float(loss.detach())
            loss.backward()
            del current, previous, terms, loss
    trainable = [p for p in actor.model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
    if any(p.grad is not None for p in actor.model.parameters() if not p.requires_grad):
        raise RuntimeError("Frozen base/vision weights received gradients")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after = adapter_fingerprint(actor.model)
    if float(norm) == 0 or after == stats["behavior_fingerprint"]:
        raise RuntimeError("Nonconstant reward group failed to update the adapter")
    return {"optimizer_step": True, "loss": loss_value, "gradient_norm": float(norm),
            "scored_assistant_tokens": scored_tokens, "max_ratio_deviation": max_deviation,
            "clipped_fraction": clipped/scored_tokens, "after_fingerprint": after}


def replay_saved_run(directory: Path) -> int:
    manifest = verify_artifacts(directory, stage=5, artifact_files=ARTIFACT_FILES)
    if manifest.get("protocol") != "move-json-v1" or manifest.get("environment_version") != "grid-v1":
        raise ValueError("Unknown Stage 5 environment/protocol")
    config = Stage5Config(**json.loads((directory / "config.json").read_text()))
    layouts = json.loads((directory / "layouts.json").read_text())
    lookup = {x["layout_id"]: Layout.from_dict(x) for group in layouts.values() for x in group}
    episodes = json.loads((directory / "episodes.json").read_text())["episodes"]
    groups = json.loads((directory / "groups.json").read_text())["groups"]
    with zipfile.ZipFile(directory / "images.zip") as archive:
        names = set(archive.namelist())
        for episode in episodes:
            replay_trajectory(lookup[episode["layout_id"]], episode)
            if any(o["image"] not in names for o in episode["observations"]):
                raise ValueError("Missing episode image")
    for group in groups:
        members = [episodes[i] for i in group["episode_indices"]]
        validate_group(members, config.group_size)
        if (len(members) != config.group_size or lookup[group["layout_id"]].split != "train"
                or any(e["layout_id"] != group["layout_id"] for e in members)
                or len({e["seed"] for e in members}) != config.group_size):
            raise ValueError("Group must contain four independent train-board trajectories from one layout")
        expected = relative_advantages([e["return"] for e in members], config.advantage_epsilon)
        if any(group[key] != value for key, value in expected.items()):
            raise ValueError("Returns or advantages do not match the recorded episodes")
        for episode in members:
            for record in episode["transitions"]:
                generation = record["generation"]
                n = len(generation["generated_ids"])
                if (not n or len(generation["old_log_probs"]) != n or len(generation["sampled_log_probs"]) != n
                        or generation["behavior_fingerprint"] != group["behavior_fingerprint"]
                        or any(not math.isfinite(v) for v in generation["old_log_probs"] + generation["sampled_log_probs"])):
                    raise ValueError("Behavior token records do not align")
    return len(episodes)


def train_remote(run_id: str, raw_config: dict, source_sha256: dict,
                 cache_receipt: dict, expected_source: dict) -> dict:
    import torch

    checks = tensor_checks()  # Differentiable objective checks BEFORE loading the VLM.
    config = Stage5Config(**raw_config)
    source = source_receipt(config)
    if source != expected_source or cache_receipt["revision"] != config.model_revision:
        raise RuntimeError("Model cache or SFT checkpoint changed after CPU preparation")
    temporary = run_path(Path("/tmp"), run_id)
    unpack_adapter(run_path(RUNS_PATH, config.sft_run_id) / "adapter.zip", temporary / "source")
    started = time.monotonic()
    actor = GRPOPolicy(config, temporary / "source")
    torch.cuda.reset_peak_memory_stats()
    layouts = {split: [x.to_dict() for x in make_layouts(split, n, config.dataset_seed)]
               for split, n in (("train", config.train_layouts), ("eval", config.episodes))}
    if {x["maze_id"] for x in layouts["train"]} & {x["maze_id"] for x in layouts["eval"]}:
        raise RuntimeError("Training and evaluation mazes overlap")
    data = {"layouts": layouts, "episodes": [], "metrics": {}}
    assets, groups = {}, []
    _evaluate(actor, data, assets, "sft")
    optimizer = torch.optim.AdamW((p for p in actor.model.parameters() if p.requires_grad),
                                 lr=config.learning_rate, weight_decay=0.0)
    mask_inspection = None
    for index in range(config.updates):
        round_started = time.monotonic()
        episodes, stats = collect_group(actor, Layout.from_dict(layouts["train"][index % config.train_layouts]), index, assets)
        stats["episode_indices"] = list(range(len(data["episodes"]), len(data["episodes"])+len(episodes)))
        data["episodes"].extend(episodes)
        if mask_inspection is None:
            record = max((t for e in episodes for t in e["transitions"]), key=lambda t: len(t["generation"]["prompt_ids"]))
            generation = record["generation"]
            p, n = len(generation["prompt_ids"]), len(generation["generated_ids"])
            mask_inspection = {"messages": record["messages"], "prompt_ids": generation["prompt_ids"],
                               "generated_ids": generation["generated_ids"], "loss_mask": [0]*p + [1]*n,
                               "prediction_positions": list(range(p-1, p+n-1)),
                               "image_paths": generation["image_paths"], "old_log_probs": generation["old_log_probs"]}
        stats["update"] = update_group(actor, optimizer, episodes, stats, assets)
        stats["seconds"] = time.monotonic()-round_started
        groups.append(stats)
        print(f"Round {index+1}: std={stats['std']:.4f} update={stats['update']}", flush=True)
    _evaluate(actor, data, assets, "grpo")
    rollouts = [e for e in data["episodes"] if e["policy"] == "rollout"]
    data["metrics"]["training_rollouts"] = summarize(rollouts)
    actual_updates = sum(g["update"]["optimizer_step"] for g in groups)
    verification = {**checks, "same_board_groups": True, "distinct_trajectory_rng_seeds": True,
                    "behavior_policy_frozen_within_groups": True, "fresh_collection_each_round": True,
                    "assistant_only_exact_generated_ids": True, "train_eval_mazes_disjoint": True,
                    "frozen_base_and_vision": True, "all_episodes_replayed": True,
                    "sampling_scoring_within_tolerance": True}
    environment = {"mode": "gpu", "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
                   "packages": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft")},
                   "total_seconds": time.monotonic()-started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "producer_task_id": os.environ["MODAL_TASK_ID"], "modal_image_id": os.environ["MODAL_IMAGE_ID"],
                   "rollout_generation_config": actor.rollout_generation.to_dict(),
                   "evaluation_generation_config": actor.generation.to_dict(), "kl_beta": 0.0,
                   "loss_normalization": "mean of per-trajectory generated-token means",
                   "scoring_mode": "eval with autograd; one turn per backward pass"}
    actor.model.save_pretrained(temporary / "adapter", safe_serialization=True, save_embedding_layers=False)
    adapter_zip = io.BytesIO()
    with zipfile.ZipFile(adapter_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((temporary / "adapter").iterdir()):
            if path.is_file():
                archive.write(path, arcname=f"adapter/{path.name}")
    training = {"collection_rounds": len(groups), "optimizer_steps": actual_updates,
                "zero_variance_groups": len(groups)-actual_updates, "verification": verification}
    directory = run_path(RUNS_PATH, run_id)
    directory.mkdir(parents=True, exist_ok=False)
    for name, value in {"config.json": asdict(config), "environment.json": environment, "layouts.json": layouts,
                        "episodes.json": {"episodes": data["episodes"]}, "groups.json": {"groups": groups},
                        "training.json": training, "metrics.json": {"metrics": data["metrics"], "verification": verification},
                        "source-adapter.json": source, "mask-inspection.json": mask_inspection,
                        "model-cache.json": cache_receipt}.items():
        write_json(directory / name, value)
    (directory / "replay.html").write_text(replay_html(data, "Stage 5: SFT and trajectory GRPO"))
    (directory / "images.zip").write_bytes(_images_zip(assets))
    (directory / "adapter.zip").write_bytes(adapter_zip.getvalue())
    (directory / "policy-prompt.txt").write_text(TOOL_PROMPT + "\n")
    (directory / "chat-template.txt").write_text(actor.processor.chat_template)
    manifest = write_manifest(directory, {"source_sha256": source_sha256, "environment_version": "grid-v1",
                                         "protocol": "move-json-v1", "mode": "gpu"},
                              stage=5, artifact_files=ARTIFACT_FILES)
    replay_saved_run(directory)
    return {"manifest": manifest, "config": asdict(config), "environment": environment,
            "metrics": data["metrics"], "verification": verification, "training": training,
            "files": {name: base64.b64encode((directory/name).read_bytes()).decode("ascii")
                      for name in (*ARTIFACT_FILES, "manifest.json")}}

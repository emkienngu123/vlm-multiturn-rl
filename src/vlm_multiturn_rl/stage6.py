"""Bounded GRPO experiments, durable round checkpoints, and paired evaluation."""

import base64
from dataclasses import asdict, dataclass, replace
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import time
import zipfile

from . import checkpoints
from .grpo import relative_advantages, validate_group
from .gridworld import Layout, make_layouts
from .reporting import compare_policies, report_html, write_curves
from .stage3 import replay_html
from .stage4 import _evaluate, _images_zip
from .stage5 import (GRPOPolicy, Stage5Config, adapter_fingerprint, collect_group,
                     source_receipt, unpack_adapter, update_group)
from .trajectories import TOOL_PROMPT, replay_trajectory
from .workspace import RUNS_PATH, run_path, sha256_file, verify_artifacts, write_json, write_manifest


ARTIFACT_FILES = (
    "config.json", "environment.json", "history.json", "comparison.json", "training.json",
    "report.html", "replay.html", "learning-curves.svg", "learning-curves.png", "learning-curves.csv",
    "images.zip", "adapter.zip", "checkpoint.zip", "source-adapter.json", "policy-prompt.txt", "model-cache.json",
)


@dataclass(frozen=True)
class Stage6Config(Stage5Config):
    checkpoint_every: int = 1
    eval_every: int = 5
    resume_run_id: str = ""
    resume_round: int = 0
    verify_resume: bool = False

    def __post_init__(self):
        super().__post_init__()
        for key in ("checkpoint_every", "eval_every"):
            if type(getattr(self, key)) is not int or not 1 <= getattr(self, key) <= 20:
                raise ValueError(f"{key} must be in [1, 20]")
        if type(self.resume_round) is not int or not 0 <= self.resume_round <= 20:
            raise ValueError("resume_round must be in [0, 20]")
        if self.resume_run_id:
            run_path(Path("."), self.resume_run_id)
        elif self.resume_round:
            raise ValueError("resume_round requires resume_run_id")
        if type(self.verify_resume) is not bool:
            raise ValueError("verify_resume must be boolean")


def find_checkpoint(config: Stage6Config, root: Path = RUNS_PATH) -> Path:
    parent = run_path(root, config.resume_run_id) / "checkpoints"
    if config.resume_round:
        directory = parent / f"round-{config.resume_round:04d}"
    else:
        candidates = sorted(p.parent for p in parent.glob("round-*/manifest.json"))
        if not candidates:
            raise ValueError("No completed Stage 6 checkpoint found; adapter-only Stage 4/5 runs cannot resume Adam")
        directory = candidates[-1]
    checkpoints.metadata(directory)
    return directory


def prepare_experiment(raw_config: dict, source_hashes: dict, root: Path = RUNS_PATH) -> dict:
    config = Stage6Config(**raw_config)
    resume = None
    if config.resume_run_id:
        path = find_checkpoint(config, root)
        info = checkpoints.metadata(path)
        if not config.sft_run_id:
            config = replace(config, sft_run_id=info["config"]["sft_run_id"])
        checkpoints.validate_resume_config(info["config"], asdict(config), info["completed_round"])
        if info["source_sha256"] != source_hashes:
            raise ValueError("Training code or pinned dependencies changed; exact resume requires the checkpoint's source")
        resume = {"run_id": config.resume_run_id, "round": info["completed_round"],
                  "manifest_sha256": sha256_file(path / "manifest.json")}
    source = source_receipt(config, root)
    if resume and info["source"] != source:
        raise ValueError("Original SFT source changed")
    return {"config": asdict(config), "source": source, "resume": resume}


def runtime_contract() -> dict:
    import torch

    return {"python": platform.python_version(), "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy", "pillow", "matplotlib")}}


def continuation_signature(episodes: list, actor, optimizer) -> dict:
    return {"trajectories": [{"seed": e["seed"], "return": e["return"],
                              "generated_ids": [t["generation"]["generated_ids"] for t in e["transitions"]]}
                             for e in episodes], "after_adapter": adapter_fingerprint(actor.model),
            "after_optimizer": checkpoints.tree_digest(optimizer.state_dict())}


def probe_continuation(actor, optimizer, layouts: list, next_round: int) -> dict:
    import torch

    parameters = {name: p.detach().cpu().clone() for name, p in actor.model.named_parameters() if p.requires_grad}
    old_optimizer = checkpoints.clone_cpu(optimizer.state_dict())
    rng = checkpoints.capture_rng()
    before = adapter_fingerprint(actor.model)
    assets = {}
    episodes, stats = collect_group(actor, Layout.from_dict(layouts[next_round % len(layouts)]), next_round, assets)
    update_group(actor, optimizer, episodes, stats, assets)
    signature = continuation_signature(episodes, actor, optimizer)
    with torch.no_grad():
        for name, p in actor.model.named_parameters():
            if name in parameters:
                p.copy_(parameters[name].to(p.device))
    optimizer.load_state_dict(old_optimizer)
    optimizer.zero_grad(set_to_none=True)
    checkpoints.restore_rng(rng)
    if (adapter_fingerprint(actor.model) != before
            or checkpoints.tree_digest(optimizer.state_dict()) != checkpoints.tree_digest(old_optimizer)):
        raise RuntimeError("Continuation-probe rollback failed")
    return {"next_round_index": next_round, "signature": signature, "rollback_verified": True}


def evaluate_round(actor, history: dict, assets: dict, completed_round: int):
    name = f"grpo_round_{completed_round}"
    _evaluate(actor, history, assets, name)
    history["evaluations"].append({"round": completed_round, "policy": name, "metrics": history["metrics"][name]})


def validate_history(history: dict) -> None:
    lookup = {x["layout_id"]: Layout.from_dict(x) for group in history["layouts"].values() for x in group}
    if {x["maze_id"] for x in history["layouts"]["train"]} & {x["maze_id"] for x in history["layouts"]["eval"]}:
        raise ValueError("Train/eval mazes overlap")
    for episode in history["episodes"]:
        replay_trajectory(lookup[episode["layout_id"]], episode)
    used = []
    for index, group in enumerate(history["groups"]):
        if group["round"] != index+1:
            raise ValueError("Round sequence has a gap")
        members = [history["episodes"][i] for i in group["episode_indices"]]
        validate_group(members)
        if lookup[group["layout_id"]].split != "train" or any(e["layout_id"] != group["layout_id"] for e in members):
            raise ValueError("Training group contains a wrong layout")
        expected = relative_advantages([e["return"] for e in members])
        if any(group[key] != value for key, value in expected.items()):
            raise ValueError("Group returns/advantages differ from replay")
        update = group["update"]
        if update["optimizer_step"] == group["zero_variance"]:
            raise ValueError("Skip/update does not match reward variance")
        if index and group["behavior_fingerprint"] != history["groups"][index-1]["update"]["after_fingerprint"]:
            raise ValueError("Behavior-policy lineage broke between rounds")
        for e in members:
            for t in e["transitions"]:
                g = t["generation"]
                if len(g["old_log_probs"]) != len(g["generated_ids"]) or g["behavior_fingerprint"] != group["behavior_fingerprint"]:
                    raise ValueError("Behavior-token record mismatch")
        used.extend(group["episode_indices"])
    rollout_indices = [i for i, e in enumerate(history["episodes"]) if e["policy"] == "rollout"]
    if sorted(used) != rollout_indices or len(used) != len(set(used)):
        raise ValueError("Training episodes must belong to exactly one group")


def replay_saved_run(directory: Path) -> int:
    verify_artifacts(directory, stage=6, artifact_files=ARTIFACT_FILES)
    history = json.loads((directory / "history.json").read_text())
    validate_history(history)
    with zipfile.ZipFile(directory / "images.zip") as archive:
        if any(o["image"] not in archive.namelist() for e in history["episodes"] for o in e["observations"]):
            raise ValueError("Missing replay image")
    comparison = compare_policies(history["episodes"], {"base": "base", "sft": "sft", "grpo": history["evaluations"][-1]["policy"]}, history["layouts"]["eval"])
    if comparison != json.loads((directory / "comparison.json").read_text()):
        raise ValueError("Comparison differs from paired episode outcomes")
    return len(history["episodes"])


def train_remote(run_id: str, prepared: dict, source_hashes: dict, cache_receipt: dict, commit_checkpoint) -> dict:
    import torch

    if prepare_experiment(prepared["config"], source_hashes) != prepared:
        raise RuntimeError("Source/checkpoint changed after CPU preparation")
    config = Stage6Config(**prepared["config"])
    if cache_receipt["revision"] != config.model_revision:
        raise RuntimeError("Model cache differs from requested revision")
    directory = run_path(RUNS_PATH, run_id)
    directory.mkdir(parents=True, exist_ok=False)
    temporary = run_path(Path("/tmp"), run_id)
    resume_path = find_checkpoint(config) if prepared["resume"] else None
    archive_path = resume_path / "adapter.zip" if resume_path else run_path(RUNS_PATH, config.sft_run_id) / "adapter.zip"
    unpack_adapter(archive_path, temporary / "adapter")
    started = time.monotonic()
    actor = GRPOPolicy(config, temporary / "adapter")
    runtime = runtime_contract()
    optimizer = torch.optim.AdamW((p for p in actor.model.parameters() if p.requires_grad), lr=config.learning_rate, weight_decay=0.0)
    resume_checks, reference = {}, None
    if resume_path:
        info = checkpoints.metadata(resume_path)
        if runtime != info["runtime"]:
            raise RuntimeError("Runtime versions/GPU changed; exact continuation is not verified")
        history, assets = checkpoints.load_history(resume_path)
        validate_history(history)
        resume_checks = checkpoints.restore_training(resume_path, actor, optimizer)
        reference = info["continuation_reference"]
        start_round = info["completed_round"]
        print(f"Restored round {start_round}, including {resume_checks['optimizer_steps_restored']} Adam steps", flush=True)
    else:
        layouts = {split: [x.to_dict() for x in make_layouts(split, n, config.dataset_seed)]
                   for split, n in (("train", config.train_layouts), ("eval", config.episodes))}
        history = {"layouts": layouts, "episodes": [], "metrics": {}, "groups": [], "evaluations": []}
        assets = {}
        with actor.model.disable_adapter():
            _evaluate(actor, history, assets, "base")
        _evaluate(actor, history, assets, "sft")
        history["evaluations"].append({"round": 0, "policy": "sft", "metrics": history["metrics"]["sft"]})
        start_round = 0
    job_peak = torch.cuda.max_memory_allocated()
    latest_checkpoint = None
    for index in range(start_round, config.updates):
        torch.cuda.reset_peak_memory_stats()
        phase_started = time.monotonic()
        episodes, stats = collect_group(actor, Layout.from_dict(history["layouts"]["train"][index % config.train_layouts]), index, assets)
        stats["collection_seconds"] = time.monotonic()-phase_started
        stats["episode_indices"] = list(range(len(history["episodes"]), len(history["episodes"])+len(episodes)))
        history["episodes"].extend(episodes)
        phase_started = time.monotonic()
        stats["update"] = update_group(actor, optimizer, episodes, stats, assets)
        stats["update_seconds"] = time.monotonic()-phase_started
        stats["generated_tokens"] = sum(len(t["generation"]["generated_ids"]) for e in episodes for t in e["transitions"])
        stats["rollout_tokens_per_second"] = stats["generated_tokens"]/stats["collection_seconds"]
        stats["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        job_peak = max(job_peak, stats["peak_allocated_bytes"])
        history["groups"].append(stats)
        if index == start_round and reference:
            if reference["next_round_index"] != index or continuation_signature(episodes, actor, optimizer) != reference["signature"]:
                raise RuntimeError("Resumed next round differs from live uninterrupted continuation")
            resume_checks["live_continuation_matches"] = True
        completed = index+1
        if completed % config.eval_every == 0 or completed == config.updates:
            evaluate_round(actor, history, assets, completed)
        proof = probe_continuation(actor, optimizer, history["layouts"]["train"], completed) if config.verify_resume and completed == config.updates else None
        job_peak = max(job_peak, torch.cuda.max_memory_allocated())
        if completed % config.checkpoint_every == 0 or completed == config.updates:
            latest_checkpoint = directory / "checkpoints" / f"round-{completed:04d}"
            checkpoints.save_checkpoint(latest_checkpoint, actor, optimizer, history, assets, prepared["source"], source_hashes, runtime, proof)
            commit_checkpoint()
            print(f"Committed {latest_checkpoint}", flush=True)
        print(f"Round {completed}/{config.updates}: variance={stats['std']**2:.4f} step={stats['update']['optimizer_step']}", flush=True)
    validate_history(history)
    comparison = compare_policies(history["episodes"], {"base": "base", "sft": "sft", "grpo": history["evaluations"][-1]["policy"]}, history["layouts"]["eval"])
    steps = sum(g["update"]["optimizer_step"] for g in history["groups"])
    training = {"collection_rounds": len(history["groups"]), "optimizer_steps": steps,
                "zero_variance_groups": len(history["groups"])-steps,
                "new_rounds_this_job": config.updates-start_round, "resume_verification": resume_checks,
                "continuation_probe_saved": config.verify_resume,
                "latest_checkpoint": str(latest_checkpoint.relative_to(directory))}
    environment = {**runtime, "mode": "gpu", "total_seconds": time.monotonic()-started,
                   "peak_allocated_bytes": job_peak, "evaluation_decoding": actor.generation.to_dict(),
                   "rollout_decoding": actor.rollout_generation.to_dict(),
                   "compute_time_note": "Current job only; excludes CPU preparation and final packaging"}
    for name, value in {"config.json": asdict(config), "environment.json": environment, "history.json": history,
                        "comparison.json": comparison, "training.json": training,
                        "source-adapter.json": prepared["source"], "model-cache.json": cache_receipt}.items():
        write_json(directory / name, value)
    write_curves(directory, history["groups"], history["evaluations"])
    (directory / "report.html").write_text(report_html(comparison, training, environment))
    display = {**history, "metrics": comparison["policies"]}
    (directory / "replay.html").write_text(replay_html(display, "Stage 6: paired comparison and training replays"))
    (directory / "images.zip").write_bytes(_images_zip(assets))
    (directory / "adapter.zip").write_bytes((latest_checkpoint / "adapter.zip").read_bytes())
    (directory / "policy-prompt.txt").write_text(TOOL_PROMPT+"\n")
    with zipfile.ZipFile(directory / "checkpoint.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in (*checkpoints.FILES, "manifest.json"):
            archive.write(latest_checkpoint / name, arcname=f"{latest_checkpoint.name}/{name}")
    manifest = write_manifest(directory, {"kind": "report", "protocol": "move-json-v1", "environment_version": "grid-v1",
                                         "source_sha256": source_hashes, "mode": "gpu"}, stage=6, artifact_files=ARTIFACT_FILES)
    replay_saved_run(directory)
    return {"manifest": manifest, "config": asdict(config), "environment": environment,
            "metrics": comparison["policies"], "training": training,
            "verification": {"paired_evaluation": True, "all_episodes_replayed": True,
                             "checkpoint_committed": True, **resume_checks},
            "files": {name: base64.b64encode((directory/name).read_bytes()).decode("ascii")
                      for name in (*ARTIFACT_FILES, "manifest.json")}}

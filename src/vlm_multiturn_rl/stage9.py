"""Browser SFT bridge: reuse the audited token loss, never grid demonstrations.

Imports stay usable in the standard-library-only local lesson environment.
"""

import base64
from dataclasses import asdict, dataclass, fields
import io
import json
import math
from pathlib import Path
import tempfile
import time
import zipfile

from .browser import (BROWSER_PROMPT, MINIWOB_REVISION, MINIWOB_ARCHIVE_SHA256, BrowserConfig, cases, collect_episode,
                      parse_action, summarize, validate_episode)
from .stage4 import HistoryVLM, _images_zip, _probe, train_adapter
from .trajectories import image_paths
from . import stage8
from .workspace import (RUNS_PATH, run_path, sha256_file, verify_artifacts,
                        write_json, write_manifest)


ARTIFACT_FILES = (*stage8.GPU_ARTIFACT_FILES, "training.json", "mask-inspection.json",
                  "adapter.zip", "comparison.json")


@dataclass(frozen=True)
class BrowserSFTConfig(BrowserConfig):
    updates: int = 2
    gradient_accumulation: int = 2
    learning_rate: float = 2e-4
    lora_rank: int = 8
    diagnostic_sft_run_id: str = ""

    def __post_init__(self):
        super().__post_init__()
        for name, maximum in (("updates", 100), ("gradient_accumulation", 4)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1, {maximum}]")
        if (not math.isfinite(self.learning_rate) or not 1e-6 <= self.learning_rate <= 1e-3
                or self.lora_rank != 8):
            raise ValueError("Use finite learning_rate in [1e-6, 1e-3] and language LoRA rank 8")
        if not isinstance(self.diagnostic_sft_run_id, str): raise ValueError("Diagnostic run ID must be a string")
        if self.diagnostic_sft_run_id: run_path(Path("."), self.diagnostic_sft_run_id)


def browser_config(config):
    return BrowserConfig(**{f.name: getattr(config, f.name) for f in fields(BrowserConfig)})


def validate_demonstrations(case_list, episodes, examples):
    if examples != stage8.demonstrations(episodes) or not examples:
        raise ValueError("Targets must be exactly the successful training oracle actions")
    lookup = {c["case_id"]: c for c in case_list}
    if len(lookup) != len(case_list): raise ValueError("Duplicate case IDs")
    for example in examples:
        if lookup[example["case_id"]]["split"] != "train":
            raise ValueError("Evaluation target leaked into training")
        if parse_action(example["target"], 498, 321) is None:
            raise ValueError("Malformed teacher action")


def validate_source(config, directory):
    """Run before requesting a GPU. Older/incomplete/changed contracts fail closed."""
    manifest = verify_artifacts(directory, stage=8, artifact_files=stage8.ARTIFACT_FILES)
    if manifest.get("mode") != "cpu" or manifest.get("protocol") != "browser-json-v1":
        raise ValueError("Browser SFT requires a completed Stage 8 CPU preflight")
    expected = json.loads(json.dumps(asdict(browser_config(config))))
    if json.loads((directory/"config.json").read_text()) != expected:
        raise ValueError("SFT/source browser configurations differ; use the source's --episodes/seed/pixel settings")
    if (directory/"policy-prompt.txt").read_text() != BROWSER_PROMPT+"\n":
        raise ValueError("Source browser protocol changed")
    environment = json.loads((directory/"environment.json").read_text())
    if (environment["miniwob_revision"] != MINIWOB_REVISION
            or environment["miniwob_archive_sha256"] != MINIWOB_ARCHIVE_SHA256):
        raise ValueError("Source MiniWoB assets changed; rerun preflight")
    stage8.replay_saved_run(directory)
    case_list = json.loads((directory/"cases.json").read_text())
    if case_list != cases(config): raise ValueError("Source task split changed")
    episodes = json.loads((directory/"episodes.json").read_text())
    examples = json.loads((directory/"demonstrations.json").read_text())
    validate_demonstrations(case_list, episodes, examples)
    receipt = {"run_id": directory.name, "manifest_sha256": sha256_file(directory/"manifest.json"),
            "training_cases": sorted({e["case_id"] for e in examples}), "training_actions": len(examples),
            "requirements_sha256": sha256_file(directory/"requirements-browser.txt")}
    if config.diagnostic_sft_run_id:
        reference = run_path(directory.parent, config.diagnostic_sft_run_id)
        replay_saved_run(reference)
        previous = BrowserSFTConfig(**json.loads((reference/"config.json").read_text()))
        for field in (*[f.name for f in fields(BrowserConfig)], "learning_rate", "gradient_accumulation", "lora_rank"):
            if json.dumps(getattr(previous, field)) != json.dumps(getattr(config, field)):
                raise ValueError(f"Diagnostic reference uses a different {field}")
        previous_source = json.loads((reference/"cpu-source.json").read_text())
        if previous_source["manifest_sha256"] != receipt["manifest_sha256"]:
            raise ValueError("Duration comparison requires exactly the same source demonstrations")
        if previous.updates >= config.updates:
            raise ValueError("Duration comparison requires more updates than the reference")
        receipt["diagnostic_reference"] = {"run_id": reference.name, "updates": previous.updates,
            "manifest_sha256": sha256_file(reference/"manifest.json"),
            "adapter_sha256": sha256_file(reference/"adapter.zip")}
    return receipt


def training_rollout_metrics(episodes):
    """Separate training-fit diagnostics from the immutable evaluation metrics."""
    selected = [e for e in episodes if e["case"]["split"] == "train" and e["policy"] in ("sft_reference", "sft")]
    return summarize([{**e, "case": {**e["case"], "split": "eval"}} for e in selected])


def _teacher_forced_mean(actor, examples, assets):
    # Same action-normalized objective as training, averaged over ALL train examples.
    losses = []
    for example in examples:
        loss, logits = _probe(actor, example, assets)
        losses.append(loss)
        del logits
    return sum(losses)/len(losses)


def _training_rollouts(actor, config, data, assets, policy):
    for case in data["cases"]:
        if case["split"] == "train":
            episode = collect_episode(config, case, policy, assets, actor=actor.act)
            validate_episode(episode)
            stage8.reexecute_episode(episode)
            data["episodes"].append(episode)
            print(f"Training diagnostic {policy} {case['case_id']}: success={episode['success']}", flush=True)


def latest_local_source(root, config):
    for directory in sorted(root.glob("stage8-*"), reverse=True):
        try: validate_source(config, directory)
        except (OSError, ValueError, KeyError): continue
        return directory.name
    raise ValueError("No matching completed CPU browser preflight. Run browser-check --episodes 2 first.")


class BrowserSFTActor(HistoryVLM):
    def sft_batch(self, example, assets):
        # Only inspection metadata is adapted to the shared trainer's interface.
        # The actual prompt/target and ordered screenshots are passed unchanged.
        batch, inspection = super().sft_batch({**example, "layout_id": example["case_id"]}, assets)
        inspection["case_id"] = inspection.pop("layout_id")
        return batch, inspection


def validate_mask(inspection):
    start, end = inspection["supervised_span"]
    ids, labels = inspection["input_ids"], inspection["labels"]
    if (not 0 < start < end == len(ids) == len(labels)
            or labels[:start] != [-100]*start or labels[start:] != ids[start:]
            or ids[-1] != inspection["eos_token_id"]
            or start != inspection["prompt_tokens"] or end-start != inspection["target_tokens"]):
        raise ValueError("Current assistant action plus EOS must be the only supervised span")
    if len(inspection["image_paths"]) != inspection["turn"]:
        raise ValueError("Changing-image history has missing or reordered observations")


def paired_comparison(episodes):
    for policy in ("base", "sft"):
        ids = [e["case"]["case_id"] for e in episodes if e["policy"] == policy and e["case"]["split"] == "eval"]
        if len(ids) != len(set(ids)): raise ValueError("Duplicate paired evaluation case")
    policies = {p: {e["case"]["case_id"]: e for e in episodes
                    if e["policy"] == p and e["case"]["split"] == "eval"} for p in ("base", "sft")}
    if not policies["base"] or policies["base"].keys() != policies["sft"].keys():
        raise ValueError("Base/SFT evaluation must use identical held-out cases")
    rows = []
    for case_id in sorted(policies["base"]):
        base, sft = (policies[p][case_id] for p in ("base", "sft"))
        if (base["case"] != sft["case"] or
                base["observations"][0] != {**sft["observations"][0], "image": base["observations"][0]["image"]}):
            raise ValueError("Paired policies did not receive the same initial instruction and pixels")
        rows.append({"case_id": case_id, "task": base["case"]["task"],
                     "base_success": base["success"], "sft_success": sft["success"],
                     "base_steps": base["steps"], "sft_steps": sft["steps"],
                     "base_invalid": base["invalid_actions"], "sft_invalid": sft["invalid_actions"]})
    return {"paired_cases": len(rows), "sft_only_successes": sum(r["sft_success"] and not r["base_success"] for r in rows),
            "base_only_successes": sum(r["base_success"] and not r["sft_success"] for r in rows), "cases": rows,
            "scope": "Held-out instance seeds within three familiar tasks; six cases are a diagnostic, not broad web generalization."}


def replay_saved_run(directory):
    verify_artifacts(directory, stage=9, artifact_files=ARTIFACT_FILES)
    config = BrowserSFTConfig(**json.loads((directory/"config.json").read_text()))
    case_list = json.loads((directory/"cases.json").read_text())
    if case_list != cases(config): raise ValueError("Saved browser task split changed")
    episodes = json.loads((directory/"episodes.json").read_text())
    with zipfile.ZipFile(directory/"images.zip") as archive:
        names = set(archive.namelist())
        for episode in episodes:
            validate_episode(episode)
            if episode["case"] not in case_list: raise ValueError("Unregistered case")
            if any(o["image"] not in names for o in episode["observations"]):
                raise ValueError("Missing screenshot")
    examples = json.loads((directory/"demonstrations.json").read_text())
    validate_demonstrations(case_list, episodes, examples)
    inspections = json.loads((directory/"mask-inspection.json").read_text())
    if len(inspections) != len(examples): raise ValueError("Not all training masks were inspected")
    for example, inspection in zip(examples, inspections):
        validate_mask(inspection)
        if (inspection["case_id"] != example["case_id"] or inspection["turn"] != example["turn"]
                or inspection["target"] != example["target"]
                or inspection["image_paths"] != image_paths(example["messages"])):
            raise ValueError("Mask inspection differs from training example/image order")
    if summarize(episodes) != json.loads((directory/"metrics.json").read_text()):
        raise ValueError("Metrics differ from saved trajectories")
    if paired_comparison(episodes) != json.loads((directory/"comparison.json").read_text()):
        raise ValueError("Paired comparison differs from saved trajectories")
    training = json.loads((directory/"training.json").read_text())
    if len(training["updates"]) != config.updates or not all(training["verification"].values()):
        raise ValueError("Incomplete SFT validation")
    if config.diagnostic_sft_run_id:
        diagnostics = training["duration_diagnostics"]
        if diagnostics["training_rollouts"] != training_rollout_metrics(episodes):
            raise ValueError("Training diagnostics differ from saved trajectories")
        for policy in ("sft_reference", "sft"):
            ids = [e["case"]["case_id"] for e in episodes if e["case"]["split"] == "train" and e["policy"] == policy]
            if len(ids) != len(set(ids)) or set(ids) != {c["case_id"] for c in case_list if c["split"] == "train"}:
                raise ValueError("Incomplete or duplicated training-fit diagnostic cases")
        for loss in diagnostics["mean_teacher_forced_loss"].values():
            if not math.isfinite(loss) or loss < 0: raise ValueError("Invalid teacher-forced diagnostic loss")
    return len(episodes)


def train_remote(run_id, raw_config, source, cache_receipt, cpu_run_id, expected_receipt):
    import torch

    config = BrowserSFTConfig(**raw_config)
    directory = run_path(RUNS_PATH, cpu_run_id)
    receipt = validate_source(config, directory)
    if receipt != expected_receipt: raise ValueError("CPU source changed after preparation")
    if sha256_file(Path("/root/requirements-browser.txt")) != receipt["requirements_sha256"]:
        raise ValueError("Browser dependency lock changed since preflight")
    data, assets, _ = stage8.load_cpu(browser_config(config), cpu_run_id)
    examples = stage8.demonstrations(data["episodes"])
    if cache_receipt["revision"] != config.model_revision: raise ValueError("Model revision changed")
    started = time.monotonic()
    actor = BrowserSFTActor(config)
    for case in data["cases"]:
        if case["split"] == "eval":
            episode = collect_episode(config, case, "base", assets, actor=actor.act)
            validate_episode(episode)
            data["episodes"].append(episode)
    # Audit every training target, including the longest changing-image history.
    inspections = []
    for example in examples:
        batch, inspection = actor.sft_batch(example, assets)
        validate_mask(inspection)
        inspections.append(inspection)
        del batch
    diagnostic_losses = {}
    if config.diagnostic_sft_run_id:
        from peft import PeftModel
        from .stage5 import unpack_adapter
        from .checkpoints import capture_rng, restore_rng

        rng = capture_rng()
        base_probe_loss, base_probe_logits = _probe(actor, examples[0], assets)
        with tempfile.TemporaryDirectory() as temporary:
            unpack_adapter(run_path(RUNS_PATH, config.diagnostic_sft_run_id)/"adapter.zip", Path(temporary)/"adapter")
            actor.model = PeftModel.from_pretrained(actor.model, Path(temporary)/"adapter", is_trainable=False).eval()
            diagnostic_losses["sft_reference"] = _teacher_forced_mean(actor, examples, assets)
            _training_rollouts(actor, config, data, assets, "sft_reference")
            # Diagnostic evaluation must not change the fresh-base training run.
            actor.model = actor.model.unload().eval()
        restore_rng(rng)
        restored_loss, restored_logits = _probe(actor, examples[0], assets)
        if (abs(base_probe_loss-restored_loss) > 1e-4
                or not torch.allclose(base_probe_logits, restored_logits, rtol=1e-4, atol=1e-4)):
            raise RuntimeError("Diagnostic adapter was not fully unloaded from the base")
    with tempfile.TemporaryDirectory() as temporary:
        adapter_dir = Path(temporary)/"adapter"
        training = train_adapter(actor, examples, assets, adapter_dir)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(adapter_dir.iterdir()): archive.write(path, "adapter/"+path.name)
        adapter_bytes = buffer.getvalue()
        training["adapter_files_bytes"] = {p.name: p.stat().st_size for p in adapter_dir.iterdir()}
    if config.diagnostic_sft_run_id:
        diagnostic_losses["sft"] = _teacher_forced_mean(actor, examples, assets)
        _training_rollouts(actor, config, data, assets, "sft")
        training["duration_diagnostics"] = {"reference": receipt["diagnostic_reference"],
            "training_rollouts": training_rollout_metrics(data["episodes"]),
            "mean_teacher_forced_loss": diagnostic_losses,
            "current_examples_processed": config.updates*config.gradient_accumulation,
            "current_dataset_passes": config.updates*config.gradient_accumulation/len(examples),
            "reference_unloaded_and_rng_restored": True,
            "scope": "Training fit only; these scores are not held-out evaluation."}
        reference_training = json.loads((run_path(RUNS_PATH, config.diagnostic_sft_run_id)/"training.json").read_text())
        training["duration_diagnostics"]["reference_prefix_max_loss_difference"] = max(
            abs(current["mean_action_loss"]-previous["mean_action_loss"])
            for current, previous in zip(training["updates"], reference_training["updates"]))
    for case in data["cases"]:
        if case["split"] == "eval":
            episode = collect_episode(config, case, "sft", assets, actor=actor.act)
            validate_episode(episode)
            stage8.reexecute_episode(episode)
            data["episodes"].append(episode)
            print(f"SFT {case['case_id']}: success={episode['success']}, invalid={episode['invalid_actions']}", flush=True)
    comparison = paired_comparison(data["episodes"])
    data["metrics"] = summarize(data["episodes"])
    data["checks"].update({"browser_training_performed": True, "training_targets_exclude_eval": True,
                           "all_training_masks_inspected": True, "identical_base_sft_initial_pixels": True,
                           "sft_adapter_reloaded_before_evaluation": True, "sft_actions_reexecuted_in_fresh_browsers": True,
                           **training["verification"]})
    environment = {**stage8.runtime(), "mode": "gpu", "gpu": torch.cuda.get_device_name(),
                   "seconds": time.monotonic()-started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "evaluation_decoding": actor.generation.to_dict(), "model_revision": config.model_revision,
                   "browser_training_performed": True, "training": "assistant-only LoRA SFT; no RL"}
    destination = run_path(RUNS_PATH, run_id)
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in {"config.json": asdict(config), "environment.json": environment,
                        "cases.json": data["cases"], "episodes.json": data["episodes"],
                        "demonstrations.json": examples, "metrics.json": data["metrics"], "checks.json": data["checks"],
                        "training.json": training, "mask-inspection.json": inspections, "comparison.json": comparison,
                        "model-cache.json": cache_receipt, "cpu-source.json": receipt}.items():
        write_json(destination/name, value)
    (destination/"adapter.zip").write_bytes(adapter_bytes)
    (destination/"images.zip").write_bytes(_images_zip(assets))
    (destination/"policy-prompt.txt").write_text(BROWSER_PROMPT+"\n")
    (destination/"requirements-browser.txt").write_bytes(Path("/root/requirements-browser.txt").read_bytes())
    (destination/"chat-template.txt").write_text(actor.processor.chat_template)
    (destination/"report.html").write_text(stage8.report_html(data, assets,
        title="MiniWoB++ browser SFT", description=(
            f"Language-side LoRA, {config.updates} updates on {len(examples)} training actions. "
            "Base and reloaded SFT adapter use identical held-out instances and greedy decoding. "
            "Oracle uses privileged DOM coordinates; policy sees only screenshots/instructions/feedback. "
            "No browser RL yet. Training and token-mask details are in training.json and mask-inspection.json.")))
    if config.diagnostic_sft_run_id:
        from html import escape
        with (destination/"report.html").open("a") as handle:
            handle.write('<h2>SFT duration diagnostics — training instances only</h2><pre>'+
                         escape(json.dumps(training["duration_diagnostics"], indent=2))+'</pre>')
    manifest = write_manifest(destination, {"mode": "gpu", "protocol": "browser-json-v1",
                              "environment_version": "browser-v1", "source_sha256": source},
                              stage=9, artifact_files=ARTIFACT_FILES)
    replay_saved_run(destination)
    return {"manifest": manifest, "metrics": data["metrics"], "training": training, "environment": environment,
            "files": {name: base64.b64encode((destination/name).read_bytes()).decode()
                      for name in (*ARTIFACT_FILES, "manifest.json")}}

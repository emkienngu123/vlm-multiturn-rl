"""Browser lesson one: executable task contracts, demonstrations, and baselines."""

import base64
from dataclasses import asdict
from html import escape
import importlib.metadata
import io
import json
from pathlib import Path
import time
import zipfile

from .browser import (BROWSER_PROMPT, MINIWOB_ARCHIVE_SHA256, MINIWOB_REVISION, BrowserConfig,
                      BrowserWorld, action_text, cases, collect_episode, oracle_actions,
                      summarize, validate_episode)
from .stage4 import HistoryVLM, _images_zip
from .workspace import RUNS_PATH, run_path, sha256_file, verify_artifacts, write_json, write_manifest


ARTIFACT_FILES = ("config.json", "environment.json", "cases.json", "episodes.json", "demonstrations.json",
                  "metrics.json", "checks.json", "report.html", "images.zip", "policy-prompt.txt",
                  "requirements-browser.txt")
GPU_ARTIFACT_FILES = (*ARTIFACT_FILES, "model-cache.json", "cpu-source.json", "chat-template.txt")


def demonstrations(episodes: list) -> list:
    # Each current assistant action appears as a target once. Prior actions are context.
    return [{"case_id": e["case"]["case_id"], "turn": t["step"], "messages": t["messages"], "target": t["text"],
             "source": "privileged_oracle"} for e in episodes
            if e["policy"] == "oracle" and e["case"]["split"] == "train" and e["success"]
            for t in e["transitions"]]


def reexecute_episode(episode: dict) -> None:
    """Replay actual actions in a new browser, without asking a model to regenerate."""
    world = BrowserWorld(episode["case"], episode["max_steps"])
    try:
        from PIL import Image
        import hashlib
        initial = Image.fromarray(world.obs["screenshot"]).convert("RGB")
        if (world.obs["goal"] != episode["observations"][0]["goal"]
                or hashlib.sha256(initial.tobytes()).hexdigest() != episode["observations"][0]["pixel_sha256"]):
            raise RuntimeError("Pinned reset changed the initial screenshot or instruction")
        for turn in episode["transitions"]:
            result = world.step(turn["text"])
            if any(result[k] != turn[k] for k in result):
                raise RuntimeError("Re-executed browser action changed its reward/status")
    finally:
        world.close()


def runtime() -> dict:
    return {"packages": {name: importlib.metadata.version(name) for name in
                          ("browsergym-core", "browsergym-miniwob", "playwright", "gymnasium", "numpy", "pillow")},
            "miniwob_revision": MINIWOB_REVISION, "miniwob_archive_sha256": MINIWOB_ARCHIVE_SHA256,
            "observation": "instruction + screenshots + public action feedback; no DOM/accessibility/element IDs",
            "reward": "BrowserGym sparse binary task reward; no per-step shaping",
            "screenshot_scale": 1.5, "max_actions": 8}


def report_html(data, assets, *, title="MiniWoB++ browser baselines", description=None):
    # Standalone replays, including the final observation and exact raw output.
    rows = []
    for episode in data["episodes"]:
        frames = []
        for index, obs in enumerate(episode["observations"]):
            image = assets[obs["image"]]
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            uri = base64.b64encode(buffer.getvalue()).decode()
            text = episode["transitions"][index]["text"] if index < episode["steps"] else "Final observation"
            frames.append(f'<div><img src="data:image/png;base64,{uri}" width="{obs["width"]}" '
                          f'height="{obs["height"]}"><pre>{escape(text)}</pre></div>')
        rows.append(f'<details><summary>{escape(episode["policy"])} — {escape(episode["case"]["case_id"])} '
                    f'— success={episode["success"]}, actions={episode["steps"]}</summary>'
                    f'<p>{escape(episode["observations"][0]["goal"])}</p><section>'+''.join(frames)+'</section></details>')
    description = description or ('First integration subset: three tasks, held-out instance seeds. '
                                 'No browser SFT or RL has been run in this lesson. Oracle uses privileged DOM coordinates.')
    return ('<!doctype html><meta charset="utf-8"><title>'+escape(title)+'</title>'
            '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto}section{display:flex;gap:16px;overflow:auto}'
            'section div{flex-shrink:0;width:500px}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:20px 0}</style>'
            '<h1>'+escape(title)+'</h1><p>'+escape(description)+'</p>'
            '<h2>Metrics — held-out instances only</h2><pre>'+escape(json.dumps(data["metrics"], indent=2))+'</pre>'
            '<h2>Validation</h2><pre>'+escape(json.dumps(data["checks"], indent=2))+'</pre>'
            '<h2>Episode replays</h2>'+''.join(rows))


def save_report(run_id, config, data, assets, environment, source, extras=None):
    directory = run_path(RUNS_PATH, run_id)
    directory.mkdir(parents=True, exist_ok=False)
    for name, value in {"config.json": asdict(config), "environment.json": environment, "cases.json": data["cases"],
                        "episodes.json": data["episodes"], "demonstrations.json": demonstrations(data["episodes"]),
                        "metrics.json": data["metrics"], "checks.json": data["checks"]}.items():
        write_json(directory/name, value)
    (directory/"policy-prompt.txt").write_text(BROWSER_PROMPT+"\n")
    (directory/"requirements-browser.txt").write_bytes(Path("/root/requirements-browser.txt").read_bytes())
    (directory/"report.html").write_text(report_html(data, assets))
    (directory/"images.zip").write_bytes(_images_zip(assets))
    for name, content in (extras or {}).items():
        (directory/name).write_bytes(content)
    files = GPU_ARTIFACT_FILES if extras else ARTIFACT_FILES
    manifest = write_manifest(directory, {"mode": environment["mode"], "environment_version": "browser-v1",
                              "protocol": "browser-json-v1", "source_sha256": source}, stage=8, artifact_files=files)
    replay_saved_run(directory)
    return {"config": asdict(config), "environment": environment, "metrics": data["metrics"],
            "verification": data["checks"], "manifest": manifest,
            "files": {name: base64.b64encode((directory/name).read_bytes()).decode()
                      for name in (*files, "manifest.json")}}


def replay_saved_run(directory: Path) -> int:
    manifest = json.loads((directory/"manifest.json").read_text())
    files = GPU_ARTIFACT_FILES if manifest.get("mode") == "gpu" else ARTIFACT_FILES
    verify_artifacts(directory, stage=8, artifact_files=files)
    episodes = json.loads((directory/"episodes.json").read_text())
    case_list = json.loads((directory/"cases.json").read_text())
    lookup = {c["case_id"]: c for c in case_list}
    if len(lookup) != len(case_list): raise ValueError("Duplicate browser case IDs")
    train = {(c["task"],c["seed"]) for c in case_list if c["split"] == "train"}
    evaluation = {(c["task"],c["seed"]) for c in case_list if c["split"] == "eval"}
    if train & evaluation: raise ValueError("Browser train/eval instances overlap")
    with zipfile.ZipFile(directory/"images.zip") as archive:
        for episode in episodes:
            validate_episode(episode)
            if lookup[episode["case"]["case_id"]] != episode["case"]: raise ValueError("Case changed")
            if any(o["image"] not in archive.namelist() for o in episode["observations"]):
                raise ValueError("Missing browser screenshot")
    if demonstrations(episodes) != json.loads((directory/"demonstrations.json").read_text()):
        raise ValueError("Demonstrations contain non-training targets")
    if summarize(episodes) != json.loads((directory/"metrics.json").read_text()):
        raise ValueError("Browser summary differs from episodes")
    return len(episodes)


def load_cpu(config, cpu_run_id):
    directory = run_path(RUNS_PATH, cpu_run_id)
    manifest = verify_artifacts(directory, stage=8, artifact_files=ARTIFACT_FILES)
    if manifest.get("mode") != "cpu": raise ValueError("GPU baseline requires the completed CPU preflight")
    if json.loads((directory/"config.json").read_text()) != json.loads(json.dumps(asdict(config))):
        raise ValueError("CPU/GPU browser configurations differ")
    data = {"cases": json.loads((directory/"cases.json").read_text()),
            "episodes": json.loads((directory/"episodes.json").read_text()),
            "checks": json.loads((directory/"checks.json").read_text())}
    from PIL import Image
    assets = {}
    with zipfile.ZipFile(directory/"images.zip") as archive:
        for name in archive.namelist():
            with Image.open(io.BytesIO(archive.read(name))) as im: assets[name] = im.convert("RGB").copy()
    return data, assets, {"run_id": cpu_run_id, "manifest_sha256": sha256_file(directory/"manifest.json")}


def evaluate_cpu(run_id, raw_config, source):
    config = BrowserConfig(**raw_config)
    started = time.monotonic()
    assets, episodes = {}, []
    case_list = cases(config)
    for case in case_list:
        episode = collect_episode(config, case, "oracle", assets)
        validate_episode(episode)
        if not episode["success"]: raise RuntimeError("Privileged oracle failed a selected browser case")
        reexecute_episode(episode)
        episodes.append(episode)
        print(f"Oracle/re-execution passed: {case['case_id']}, {episode['steps']} actions", flush=True)
    for case in case_list:
        if case["split"] == "eval": episodes.append(collect_episode(config, case, "random", assets))
    # Invalid JSON consumes the full cap; wrong submission ends with reward zero.
    control_case = next(c for c in case_list if c["task"] == "login-user")
    invalid = collect_episode(config, control_case, "invalid_control", assets,
                              actor=lambda *_: {"text": "not an action", "generated_ids": None})
    if invalid["steps"] != 8 or invalid["success"] or invalid["invalid_actions"] != 8:
        raise RuntimeError("Invalid-action cap contract failed")
    episodes.append(invalid)
    world = BrowserWorld(control_case)
    try:
        result = world.step(oracle_actions(world)[-1])
        if result["raw_reward"] != -1 or result["reward"] != 0 or not result["terminated"]:
            raise RuntimeError("Wrong submission did not fail")
        try: world.step("invalid")
        except RuntimeError: pass
        else: raise RuntimeError("Terminal environment accepted another action")
        chromium = world.env.unwrapped.browser.version
    finally: world.close()
    checks = {"oracle_solves_all_cases": True, "oracle_actions_reexecuted_in_fresh_browsers": True,
              "deterministic_initial_screenshots_and_goals": True, "coordinate_scale_checked_by_successful_clicks": True,
              "invalid_actions_consume_eight_turns": True, "wrong_submission_returns_zero_and_terminates": True,
              "terminal_environment_rejects_actions": True, "public_observation_whitelist": True,
              "train_eval_seed_pairs_disjoint": True, "training_examples_exclude_eval": True,
              "browser_training_performed": False}
    data = {"cases": case_list, "episodes": episodes, "metrics": summarize(episodes), "checks": checks}
    environment = {**runtime(), "mode": "cpu", "chromium": chromium, "seconds": time.monotonic()-started}
    return save_report(run_id, config, data, assets, environment, source)


def evaluate_gpu(run_id, raw_config, source, cache_receipt, cpu_run_id):
    import torch

    config = BrowserConfig(**raw_config)
    data, assets, receipt = load_cpu(config, cpu_run_id)
    if cache_receipt["revision"] != config.model_revision: raise RuntimeError("Model cache revision changed")
    started = time.monotonic()
    actor = HistoryVLM(config)  # Pinned base model; no grid LoRA adapter is loaded.
    for case in data["cases"]:
        if case["split"] == "eval":
            episode = collect_episode(config, case, "base", assets, actor=actor.act)
            validate_episode(episode)
            reexecute_episode(episode)
            data["episodes"].append(episode)
            print(f"Base {case['case_id']}: success={episode['success']}, invalid={episode['invalid_actions']}", flush=True)
    data["metrics"] = summarize(data["episodes"])
    data["checks"]["base_actions_reexecuted_without_model"] = True
    environment = {**runtime(), "mode": "gpu", "gpu": torch.cuda.get_device_name(),
                   "seconds": time.monotonic()-started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "evaluation_decoding": actor.generation.to_dict(), "model_revision": config.model_revision,
                   "browser_training_performed": False}
    return save_report(run_id, config, data, assets, environment, source,
        {"model-cache.json": json.dumps(cache_receipt, indent=2).encode(),
         "cpu-source.json": json.dumps(receipt, indent=2).encode(),
         "chat-template.txt": actor.processor.chat_template.encode()})

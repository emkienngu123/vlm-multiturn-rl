"""Training-only difficulty probes on new MiniWoB templates; no optimization."""

import base64
from dataclasses import asdict, dataclass, fields
from html import escape
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile

from .browser import (BrowserConfig, HARD_TASKS, MINIWOB_ARCHIVE_SHA256, MINIWOB_REVISION,
                      browser_prompt, collect_episode, summarize, validate_episode)
from .browser_policy import BrowserCachedPolicy
from .stage4 import _images_zip
from .stage5 import adapter_fingerprint, unpack_adapter
from .stage10 import _score_record, _write_progress, group_record, trajectory_seed
from . import stage8, stage9
from .vision import Stage2Config
from .workspace import RUNS_PATH, run_path, sha256_file, verify_artifacts, write_json, write_manifest

CPU_FILES = ("config.json", "environment.json", "cases.json", "reserved-eval.json", "episodes.json",
             "demonstrations.json", "metrics.json", "checks.json", "images.zip", "report.html",
             "policy-prompt.txt", "requirements-browser.txt", "source-adapter.json")
GPU_FILES = (*CPU_FILES, "groups.json", "difficulty.json", "cpu-source.json", "model-cache.json", "chat-template.txt")


@dataclass(frozen=True)
class BrowserDifficultyConfig(BrowserConfig):
    task_suite = HARD_TASKS
    action_budget = 12
    context_budget = 8192
    tasks: tuple = HARD_TASKS
    max_steps: int = 12
    max_sequence_tokens: int = 8192
    sft_run_id: str = ""
    group_size: int = 4
    sampling_logp_tolerance: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.episodes not in (1, 2): raise ValueError("Difficulty pilot allows one or two training seeds per task")
        if self.group_size != 4 or self.sampling_logp_tolerance != 0.1:
            raise ValueError("Use four trajectories and the unchanged 0.1 scoring tolerance")
        if self.sft_run_id: run_path(Path("."), self.sft_run_id)


def build_cases(config):
    return [{"case_id": f"{task}-hard-train-{i:03d}", "task": task, "split": "train", "seed": 3000+i}
            for task in config.tasks for i in range(config.episodes)]


def reserved_eval(config):
    # Declared before inspecting pilot rewards, but never opened or demonstrated here.
    return [{"case_id": f"{task}-hard-eval-{i:03d}", "task": task, "split": "eval", "seed": 30000+i}
            for task in config.tasks for i in range(8)]


def source_receipt(config, root=RUNS_PATH):
    if not config.sft_run_id: raise ValueError("Supply the completed browser SFT adapter with --sft-run-id")
    directory = run_path(root, config.sft_run_id)
    stage9.replay_saved_run(directory)
    manifest = verify_artifacts(directory, stage=9, artifact_files=stage9.ARTIFACT_FILES)
    if manifest.get("protocol") != "browser-json-v1": raise ValueError("Browser tool protocol changed")
    previous = json.loads((directory/"config.json").read_text())
    for field in fields(Stage2Config):
        if previous[field.name] != getattr(config, field.name):
            raise ValueError(f"Adapter/model interface differs: {field.name}")
    if (directory/"policy-prompt.txt").read_text() != browser_prompt(8)+"\n":
        raise ValueError("Unexpected source adapter prompt")
    env = json.loads((directory/"environment.json").read_text())
    if env["miniwob_revision"] != MINIWOB_REVISION or env["miniwob_archive_sha256"] != MINIWOB_ARCHIVE_SHA256:
        raise ValueError("MiniWoB assets changed")
    old_pairs = {(c["task"], c["seed"]) for c in json.loads((directory/"cases.json").read_text())}
    if any((c["task"], c["seed"]) in old_pairs for c in build_cases(config)+reserved_eval(config)):
        raise ValueError("Difficulty pools overlap SFT instances")
    return {"run_id":directory.name, "sft_updates":previous["updates"],
            "manifest_sha256":sha256_file(directory/"manifest.json"),
            "adapter_sha256":sha256_file(directory/"adapter.zip"),
            "requirements_sha256":sha256_file(directory/"requirements-browser.txt"),
            "intentional_transfer_changes":{"tasks":list(config.tasks),"max_steps":config.max_steps,
                                            "max_sequence_tokens":config.max_sequence_tokens,
                                            "prompt_change":"action budget only: eight -> twelve"}}


def difficulty_report(episodes, groups, task):
    members = [e for e in episodes if e["policy"].startswith("sample-")]
    # Pool membership remains train in saved episodes. Relabel only a temporary
    # aggregation view because the shared evaluator intentionally filters eval.
    aggregate = summarize([{**e,"policy":"sampled","case":{**e["case"],"split":"eval"}} for e in members])
    mixed = sum(not g["zero_variance"] for g in groups)
    all_success = sum(all(r == 1 for r in g["returns"]) for g in groups)
    all_failure = sum(all(r == 0 for r in g["returns"]) for g in groups)
    if mixed: decision = "Candidate for a bounded GRPO correctness update; mixed training returns observed."
    elif all_success == len(groups): decision = "Saturated under this pilot; expand training diversity before RL."
    else: decision = "No mixed group observed; inspect failures and add a brief training-only SFT bridge."
    return {"task":task,"optimizer_steps":0,"groups":len(groups),"mixed_return_groups":mixed,
            "all_success_groups":all_success,"all_failure_groups":all_failure,
            "ready_for_nonzero_grpo_signal":bool(mixed),"training_metrics":aggregate.get("sampled",{}),
            "within_group_reward_variances":[g["std"]**2 for g in groups],"decision":decision,
            "scope":"Training-only task selection pilot; reserved evaluation instances were not opened. "
                    "A small pilot with four samples per seed is not a reliable performance estimate."}


def replay_saved_run(directory):
    manifest = json.loads((directory/"manifest.json").read_text())
    mode = manifest.get("mode")
    if mode not in ("cpu","gpu"): raise ValueError("Unknown difficulty run mode")
    verify_artifacts(directory,stage=11,artifact_files=GPU_FILES if mode=="gpu" else CPU_FILES)
    config = BrowserDifficultyConfig(**json.loads((directory/"config.json").read_text()))
    cases = build_cases(config)
    if (json.loads((directory/"cases.json").read_text()) != cases
            or json.loads((directory/"reserved-eval.json").read_text()) != reserved_eval(config)
            or (directory/"policy-prompt.txt").read_text() != browser_prompt(config.max_steps)+"\n"):
        raise ValueError("Task pools or prompt changed")
    episodes = json.loads((directory/"episodes.json").read_text())
    lookup = {e["episode_id"]:e for e in episodes}
    if len(lookup) != len(episodes): raise ValueError("Duplicate episode IDs")
    with zipfile.ZipFile(directory/"images.zip") as archive:
        for e in episodes:
            validate_episode(e)
            if e["case"] not in cases or e["max_steps"] != config.max_steps:
                raise ValueError("Unexpected case/action budget; evaluation cannot enter this pilot")
            if any(o["image"] not in archive.namelist() for o in e["observations"]):
                raise ValueError("Missing screenshot")
    oracle = [e for e in episodes if e["policy"]=="oracle"]
    if (len(oracle) != len(cases) or not all(e["success"] for e in oracle)
            or {e["case"]["case_id"] for e in oracle} != {c["case_id"] for c in cases}):
        raise ValueError("Each training case needs a successful oracle")
    if stage8.demonstrations(oracle) != json.loads((directory/"demonstrations.json").read_text()):
        raise ValueError("Training demonstrations changed")
    if summarize(episodes) != json.loads((directory/"metrics.json").read_text()):
        raise ValueError("Evaluation metrics changed")
    if mode=="cpu":
        if len(episodes)!=len(oracle): raise ValueError("Unexpected CPU episodes")
    else:
        task = manifest.get("task")
        if task not in config.tasks: raise ValueError("Unknown pilot task shard")
        groups = json.loads((directory/"groups.json").read_text())
        selected = [c for c in cases if c["task"]==task]
        if "case_ids" in manifest:
            if not manifest["case_ids"] or len(set(manifest["case_ids"]))!=len(manifest["case_ids"]):
                raise ValueError("Invalid shard case IDs")
            selected = [c for c in selected if c["case_id"] in manifest["case_ids"]]
            if [c["case_id"] for c in selected]!=manifest["case_ids"]: raise ValueError("Unknown shard case IDs")
        if len(groups)!=len(selected): raise ValueError("Incomplete task shard")
        ids, fingerprints = [], set()
        for index,(group,case) in enumerate(zip(groups,selected)):
            members = [lookup[key] for key in group["member_ids"]]
            group_index = cases.index(case) % config.episodes
            if (group != group_record(config,"new_train",members,group["behavior_fingerprint"])
                    or any(e["case"]!=case for e in members)
                    or group["trajectory_seeds"] != [trajectory_seed(config,group_index,i) for i in range(4)]):
                raise ValueError("Probe advantages, states or RNG streams changed")
            ids.extend(group["member_ids"]); fingerprints.add(group["behavior_fingerprint"])
        if (len(ids)!=len(set(ids)) or len(fingerprints)!=1
                or set(lookup)!={e["episode_id"] for e in oracle}|set(ids)):
            raise ValueError("Unexpected probe members or policy changes")
        if difficulty_report(episodes,groups,task) != json.loads((directory/"difficulty.json").read_text()):
            raise ValueError("Difficulty summary changed")
    return len(episodes)


def _save(run_id,config,data,assets,environment,source,receipt,extras=None):
    directory = run_path(RUNS_PATH,run_id)
    if (directory/"manifest.json").exists(): raise FileExistsError("Refusing to overwrite a completed run")
    directory.mkdir(parents=True,exist_ok=True)
    objects = {"config.json":asdict(config),"environment.json":environment,"cases.json":data["cases"],
               "reserved-eval.json":reserved_eval(config),"episodes.json":data["episodes"],
               "demonstrations.json":stage8.demonstrations(data["episodes"]),"metrics.json":summarize(data["episodes"]),
               "checks.json":data["checks"],"source-adapter.json":receipt}
    for name,value in objects.items(): write_json(directory/name,value)
    (directory/"policy-prompt.txt").write_text(browser_prompt(config.max_steps)+"\n")
    (directory/"requirements-browser.txt").write_bytes(Path("/root/requirements-browser.txt").read_bytes())
    (directory/"images.zip").write_bytes(_images_zip(assets))
    html = stage8.report_html({**data,"metrics":{}},assets,title="Harder browser training tasks",
                             description="Training-only pilot. Reserved evaluation instances are unopened. No optimizer updates.")
    if extras: html += '<h2>Training difficulty</h2><pre>'+escape(extras["difficulty.json"].decode())+'</pre>'
    (directory/"report.html").write_text(html)
    for name,value in (extras or {}).items(): (directory/name).write_bytes(value)
    files = GPU_FILES if extras else CPU_FILES
    manifest = write_manifest(directory,{"mode":environment["mode"],"task":environment.get("task"),
        **({"case_ids":environment["case_ids"]} if "case_ids" in environment else {}),
        "protocol":"browser-json-v1","environment_version":"browser-v1","source_sha256":source},stage=11,artifact_files=files)
    replay_saved_run(directory)
    return {"manifest":manifest,"environment":environment,
            "difficulty":json.loads(extras["difficulty.json"]) if extras else None,
            "files":{name:base64.b64encode((directory/name).read_bytes()).decode() for name in (*files,"manifest.json")}}


def evaluate_cpu(run_id,raw_config,source):
    config = BrowserDifficultyConfig(**raw_config)
    receipt = source_receipt(config)
    if sha256_file(Path("/root/requirements-browser.txt")) != receipt["requirements_sha256"]:
        raise ValueError("Browser lock changed")
    started = time.monotonic()
    data = {"cases":build_cases(config),"episodes":[],"checks":{}}
    assets = {}
    for case in data["cases"]:
        episode = collect_episode(config,case,"oracle",assets)
        episode["episode_id"] = "oracle/"+case["case_id"]
        validate_episode(episode)
        if not episode["success"]: raise RuntimeError("Reactive oracle failed "+case["case_id"])
        stage8.reexecute_episode(episode)
        data["episodes"].append(episode)
        print(f"Hard-task oracle/replay passed {case['case_id']}: {episode['steps']} actions",flush=True)
    data["checks"] = {"all_training_cases_oracle_solvable":True,"oracle_live_reexecution":True,
                      "reserved_evaluation_unopened":True,"optimization_not_performed":True}
    return _save(run_id,config,data,assets,{**stage8.runtime(),"mode":"cpu","max_actions":config.max_steps,
                                          "seconds":time.monotonic()-started},source,receipt)


def validate_cpu_preflight(config,cpu_run_id,root=RUNS_PATH):
    directory = run_path(root,cpu_run_id)
    verify_artifacts(directory,stage=11,artifact_files=CPU_FILES)
    replay_saved_run(directory)
    receipt = source_receipt(config,root)
    if (json.loads((directory/"config.json").read_text()) != json.loads(json.dumps(asdict(config)))
            or json.loads((directory/"source-adapter.json").read_text()) != receipt
            or sha256_file(directory/"requirements-browser.txt") != receipt["requirements_sha256"]):
        raise ValueError("CPU preflight config, adapter or lock changed")
    return receipt


def evaluate_gpu(run_id,raw_config,source,cache_receipt,cpu_run_id,task,case_index):
    import torch
    from PIL import Image

    config = BrowserDifficultyConfig(**raw_config)
    if task not in config.tasks: raise ValueError("Unknown pilot task")
    if type(case_index) is not int or not 0 <= case_index < config.episodes: raise ValueError("Unknown pilot case index")
    receipt = validate_cpu_preflight(config,cpu_run_id)
    directory = run_path(RUNS_PATH,cpu_run_id)
    if (sha256_file(Path("/root/requirements-browser.txt"))!=receipt["requirements_sha256"]
            or cache_receipt["revision"]!=config.model_revision): raise ValueError("Model or lock changed")
    data = {name:json.loads((directory/(name+".json")).read_text()) for name in ("cases","episodes","checks")}
    assets = {}
    with zipfile.ZipFile(directory/"images.zip") as archive:
        for name in archive.namelist():
            with Image.open(io.BytesIO(archive.read(name))) as im: assets[name]=im.convert("RGB").copy()
    started, groups = time.monotonic(), []
    try:
        with tempfile.TemporaryDirectory() as temporary:
            unpack_adapter(run_path(RUNS_PATH,config.sft_run_id)/"adapter.zip",Path(temporary)/"adapter")
            actor = BrowserCachedPolicy(config,Path(temporary)/"adapter")
        fingerprint = adapter_fingerprint(actor.model)
        selected_cases = [c for c in data["cases"] if c["task"]==task][case_index:case_index+1]
        # Fail fast on 1- and 3-image training histories before a full group.
        teacher = next(e for e in data["episodes"] if e["case"]==selected_cases[0])
        smoke = []
        for turn in (teacher["transitions"][0],teacher["transitions"][min(2,teacher["steps"]-1)]):
            from .trajectories import image_paths
            paths = image_paths(turn["messages"])
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(config.seed);torch.cuda.manual_seed_all(config.seed)
                decision = actor.sample(turn["messages"],[assets[p] for p in paths])
            record = {"messages":turn["messages"],"generation":{k:v for k,v in decision.items() if k!="text"}}
            delta = _score_record(actor,record,assets,fingerprint)
            if delta["prefix_max_delta"]>config.sampling_logp_tolerance: raise RuntimeError("Cached sampling/replay smoke mismatch")
            smoke.append({"images":len(paths),**delta})
            print(f"Cached replay smoke: {smoke[-1]}",flush=True)
        for case in selected_cases:
            members = []
            for member in range(4):
                seed = trajectory_seed(config,case_index,member)
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                policy = f"sample-{len(groups):03d}-{member}"
                episode = collect_episode(config,case,policy,assets,actor=actor.sample)
                episode.update(episode_id=policy+"/"+case["case_id"],sampling_seed=seed)
                data["episodes"].append(episode)
                validate_episode(episode)
                for turn in episode["transitions"]: _score_record(actor,turn,assets,fingerprint)
                stage8.reexecute_episode(episode)
                members.append(episode)
                print(f"Hard probe {case['case_id']} member {member+1}/4: return={episode['return']}, actions={episode['steps']}, elapsed={time.monotonic()-started:.1f}s",flush=True)
                _write_progress(run_id,config,data,assets,groups)
            if adapter_fingerprint(actor.model)!=fingerprint: raise RuntimeError("Behavior adapter changed")
            groups.append(group_record(config,"new_train",members,fingerprint))
            _write_progress(run_id,config,data,assets,groups)
        report = difficulty_report(data["episodes"],groups,task)
        data["checks"].update(frozen_behavior_policy=True,exact_assistant_tokens_scored=True,
            sampling_scoring_consistent=True,all_sampled_actions_live_reexecuted=True,independent_trajectory_rng=True)
        environment = {**stage8.runtime(),"mode":"gpu","task":task,"max_actions":config.max_steps,
            "case_ids":[c["case_id"] for c in selected_cases],"scoring_smoke":smoke,
            "gpu":torch.cuda.get_device_name(),"seconds":time.monotonic()-started,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"optimizer_steps":0,
            "behavior_fingerprint":fingerprint,"policy_scoring":"cached-replay-v1",
            "sampling_decoding":actor.rollout_generation.to_dict()}
        extra_objects = {"groups.json":groups,"difficulty.json":report,"model-cache.json":cache_receipt,
                         "cpu-source.json":{"run_id":cpu_run_id,"manifest_sha256":sha256_file(directory/"manifest.json")}}
        extras = {name:json.dumps(value,indent=2,allow_nan=False).encode() for name,value in extra_objects.items()}
        extras["chat-template.txt"] = actor.processor.chat_template.encode()
        result = _save(run_id,config,data,assets,environment,source,receipt,extras)
        write_json(run_path(RUNS_PATH,run_id)/"progress.json",{"status":"complete","optimizer_steps":0,
                   "episodes":len(data["episodes"]),"completed_groups":len(groups)})
        return result
    except Exception as exc:
        _write_progress(run_id,config,data,assets,groups,error=f"{type(exc).__name__}: {exc}")
        raise

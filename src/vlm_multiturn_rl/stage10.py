"""Fresh browser evaluation and frozen-policy GRPO readiness, without updates."""

import base64
from dataclasses import asdict, dataclass, fields
from html import escape
import io
import json
import math
from pathlib import Path
import tempfile
import time
import zipfile

from .browser import (BROWSER_PROMPT, MINIWOB_REVISION, MINIWOB_ARCHIVE_SHA256, BrowserConfig, cases as sft_cases, collect_episode,
                      summarize, validate_episode)
from .grpo import relative_advantages
from .reporting import wilson_interval
from .stage4 import _images_zip
from .stage5 import adapter_fingerprint, unpack_adapter
from .browser_policy import BrowserPrefixPolicy
from . import stage8, stage9
from .trajectories import image_paths
from .workspace import RUNS_PATH, run_path, sha256_file, verify_artifacts, write_json, write_manifest


CPU_FILES = ("config.json", "environment.json", "cases.json", "pools.json", "episodes.json",
             "metrics.json", "checks.json", "images.zip", "report.html", "policy-prompt.txt",
             "requirements-browser.txt", "source-adapter.json")
GPU_FILES = (*CPU_FILES, "groups.json", "readiness.json", "cpu-source.json", "model-cache.json", "chat-template.txt")


@dataclass(frozen=True)
class BrowserReadinessConfig(BrowserConfig):
    sft_run_id: str = ""
    fresh_cases: int = 8  # Per task, entirely new evaluation seeds.
    probe_cases: int = 1  # Per task, for EACH of seen/new training pools.
    group_size: int = 4
    sampling_logp_tolerance: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        for name, maximum in (("fresh_cases", 16), ("probe_cases", min(2, self.episodes))):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1, {maximum}]")
        if self.group_size != 4 or self.sampling_logp_tolerance != 0.1:
            raise ValueError("Readiness fixes four trajectories and log-probability tolerance 0.1")
        if self.sft_run_id: run_path(Path("."), self.sft_run_id)


def build_cases(config):
    seen = [c for c in sft_cases(config) if c["split"] == "train" and c["seed"] < config.probe_cases]
    new = [{"case_id": f"{task}-rltrain-{i:03d}", "task": task, "split": "train", "seed": 1000+i}
           for task in config.tasks for i in range(config.probe_cases)]
    evaluation = [{"case_id": f"{task}-fresh-eval-{i:03d}", "task": task, "split": "eval", "seed": 20000+i}
                  for task in config.tasks for i in range(config.fresh_cases)]
    pools = {name: [c["case_id"] for c in group] for name, group in
             (("seen_train", seen), ("new_train", new), ("fresh_eval", evaluation))}
    return seen+new+evaluation, pools


def source_receipt(config, root=RUNS_PATH):
    if not config.sft_run_id: raise ValueError("Supply a completed browser SFT run with --sft-run-id")
    directory = run_path(root, config.sft_run_id)
    stage9.replay_saved_run(directory)
    manifest = verify_artifacts(directory, stage=9, artifact_files=stage9.ARTIFACT_FILES)
    if manifest.get("protocol") != "browser-json-v1": raise ValueError("Browser SFT protocol changed")
    previous = json.loads((directory/"config.json").read_text())
    for field in fields(BrowserConfig):
        if json.dumps(previous[field.name]) != json.dumps(getattr(config, field.name)):
            raise ValueError(f"Browser readiness/source {field.name} differs")
    if (directory/"policy-prompt.txt").read_text() != BROWSER_PROMPT+"\n":
        raise ValueError("Browser policy prompt changed")
    environment=json.loads((directory/"environment.json").read_text())
    if (environment["miniwob_revision"] != MINIWOB_REVISION
            or environment["miniwob_archive_sha256"] != MINIWOB_ARCHIVE_SHA256):
        raise ValueError("MiniWoB assets differ from the SFT source")
    case_list, _ = build_cases(config)
    old_cases = json.loads((directory/"cases.json").read_text())
    old_pairs = {(c["task"], c["seed"]) for c in old_cases}
    if any((c["task"], c["seed"]) in old_pairs for c in case_list if c["seed"] >= 1000):
        raise ValueError("Fresh training/evaluation instances overlap the previous experiment")
    return {"run_id": directory.name, "sft_updates": previous["updates"],
            "manifest_sha256": sha256_file(directory/"manifest.json"),
            "adapter_sha256": sha256_file(directory/"adapter.zip"),
            "requirements_sha256": sha256_file(directory/"requirements-browser.txt")}


def latest_local_sft(root):
    candidates = sorted(root.glob("stage9-*/manifest.json"), reverse=True)
    for path in candidates:
        try: stage9.replay_saved_run(path.parent)
        except (OSError, ValueError, KeyError): continue
        return path.parent.name  # Latest complete run, never selected by reward.
    raise ValueError("No completed local browser SFT run; supply --sft-run-id")


def trajectory_seed(config, group_index, member):
    return (config.seed+1009*group_index+9176*member) % 2**32


def group_record(config, pool, members, fingerprint):
    if len(members) != config.group_size: raise ValueError("A probe group needs four trajectories")
    initial = lambda e: (e["case"], e["max_steps"], e["observations"][0]["goal"],
                         e["observations"][0]["pixel_sha256"])
    if any(initial(e) != initial(members[0]) for e in members):
        raise ValueError("Sampled trajectories need identical initial browser states")
    if members[0]["case"]["split"] != "train" or pool not in ("seen_train", "new_train"):
        raise ValueError("Sampling groups must never contain evaluation instances")
    if len({e["sampling_seed"] for e in members}) != config.group_size:
        raise ValueError("Each trajectory needs an independent RNG stream")
    delta = 0.0
    worst = None
    batched_delta = None
    for episode in members:
        for turn in episode["transitions"]:
            g = turn["generation"]
            if (g["behavior_fingerprint"] != fingerprint or g["image_paths"] != image_paths(turn["messages"])
                    or len(g["image_paths"]) != turn["step"]):
                raise ValueError("Behavior weights or ordered observation history changed")
            ids, old, sampled = g["generated_ids"], g["old_log_probs"], g["sampled_log_probs"]
            if (not ids or not len(ids) == len(old) == len(sampled)
                    or g["action_loss_span"] != [len(g["prompt_ids"]), len(g["prompt_ids"])+len(ids)]
                    or any(not math.isfinite(x) for x in (*old, *sampled))):
                raise ValueError("Assistant token scores/spans are not aligned")
            for index,(a,b) in enumerate(zip(old,sampled)):
                if abs(a-b) > delta:
                    delta = abs(a-b)
                    worst = {"episode_id":episode["episode_id"],"turn":turn["step"],
                             "token_index":index,"token_id":ids[index],"sampled_logp":b,"scored_logp":a}
            if "batched_diagnostic_log_probs" in g:
                batched = g["batched_diagnostic_log_probs"]
                if len(batched) != len(ids) or any(not math.isfinite(x) for x in batched):
                    raise ValueError("Invalid batched diagnostic scores")
                batched_delta = max(batched_delta or 0.0, max(abs(a-b) for a,b in zip(batched,sampled)))
    if delta > config.sampling_logp_tolerance:
        raise ValueError(f"Sampling/scoring log-probability mismatch: {delta:.6f}; worst={worst}")
    stats = relative_advantages([e["return"] for e in members])
    result = {**stats, "pool": pool, "case_id": members[0]["case"]["case_id"],
            "member_ids": [e["episode_id"] for e in members],
            "trajectory_seeds": [e["sampling_seed"] for e in members],
            "behavior_fingerprint": fingerprint, "max_sampling_scoring_logp_delta": delta}
    if batched_delta is not None: result["max_batched_diagnostic_logp_delta"] = batched_delta
    return result


def readiness_report(episodes, groups):
    result = {"optimizer_steps": 0, "groups": len(groups), "pools": {}}
    lookup = {e["episode_id"]: e for e in episodes}
    for pool in ("seen_train", "new_train"):
        selected = [g for g in groups if g["pool"] == pool]
        members = [lookup[key] for g in selected for key in g["member_ids"]]
        # Reuse aggregation only; original training episodes remain unchanged.
        summary = summarize([{**e, "policy": "sampled", "case": {**e["case"], "split": "eval"}} for e in members])
        result["pools"][pool] = {"groups": len(selected),
            "mixed_return_groups": sum(not g["zero_variance"] for g in selected),
            "all_success_groups": sum(all(r == 1 for r in g["returns"]) for g in selected),
            "all_failure_groups": sum(all(r == 0 for r in g["returns"]) for g in selected),
            "within_group_reward_variances": [g["std"]**2 for g in selected],
            "sampled_metrics": summary.get("sampled", {})}
    metrics = summarize(episodes).get("sft", {})
    result["fresh_evaluation"] = {task: {**row, "success_wilson_95": wilson_interval(row["successes"],row["episodes"])}
                                  for task,row in metrics.items()}
    result["mixed_return_groups"] = sum(not g["zero_variance"] for g in groups)
    result["ready_for_nonzero_grpo_signal"] = bool(result["mixed_return_groups"])
    result["decision"] = ("Mixed-return training groups observed; proceed to a bounded GRPO correctness update."
                          if result["mixed_return_groups"] else
                          "No relative-reward signal observed. Inspect saturated/failing training pools before GRPO.")
    result["scope"] = ("Fresh instance seeds within three familiar templates, not unseen task types/sites. "
                       "Wilson intervals are illustrative binomial intervals; task-template dependence remains.")
    return result


def replay_saved_run(directory):
    raw_manifest = json.loads((directory/"manifest.json").read_text())
    mode = raw_manifest.get("mode")
    if mode not in ("cpu", "gpu"): raise ValueError("Unknown readiness report mode")
    verify_artifacts(directory, stage=10, artifact_files=GPU_FILES if mode == "gpu" else CPU_FILES)
    config = BrowserReadinessConfig(**json.loads((directory/"config.json").read_text()))
    expected_cases, pools = build_cases(config)
    if (json.loads((directory/"cases.json").read_text()) != expected_cases
            or json.loads((directory/"pools.json").read_text()) != pools):
        raise ValueError("Readiness pools or task seeds changed")
    episodes = json.loads((directory/"episodes.json").read_text())
    lookup = {e["episode_id"]: e for e in episodes}
    if len(lookup) != len(episodes): raise ValueError("Duplicate episode IDs")
    with zipfile.ZipFile(directory/"images.zip") as archive:
        names = set(archive.namelist())
        for episode in episodes:
            validate_episode(episode)
            if episode["case"] not in expected_cases: raise ValueError("Unregistered readiness instance")
            if any(o["image"] not in names for o in episode["observations"]): raise ValueError("Missing screenshot")
    oracle = [e for e in episodes if e["policy"] == "oracle"]
    if (len(oracle) != len(expected_cases) or not all(e["success"] for e in oracle)
            or {e["case"]["case_id"] for e in oracle} != {c["case_id"] for c in expected_cases}):
        raise ValueError("Every preflight case needs exactly one successful oracle episode")
    if summarize(episodes) != json.loads((directory/"metrics.json").read_text()):
        raise ValueError("Evaluation metrics differ from saved episodes")
    if mode == "gpu":
        greedy = [e for e in episodes if e["policy"] == "sft"]
        if len(greedy) != len(pools["fresh_eval"]) or {e["case"]["case_id"] for e in greedy} != set(pools["fresh_eval"]):
            raise ValueError("Incomplete fresh evaluation")
        groups = json.loads((directory/"groups.json").read_text())
        if len(groups) != len(pools["seen_train"])+len(pools["new_train"]): raise ValueError("Incomplete probe groups")
        fingerprints, member_ids, group_cases = set(), [], []
        for index, group in enumerate(groups):
            members = [lookup[key] for key in group["member_ids"]]
            if group != group_record(config,group["pool"],members,group["behavior_fingerprint"]):
                raise ValueError("Probe advantages differ from episode rewards")
            if group["case_id"] not in pools[group["pool"]]: raise ValueError("Case assigned to wrong probe pool")
            if group["trajectory_seeds"] != [trajectory_seed(config,index,i) for i in range(config.group_size)]:
                raise ValueError("Trajectory RNG streams changed")
            fingerprints.add(group["behavior_fingerprint"])
            group_cases.append(group["case_id"]); member_ids.extend(group["member_ids"])
        if (len(fingerprints) != 1 or len(member_ids) != len(set(member_ids))
                or set(group_cases) != set(pools["seen_train"]+pools["new_train"])):
            raise ValueError("Probe policies, groups or members changed")
        if set(lookup) != {e["episode_id"] for e in oracle+greedy} | set(member_ids):
            raise ValueError("Unexpected episodes outside the declared evaluation/probe groups")
        if readiness_report(episodes,groups) != json.loads((directory/"readiness.json").read_text()):
            raise ValueError("Readiness summary differs from probe groups")
    elif len(episodes) != len(oracle): raise ValueError("Unexpected CPU preflight episodes")
    return len(episodes)


def _save(run_id, config, data, assets, environment, source, receipt, extras=None):
    directory = run_path(RUNS_PATH,run_id)
    if (directory/"manifest.json").exists(): raise FileExistsError("Refusing to overwrite a completed run")
    directory.mkdir(parents=True,exist_ok=True)  # May already contain incomplete progress files.
    for name,value in {"config.json":asdict(config),"environment.json":environment,"cases.json":data["cases"],
                       "pools.json":data["pools"],"episodes.json":data["episodes"],"metrics.json":data["metrics"],
                       "checks.json":data["checks"],"source-adapter.json":receipt}.items(): write_json(directory/name,value)
    (directory/"policy-prompt.txt").write_text(BROWSER_PROMPT+"\n")
    (directory/"requirements-browser.txt").write_bytes(Path("/root/requirements-browser.txt").read_bytes())
    (directory/"images.zip").write_bytes(_images_zip(assets))
    description=("Fresh greedy evaluation plus frozen-policy training probes. No optimizer updates. Oracle has privileged DOM access."
                 if extras else "CPU preflight only: privileged oracle validates every selected case. Model evaluation and sampling are pending.")
    html = stage8.report_html(data,assets,title="Browser generalization and GRPO readiness",description=description)
    if extras:
        html += '<h2>GRPO readiness — training pools only</h2><pre>'+escape(extras["readiness.json"].decode())+'</pre>'
    (directory/"report.html").write_text(html)
    for name,value in (extras or {}).items(): (directory/name).write_bytes(value)
    files = GPU_FILES if extras else CPU_FILES
    manifest=write_manifest(directory,{"mode":environment["mode"],"protocol":"browser-json-v1",
                            "environment_version":"browser-v1","source_sha256":source},stage=10,artifact_files=files)
    replay_saved_run(directory)
    return {"manifest":manifest,"metrics":data["metrics"],"environment":environment,
            "readiness":json.loads(extras["readiness.json"]) if extras else None,
            "files":{name:base64.b64encode((directory/name).read_bytes()).decode() for name in (*files,"manifest.json")}}


def evaluate_cpu(run_id,raw_config,source):
    config=BrowserReadinessConfig(**raw_config)
    receipt=source_receipt(config)
    if sha256_file(Path("/root/requirements-browser.txt")) != receipt["requirements_sha256"]:
        raise ValueError("Browser dependency lock changed from the SFT source")
    started=time.monotonic()
    case_list,pools=build_cases(config)
    data={"cases":case_list,"pools":pools,"episodes":[],"checks":{}}
    assets={}
    for case in case_list:
        episode=collect_episode(config,case,"oracle",assets)
        episode["episode_id"]="oracle/"+case["case_id"]
        validate_episode(episode)
        if not episode["success"]: raise RuntimeError("Readiness oracle failed a selected instance")
        stage8.reexecute_episode(episode)
        data["episodes"].append(episode)
        print(f"Readiness oracle passed {case['case_id']}",flush=True)
    data["metrics"]=summarize(data["episodes"])
    data["checks"]={"all_cases_oracle_solvable":True,"oracle_live_reexecution":True,
                    "new_pools_disjoint_from_sft_instances":True,"train_eval_pools_disjoint":True,
                    "optimization_not_performed":True}
    return _save(run_id,config,data,assets,{**stage8.runtime(),"mode":"cpu","seconds":time.monotonic()-started},source,receipt)


def validate_cpu_preflight(config, cpu_run_id, root=RUNS_PATH):
    directory = run_path(root, cpu_run_id)
    verify_artifacts(directory, stage=10, artifact_files=CPU_FILES)
    replay_saved_run(directory)
    if json.loads((directory/"config.json").read_text()) != json.loads(json.dumps(asdict(config))):
        raise ValueError("Readiness CPU/GPU configurations differ")
    receipt = source_receipt(config, root)
    if json.loads((directory/"source-adapter.json").read_text()) != receipt:
        raise ValueError("SFT source changed since preflight")
    if (sha256_file(directory/"requirements-browser.txt") != receipt["requirements_sha256"]
            or (directory/"policy-prompt.txt").read_text() != BROWSER_PROMPT+"\n"):
        raise ValueError("Preflight dependencies or browser protocol changed")
    return receipt


def _write_progress(run_id, config, data, assets, groups, error=None):
    """Incomplete evidence deliberately has no completion manifest."""
    directory = run_path(RUNS_PATH, run_id)
    if (directory/"manifest.json").exists(): return
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"progress.json", {
        "status":"incomplete", "optimizer_steps":0, "episodes":len(data["episodes"]),
        "completed_groups":len(groups), "evaluation_metrics":summarize(data["episodes"]),
        "error":error, "config":asdict(config)})
    write_json(directory/"partial-episodes.json", data["episodes"])
    write_json(directory/"partial-groups.json", groups)
    (directory/"partial-images.zip").write_bytes(_images_zip(assets))


def _score_record(actor, turn, assets, fingerprint):
    import torch

    with torch.no_grad():
        scores = actor.score(turn, assets).cpu().tolist()
        batched = actor.score_batched_diagnostic(turn, assets).cpu().tolist()
    g = turn["generation"]
    g.update(old_log_probs=scores, batched_diagnostic_log_probs=batched,
             behavior_fingerprint=fingerprint,
             action_loss_span=[len(g["prompt_ids"]),len(g["prompt_ids"])+len(g["generated_ids"])])
    return {"prefix_max_delta":max(abs(a-b) for a,b in zip(scores,g["sampled_log_probs"])),
            "batched_max_delta":max(abs(a-b) for a,b in zip(batched,g["sampled_log_probs"]))}


def _scoring_smoke(actor, config, data, assets, fingerprint, run_id):
    import torch

    # Include the new enter-text case implicated by the original crash, with
    # one, two, and three images. These are training-only counterfactual actions.
    teacher = next(e for e in data["episodes"] if e["policy"]=="oracle"
                   and e["case"]["task"]=="enter-text" and e["case"]["seed"]==1000)
    rows = []
    for turn in teacher["transitions"]:
        paths = image_paths(turn["messages"])
        torch.manual_seed(config.seed+turn["step"])
        torch.cuda.manual_seed_all(config.seed+turn["step"])
        decision = actor.sample(turn["messages"], [assets[p] for p in paths])
        record = {"messages":turn["messages"], "generation":{k:v for k,v in decision.items() if k!="text"}}
        deltas = _score_record(actor, record, assets, fingerprint)
        rows.append({"turn":turn["step"],"images":len(paths),"text":decision["text"],
                     "sampled_tokens":len(decision["generated_ids"]),**deltas})
        directory = run_path(RUNS_PATH,run_id)
        directory.mkdir(parents=True,exist_ok=True)
        write_json(directory/"scoring-smoke.json",rows)
        print(f"Prefix scoring smoke: {rows[-1]}",flush=True)
        if deltas["prefix_max_delta"] > config.sampling_logp_tolerance:
            raise RuntimeError("Prefix sampling/scoring smoke failed before browser evaluation")
    return rows


def evaluate_gpu(run_id,raw_config,source,cache_receipt,cpu_run_id):
    import torch
    from PIL import Image

    config = BrowserReadinessConfig(**raw_config)
    receipt = validate_cpu_preflight(config,cpu_run_id)
    cpu_directory = run_path(RUNS_PATH,cpu_run_id)
    if (sha256_file(Path("/root/requirements-browser.txt")) != receipt["requirements_sha256"]
            or cache_receipt["revision"] != config.model_revision):
        raise ValueError("Browser lock or model revision changed")
    data = {name:json.loads((cpu_directory/(name+".json")).read_text())
            for name in ("cases","pools","episodes","checks")}
    assets = {}
    with zipfile.ZipFile(cpu_directory/"images.zip") as archive:
        for name in archive.namelist():
            with Image.open(io.BytesIO(archive.read(name))) as im: assets[name]=im.convert("RGB").copy()
    started = time.monotonic()
    groups = []
    try:
        with tempfile.TemporaryDirectory() as temporary:
            unpack_adapter(run_path(RUNS_PATH,config.sft_run_id)/"adapter.zip",Path(temporary)/"adapter")
            actor = BrowserPrefixPolicy(config,Path(temporary)/"adapter")
        fingerprint = adapter_fingerprint(actor.model)
        scoring_smoke = _scoring_smoke(actor,config,data,assets,fingerprint,run_id)
        for case in data["cases"]:
            if case["split"] != "eval": continue
            episode = collect_episode(config,case,"sft",assets,actor=actor.act)
            episode["episode_id"] = "sft/"+case["case_id"]
            data["episodes"].append(episode)
            validate_episode(episode)
            stage8.reexecute_episode(episode)
            print(f"Fresh eval {case['case_id']}: success={episode['success']}",flush=True)
        _write_progress(run_id,config,data,assets,groups)
        for pool in ("seen_train","new_train"):
            for case_id in data["pools"][pool]:
                case = next(c for c in data["cases"] if c["case_id"]==case_id)
                group_index = len(groups)
                members = []
                for member in range(config.group_size):
                    seed = trajectory_seed(config,group_index,member)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    policy = f"sample-{group_index:03d}-{member}"
                    episode = collect_episode(config,case,policy,assets,actor=actor.sample)
                    episode.update(episode_id=policy+"/"+case_id,sampling_seed=seed)
                    # Retain the last group's records even if a scoring check fails.
                    data["episodes"].append(episode)
                    validate_episode(episode)
                    for turn in episode["transitions"]:
                        _score_record(actor,turn,assets,fingerprint)
                    stage8.reexecute_episode(episode)
                    members.append(episode)
                    print(f"Probe {pool} {case_id} member {member+1}/4: return={episode['return']}",flush=True)
                if adapter_fingerprint(actor.model) != fingerprint:
                    raise RuntimeError("Policy changed during readiness collection")
                groups.append(group_record(config,pool,members,fingerprint))
                _write_progress(run_id,config,data,assets,groups)
        data["metrics"] = summarize(data["episodes"])
        readiness = readiness_report(data["episodes"],groups)
        data["checks"].update(frozen_behavior_policy=True,independent_trajectory_rng=True,
            exact_assistant_tokens_scored=True,sampling_scoring_consistent=True,
            all_sampled_actions_live_reexecuted=True,fresh_evaluation_excluded_from_groups=True,
            prefix_scoring_smoke_passed=True)
        environment = {**stage8.runtime(),"mode":"gpu","gpu":torch.cuda.get_device_name(),
            "seconds":time.monotonic()-started,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "evaluation_decoding":actor.generation.to_dict(),"sampling_decoding":actor.rollout_generation.to_dict(),
            "behavior_fingerprint":fingerprint,"optimizer_steps":0,"policy_scoring":"prefix-v1",
            "scoring_smoke":scoring_smoke}
        extras = {"groups.json":json.dumps(groups,indent=2,allow_nan=False).encode(),
            "readiness.json":json.dumps(readiness,indent=2,allow_nan=False).encode(),
            "cpu-source.json":json.dumps({"run_id":cpu_run_id,"manifest_sha256":sha256_file(cpu_directory/"manifest.json")}).encode(),
            "model-cache.json":json.dumps(cache_receipt).encode(),"chat-template.txt":actor.processor.chat_template.encode()}
        result = _save(run_id,config,data,assets,environment,source,receipt,extras)
        write_json(run_path(RUNS_PATH,run_id)/"progress.json",{"status":"complete","optimizer_steps":0,
                   "episodes":len(data["episodes"]),"completed_groups":len(groups)})
        return result
    except Exception as exc:
        _write_progress(run_id,config,data,assets,groups,error=f"{type(exc).__name__}: {exc}")
        raise

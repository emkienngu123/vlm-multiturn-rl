"""Two fresh browser GRPO rounds, differentiable cached replay, and held-out checks."""

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

from .browser import browser_prompt, collect_episode, summarize, validate_episode
from .browser_policy import BrowserTrainPolicy, saved_activation_memory
from .checkpoints import adapter_zip, capture_rng, clone_cpu, restore_rng, tree_digest
from .grpo import clipped_token_terms, tensor_checks
from .stage4 import _images_zip
from .stage5 import adapter_fingerprint, unpack_adapter
from .stage10 import group_record, trajectory_seed
from .stage11 import BrowserDifficultyConfig
from . import stage8, stage11
from .trajectories import image_paths
from .workspace import RUNS_PATH, run_path, sha256_file, verify_artifacts, write_json, write_manifest

BASE_FILES = ("config.json","environment.json","cases.json","episodes.json","groups.json","training.json",
              "metrics.json","checks.json","images.zip","report.html","source-adapter.json","cpu-source.json",
              "model-cache.json","policy-prompt.txt","requirements-browser.txt","chat-template.txt")
CHECKPOINT_FILES = ("adapter.zip","training-state.pt","checkpoint.json")


@dataclass(frozen=True)
class BrowserGRPOConfig(BrowserDifficultyConfig):
    updates: int = 2
    learning_rate: float = 1e-5
    clip_epsilon: float = .2
    ratio_tolerance: float = 1e-4
    kl_beta: float = 0.

    def __post_init__(self):
        super().__post_init__()
        if (self.episodes != 2 or type(self.updates) is not int or self.updates not in (1,2)
                or self.learning_rate!=1e-5 or self.clip_epsilon!=.2 or self.ratio_tolerance!=1e-4 or self.kl_beta!=0):
            raise ValueError("Browser correctness run fixes two train seeds, at most two rounds, lr=1e-5, clip=.2, beta=0")


def pilot_config(config):
    return BrowserDifficultyConfig(**{f.name:getattr(config,f.name) for f in fields(BrowserDifficultyConfig)})


def build_cases(config):
    parent = pilot_config(config)
    return ([c for c in stage11.build_cases(parent) if c["task"]=="multi-orderings"] +
            [c for c in stage11.reserved_eval(parent) if c["task"]=="multi-orderings"])


def artifact_files(config):
    return (*BASE_FILES,*(f"checkpoints/round-{r:03d}/{name}" for r in range(1,config.updates+1) for name in CHECKPOINT_FILES))


def activation_memory_checks():
    """Real CUDA saved-tensor offload must preserve values AND gradients."""
    import torch

    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(123)
        model=torch.nn.Linear(256,16,bias=False,device='cuda')
        x=torch.randn(512,256,device='cuda',requires_grad=True)
        loss=(model(x).sin().square()).mean()
        expected=torch.autograd.grad(loss,model.weight)[0]
        with saved_activation_memory(model,backward=True):
            actual_loss=(model(x).sin().square()).mean()
        actual=torch.autograd.grad(actual_loss,model.weight)[0]
        delta=float((expected-actual).abs().max())
        if delta>1e-6:raise RuntimeError('Activation offload changed gradients')
        with saved_activation_memory(model,backward=False):
            discarded=(model(x).sin().square()).mean()
        try:torch.autograd.grad(discarded,model.weight)
        except RuntimeError as exc:
            if 'Collection-only' not in str(exc):raise
        else:raise RuntimeError('Discarded collection graph allowed backward')
    return {'offload_max_gradient_delta':delta,'collection_graph_rejects_backward':True}


def score_check(actor, turn, assets, fingerprint, *, backward=False):
    import torch

    scores = actor.score_grad(turn,assets,backward=backward)
    sampled = torch.tensor(turn["generation"]["sampled_log_probs"],device=scores.device)
    if scores.shape!=sampled.shape or not torch.isfinite(scores).all(): raise RuntimeError("Bad gradient score shape/values")
    delta = float((scores.detach()-sampled).abs().max())
    ratio_delta = float(((scores.detach()-sampled).exp()-1).abs().max())
    if ratio_delta>actor.config.ratio_tolerance:
        raise RuntimeError(f"Gradient behavior ratio differs before update: {ratio_delta}; logp delta={delta}")
    g = turn["generation"]
    g.update(old_log_probs=scores.detach().cpu().tolist(),behavior_fingerprint=fingerprint,
             action_loss_span=[len(g["prompt_ids"]),len(g["prompt_ids"])+len(g["generated_ids"])])
    return scores, {"max_logp_delta":delta,"max_ratio_deviation":ratio_delta}


def gradient_smoke(actor, teacher, assets):
    import torch

    fingerprint = adapter_fingerprint(actor.model)
    rows = []
    for index in (0,2,len(teacher["transitions"])-1):
        turn = teacher["transitions"][index]
        paths = image_paths(turn["messages"])
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(actor.config.seed+index);torch.cuda.manual_seed_all(actor.config.seed+index)
            decision = actor.sample(turn["messages"],[assets[p] for p in paths])
        record = {"messages":turn["messages"],"generation":{k:v for k,v in decision.items() if k!="text"}}
        scores, check = score_check(actor,record,assets,fingerprint,backward=True)
        (-scores.mean()).backward()
        trainable = [p for p in actor.model.parameters() if p.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
        if float(norm)<=0 or any(p.grad is not None for p in actor.model.parameters() if not p.requires_grad):
            raise RuntimeError("Gradient smoke requires nonzero LoRA-only gradients")
        actor.model.zero_grad(set_to_none=True)
        if adapter_fingerprint(actor.model)!=fingerprint: raise RuntimeError("Calibration changed model weights")
        row = {"images":len(paths),"tokens":scores.numel(),"gradient_norm":float(norm),
               "cache_graph_preserved":True,"frozen_weights_have_no_gradients":True,**check}
        rows.append(row);print(f"Gradient scoring smoke: {row}",flush=True)
        del scores
    return rows


def update_group(actor, optimizer, episodes, group, assets, *, timing=None):
    import torch

    def stamp():
        if timing is not None: torch.cuda.synchronize()
        return time.monotonic()

    def elapsed(name, started):
        if timing is not None: timing[name] = timing.get(name, 0.) + stamp()-started

    config = actor.config
    fingerprint = group["behavior_fingerprint"]
    if adapter_fingerprint(actor.model)!=fingerprint: raise RuntimeError("Stale behavior policy")
    optimizer.zero_grad(set_to_none=True)
    if group["zero_variance"]:
        return {"optimizer_step":False,"reason":"zero_return_variance","gradient_norm":0.,"loss":0.,
                "max_ratio_deviation":0.,"clipped_fraction":0.,"after_fingerprint":fingerprint}
    total, tokens, clipped, max_ratio = 0.,0,0,0.
    for episode,advantage in zip(episodes,group["advantages"]):
        n = sum(len(t["generation"]["generated_ids"]) for t in episode["transitions"])
        for turn in episode["transitions"]:
            g = turn["generation"]
            if g["behavior_fingerprint"]!=fingerprint: raise RuntimeError("Mixed behavior weights")
            started = stamp() if timing is not None else 0.
            current = actor.score_grad(turn,assets)
            elapsed('scoring_forward_seconds', started)
            if timing is not None:
                for key in ('retained_bytes', 'offloaded_bytes'):
                    timing[key] = timing.get(key, 0) + actor.last_activation_stats[key]
            previous = torch.tensor(g["old_log_probs"],device=current.device)
            ratios = (current.detach()-previous).exp()
            deviation = float((ratios-1).abs().max())
            if deviation>config.ratio_tolerance: raise RuntimeError("Unchanged-weight gradient scoring drift")
            max_ratio = max(max_ratio,deviation)
            clipped += int(((ratios-1).abs()>config.clip_epsilon).sum());tokens+=current.numel()
            loss = clipped_token_terms(current,previous,advantage,config.clip_epsilon).sum()/(4*n)
            total += float(loss.detach())
            started = stamp() if timing is not None else 0.
            loss.backward()  # Release this action's graph before scoring the next.
            elapsed('backward_seconds', started)
            del current,previous,loss
    parameters = [p for p in actor.model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    if float(norm)<=0 or any(p.grad is not None for p in actor.model.parameters() if not p.requires_grad):
        raise RuntimeError("Expected finite nonzero LoRA-only gradients")
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    after = adapter_fingerprint(actor.model)
    if after==fingerprint: raise RuntimeError("Mixed group failed to change the adapter")
    return {"optimizer_step":True,"loss":total,"gradient_norm":float(norm),
            "scored_assistant_tokens":tokens,"max_ratio_deviation":max_ratio,
            "clipped_fraction":clipped/tokens,"after_fingerprint":after}


def save_checkpoint(directory,actor,optimizer,round_number,optimizer_steps):
    import torch

    directory.mkdir(parents=True,exist_ok=False)
    (directory/"adapter.zip").write_bytes(adapter_zip(actor.model))
    state = {"optimizer":clone_cpu(optimizer.state_dict()),"rng":capture_rng()}
    torch.save(state,directory/"training-state.pt")
    info = {"round":round_number,"optimizer_steps":optimizer_steps,"config":asdict(actor.config),
            "policy_scoring":"grad-cache-v1",
            "parameter_names":[n for n,p in actor.model.named_parameters() if p.requires_grad],
            "adapter_fingerprint":adapter_fingerprint(actor.model),"optimizer_digest":tree_digest(state["optimizer"]),
            "rng_digest":tree_digest(state["rng"]),
            "sha256":{name:sha256_file(directory/name) for name in ("adapter.zip","training-state.pt")}}
    write_json(directory/"checkpoint.json",info)
    return info


def reload_checkpoint(directory,actor):
    import torch
    from peft import PeftModel

    info = json.loads((directory/"checkpoint.json").read_text())
    if any(sha256_file(directory/name)!=digest for name,digest in info["sha256"].items()):
        raise RuntimeError("Checkpoint checksum mismatch")
    actor.model = actor.model.unload().eval()
    with tempfile.TemporaryDirectory() as temporary:
        unpack_adapter(directory/"adapter.zip",Path(temporary)/"adapter")
        actor.model = PeftModel.from_pretrained(actor.model,Path(temporary)/"adapter",is_trainable=True).eval()
    if (adapter_fingerprint(actor.model)!=info["adapter_fingerprint"] or
            [n for n,p in actor.model.named_parameters() if p.requires_grad]!=info["parameter_names"]):
        raise RuntimeError("Reload changed adapter parameters or ordering")
    optimizer = torch.optim.AdamW([p for p in actor.model.parameters() if p.requires_grad],
                                   lr=actor.config.learning_rate,weight_decay=0.)
    state = torch.load(directory/"training-state.pt",map_location="cpu",weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    if tree_digest(optimizer.state_dict())!=info["optimizer_digest"]: raise RuntimeError("Adam restore differs")
    steps = [int(s["step"]) for s in optimizer.state_dict()["state"].values()]
    if any(s!=info["optimizer_steps"] for s in steps) or (info["optimizer_steps"] and not steps):
        raise RuntimeError("Restored Adam step counters differ")
    restore_rng(state["rng"])
    if tree_digest(capture_rng())!=info["rng_digest"]: raise RuntimeError("RNG restore differs")
    return optimizer,{"adapter_matches":True,"adam_matches":True,"rng_matches":True,"optimizer_steps":info["optimizer_steps"]}


def paired_comparison(episodes):
    by_policy = {p:{e["case"]["case_id"]:e for e in episodes if e["policy"]==p} for p in ("sft","grpo")}
    if set(by_policy["sft"])!=set(by_policy["grpo"]): raise ValueError("Unpaired browser evaluation")
    rows=[]
    for key,a in by_policy["sft"].items():
        b=by_policy["grpo"][key]
        if (a["observations"][0]["goal"],a["observations"][0]["pixel_sha256"]) != (b["observations"][0]["goal"],b["observations"][0]["pixel_sha256"]):
            raise ValueError("Evaluation initial states differ")
        rows.append({"case_id":key,"sft_success":a["success"],"grpo_success":b["success"]})
    return rows


def replay_saved_run(directory):
    config = BrowserGRPOConfig(**json.loads((directory/"config.json").read_text()))
    verify_artifacts(directory,stage=12,artifact_files=artifact_files(config))
    cases = build_cases(config)
    if json.loads((directory/"cases.json").read_text())!=cases: raise ValueError("Task/evaluation cases changed")
    if (directory/"policy-prompt.txt").read_text()!=browser_prompt(12)+"\n": raise ValueError("Policy prompt changed")
    episodes = json.loads((directory/"episodes.json").read_text());lookup={e["episode_id"]:e for e in episodes}
    if len(lookup)!=len(episodes):raise ValueError("Duplicate episode IDs")
    with zipfile.ZipFile(directory/"images.zip") as archive:
        for e in episodes:
            validate_episode(e)
            if e["case"] not in cases or e["max_steps"]!=12: raise ValueError("Unknown training/evaluation case")
            if any(o["image"] not in archive.namelist() for o in e["observations"]):raise ValueError("Missing image")
    groups=json.loads((directory/"groups.json").read_text());training=json.loads((directory/"training.json").read_text())
    if len(groups)!=config.updates:raise ValueError("Incomplete GRPO rounds")
    previous=None;used=[]
    train=[c for c in cases if c["split"]=="train"]
    for index,g in enumerate(groups):
        members=[lookup[k] for k in g["member_ids"]]
        expected=group_record(config,"new_train",members,g["behavior_fingerprint"])
        if ({k:v for k,v in g.items() if k not in {"round","update"}}!=expected
                or g["round"]!=index+1 or any(e["case"]!=train[index] for e in members)
                or g["trajectory_seeds"]!=[trajectory_seed(config,index,m) for m in range(4)]):
            raise ValueError("Group returns, advantages, cases or random streams differ")
        if previous and g["behavior_fingerprint"]!=previous:raise ValueError("Round used stale adapter")
        u=g["update"]
        if u["optimizer_step"]==g["zero_variance"]:raise ValueError("Wrong zero-variance skip/update")
        if (u["max_ratio_deviation"]>config.ratio_tolerance or not math.isfinite(u["gradient_norm"])
                or (u["optimizer_step"] and (u["gradient_norm"]<=0 or u["after_fingerprint"]==g["behavior_fingerprint"]))):
            raise ValueError("Invalid update proof")
        previous=u["after_fingerprint"];used.extend(g["member_ids"])
        cp=directory/f"checkpoints/round-{index+1:03d}"
        info=json.loads((cp/"checkpoint.json").read_text())
        if (info["round"]!=index+1 or info["adapter_fingerprint"]!=previous
                or info["optimizer_steps"]!=sum(x["update"]["optimizer_step"] for x in groups[:index+1])
                or any(sha256_file(cp/n)!=h for n,h in info["sha256"].items())):
            raise ValueError("Checkpoint differs from update history")
    oracle=[e for e in episodes if e["policy"]=="oracle"]
    evaluations=[e for e in episodes if e["policy"] in ("sft","grpo")]
    if (len(oracle)!=len(cases) or not all(e["success"] for e in oracle)
            or {e["case"]["case_id"] for e in oracle}!={c["case_id"] for c in cases}):
        raise ValueError("Incomplete successful oracle set")
    eval_ids={c["case_id"] for c in cases if c["split"]=="eval"}
    for policy in ("sft","grpo"):
        actual=[e for e in evaluations if e["policy"]==policy]
        if len(actual)!=len(eval_ids) or {e["case"]["case_id"] for e in actual}!=eval_ids:raise ValueError("Incomplete evaluation")
    if (len(used)!=len(set(used)) or set(lookup)!={e["episode_id"] for e in oracle+evaluations}|set(used)
            or training["optimizer_steps"]!=sum(g["update"]["optimizer_step"] for g in groups)
            or training["paired_comparison"]!=paired_comparison(episodes)
            or summarize(episodes)!=json.loads((directory/"metrics.json").read_text())):
        raise ValueError("Episode membership or final metrics differ")
    return len(episodes)


def train_remote(run_id,raw_config,source,cache_receipt,cpu_run_id,commit_checkpoint):
    import torch
    from PIL import Image

    config=BrowserGRPOConfig(**raw_config);parent=pilot_config(config)
    receipt=stage11.validate_cpu_preflight(parent,cpu_run_id)
    if (sha256_file(Path('/root/requirements-browser.txt'))!=receipt['requirements_sha256']
            or cache_receipt['revision']!=config.model_revision):raise ValueError("Model/browser dependency interface changed")
    directory=run_path(RUNS_PATH,run_id);directory.mkdir(parents=True,exist_ok=False)
    cpu=run_path(RUNS_PATH,cpu_run_id)
    write_json(directory/'config.json',asdict(config))
    write_json(directory/'source-adapter.json',receipt)
    write_json(directory/'source-code.json',source)
    data={"cases":build_cases(config),"episodes":[],"checks":{}};assets={};groups=[];started=time.monotonic()
    def progress(error=None):
        write_json(directory/'progress.json',{"status":"incomplete","config":asdict(config),"error":error,
                   "completed_rounds":len(groups),"optimizer_steps":sum(g['update']['optimizer_step'] for g in groups)})
        write_json(directory/'partial-episodes.json',data['episodes']);write_json(directory/'partial-groups.json',groups)
        (directory/'partial-images.zip').write_bytes(_images_zip(assets))
    try:
        # Training oracles come from the checked preflight. Evaluation oracles are
        # generated only now; they are never used as demonstrations or gradients.
        old=json.loads((cpu/'episodes.json').read_text())
        teachers=[e for e in old if e['case']['task']=='multi-orderings']
        with zipfile.ZipFile(cpu/'images.zip') as archive:
            for e in teachers:
                for o in e['observations']:
                    with Image.open(io.BytesIO(archive.read(o['image']))) as im:assets[o['image']]=im.convert('RGB').copy()
        data['episodes'].extend(teachers)
        for case in data['cases']:
            if case['split']!='eval':continue
            e=collect_episode(config,case,'oracle',assets);e['episode_id']='oracle/'+case['case_id']
            validate_episode(e)
            if not e['success']:raise RuntimeError('Held-out oracle failed')
            stage8.reexecute_episode(e);data['episodes'].append(e)
        with tempfile.TemporaryDirectory() as temporary:
            unpack_adapter(run_path(RUNS_PATH,config.sft_run_id)/'adapter.zip',Path(temporary)/'adapter')
            actor=BrowserTrainPolicy(config,Path(temporary)/'adapter')
        checks={**tensor_checks(),**activation_memory_checks()};smoke=gradient_smoke(actor,teachers[0],assets)
        optimizer=torch.optim.AdamW([p for p in actor.model.parameters() if p.requires_grad],lr=config.learning_rate,weight_decay=0.)
        for case in data['cases']:
            if case['split']=='eval':
                e=collect_episode(config,case,'sft',assets,actor=actor.act);e['episode_id']='sft/'+case['case_id']
                validate_episode(e);stage8.reexecute_episode(e);data['episodes'].append(e)
                print(f"Before GRPO {case['case_id']}: success={e['success']}",flush=True)
        for index,case in enumerate([c for c in data['cases'] if c['split']=='train'][:config.updates]):
            fingerprint=adapter_fingerprint(actor.model);members=[]
            for member in range(4):
                seed=trajectory_seed(config,index,member);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                policy=f'round-{index+1:03d}-sample-{member}'
                e=collect_episode(config,case,policy,assets,actor=actor.sample)
                e.update(episode_id=policy+'/'+case['case_id'],sampling_seed=seed)
                validate_episode(e);data['episodes'].append(e)
                for turn in e['transitions']:
                    scores,_=score_check(actor,turn,assets,fingerprint);del scores
                stage8.reexecute_episode(e);members.append(e)
                progress()
                print(f"GRPO round {index+1} sample {member+1}/4: return={e['return']}, actions={e['steps']}",flush=True)
            group=group_record(config,'new_train',members,fingerprint)
            if any(p.grad is not None for p in actor.model.parameters()):raise RuntimeError('Collection accumulated gradients')
            group.update(round=index+1,update=update_group(actor,optimizer,members,group,assets));groups.append(group)
            save_checkpoint(directory/f'checkpoints/round-{index+1:03d}',actor,optimizer,index+1,
                            sum(g['update']['optimizer_step'] for g in groups))
            progress();commit_checkpoint()
            print(f"GRPO round {index+1}: {group['update']}",flush=True)
        # Prove real adapter + Adam + RNG reload before the after evaluation.
        probe=teachers[0]['transitions'][-1]
        paths=image_paths(probe['messages'])
        with torch.random.fork_rng(devices=[0]):
            decision=actor.sample(probe['messages'],[assets[p] for p in paths])
        record={'messages':probe['messages'],'generation':{k:v for k,v in decision.items() if k!='text'}}
        before=actor.score_grad(record,assets,backward=False).detach().cpu()
        del optimizer
        optimizer,reload_proof=reload_checkpoint(directory/f'checkpoints/round-{config.updates:03d}',actor)
        after=actor.score_grad(record,assets,backward=False).detach().cpu()
        reload_proof['max_logp_delta']=float((before-after).abs().max())
        if reload_proof['max_logp_delta']>1e-5:raise RuntimeError('Reload changed policy scores')
        for case in data['cases']:
            if case['split']=='eval':
                e=collect_episode(config,case,'grpo',assets,actor=actor.act);e['episode_id']='grpo/'+case['case_id']
                validate_episode(e);stage8.reexecute_episode(e);data['episodes'].append(e)
                print(f"After GRPO {case['case_id']}: success={e['success']}",flush=True)
        training={'optimizer_steps':sum(g['update']['optimizer_step'] for g in groups),'gradient_smoke':smoke,
                  'tensor_checks':checks,'reload':reload_proof,'paired_comparison':paired_comparison(data['episodes']),
                  'normalization':'mean of per-trajectory assistant-token means','kl_beta':config.kl_beta}
        data['checks']={'gradient_smoke_passed':True,'assistant_only_scores':True,'cache_graph_preserved':True,
                        'frozen_behavior_within_groups':True,'fresh_groups_after_each_round':True,
                        'evaluation_excluded_from_groups':True,'all_model_actions_live_reexecuted':True,
                        'adapter_adam_rng_reloaded':True}
        environment={**stage8.runtime(),'mode':'gpu','gpu':torch.cuda.get_device_name(),'max_actions':12,
                     'seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                     'policy_scoring':'grad-cache-v1','evaluation_decoding':actor.generation.to_dict(),
                     'activation_memory':'discard collection saved tensors; exact CPU offload for backward; GPU parameters retained',
                     'sampling_decoding':actor.rollout_generation.to_dict()}
        objects={'config.json':asdict(config),'environment.json':environment,'cases.json':data['cases'],
                 'episodes.json':data['episodes'],'groups.json':groups,'training.json':training,'metrics.json':summarize(data['episodes']),
                 'checks.json':data['checks'],'source-adapter.json':receipt,'model-cache.json':cache_receipt,
                 'cpu-source.json':{'run_id':cpu_run_id,'manifest_sha256':sha256_file(cpu/'manifest.json')}}
        for name,value in objects.items():write_json(directory/name,value)
        (directory/'images.zip').write_bytes(_images_zip(assets))
        (directory/'policy-prompt.txt').write_text(browser_prompt(12)+'\n')
        (directory/'requirements-browser.txt').write_bytes(Path('/root/requirements-browser.txt').read_bytes())
        (directory/'chat-template.txt').write_text(actor.processor.chat_template)
        html=stage8.report_html({**data,'metrics':summarize(data['episodes'])},assets,title='Two-round browser GRPO correctness',
                               description='Fresh training groups and paired held-out SFT/GRPO evaluation. Two rounds are a correctness experiment.')
        (directory/'report.html').write_text(html+'<h2>Training checks</h2><pre>'+escape(json.dumps(training,indent=2))+'</pre>')
        manifest=write_manifest(directory,{'mode':'gpu','protocol':'browser-json-v1','source_sha256':source},
                                stage=12,artifact_files=artifact_files(config))
        replay_saved_run(directory)
        write_json(directory/'progress.json',{'status':'complete','optimizer_steps':training['optimizer_steps']})
        return {'manifest':manifest,'training':training,'metrics':summarize(data['episodes']),'environment':environment,
                'files':{name:base64.b64encode((directory/name).read_bytes()).decode() for name in (*artifact_files(config),'manifest.json')}}
    except Exception as exc:
        progress(error=f'{type(exc).__name__}: {exc}');raise

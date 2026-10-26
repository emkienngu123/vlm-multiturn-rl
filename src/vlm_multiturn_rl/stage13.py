"""CPU-only browser grounding audit and training-only form demonstrations."""

import base64
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from html import escape
import io
import json
import math
from pathlib import Path
import re
import struct
import time
import zipfile

from .browser import (BrowserConfig, BrowserWorld, action_text, browser_prompt,
                      feedback_message, initial_history, public_observation, validate_episode)
from . import stage8, stage12
from .stage4 import _images_zip
from .workspace import RUNS_PATH, run_path, sha256_file, verify_artifacts, write_json, write_manifest

FILES = ("config.json", "pools.json", "episodes.json", "demonstrations.json", "audit.json",
         "summary.json", "checks.json", "environment.json", "source-audit.json", "images.zip",
         "image-index.json", "report.html", "policy-prompt.txt", "requirements-browser.txt")


@dataclass(frozen=True)
class FormDatasetConfig(BrowserConfig):
    task_suite = ("multi-orderings",)
    action_budget = 12
    context_budget = 8192
    tasks: tuple = ("multi-orderings",)
    max_steps: int = 12
    max_sequence_tokens: int = 8192
    clean_episodes: int = 100
    recovery_episodes: int = 20
    dev_episodes: int = 16
    final_episodes: int = 32

    def __post_init__(self):
        super().__post_init__()
        # Fix the study size and split declaration before opening new instances.
        counts=(self.clean_episodes, self.recovery_episodes, self.dev_episodes, self.final_episodes)
        if any(type(n) is not int for n in counts) or counts != (100,20,16,32):
            raise ValueError("This dataset fixes 100 clean, 20 recovery, 16 dev, and 32 final instances")


def pools(config):
    return {name:[{"case_id":f"forms13-{name}-{i:03d}", "task":"multi-orderings",
                   "split":"train" if name in ("clean", "recovery") else name, "seed":offset+i}
                  for i in range(count)] for name,offset,count in (
        ("clean",50000,config.clean_episodes), ("recovery",50100,config.recovery_episodes),
        ("dev",60000,config.dev_episodes), ("final",70000,config.final_episodes))}


def scaled_box(box, scale):
    return {k:box[k]*scale for k in ("x","y","width","height")}


def contains(box, x, y):
    return box["x"] <= x < box["x"]+box["width"] and box["y"] <= y < box["y"]+box["height"]


def snapshot(world):
    """Private diagnostics only. Never append this object to policy messages."""
    page = world.env.unwrapped.page
    state = page.evaluate("""() => {
      const describe = e => e ? {tag:e.tagName, id:e.id,
        label:e.closest('tr')?.querySelector('th')?.textContent?.trim() || ''} : null;
      const targets = [...document.querySelectorAll('#area input, #area .final')].map(e => {
        const r=e.getBoundingClientRect();
        return {...describe(e), box:{x:r.x,y:r.y,width:r.width,height:r.height},
          value: e.value || '', focused:e===document.activeElement};
      }).filter(e => e.box.width>0 && e.box.height>0);
      return {targets, viewport:{width:innerWidth,height:innerHeight},
        focused:describe(document.activeElement)};
    }""")
    state["scale"] = float(page._bgym_scale_factor)
    for target in state["targets"]:
        target["screenshot_box"] = scaled_box(target["box"], state["scale"])
    return state


def click_target(target):
    box=target["screenshot_box"]
    return action_text("click",x=round(box["x"]+box["width"]/2,2),y=round(box["y"]+box["height"]/2,2))


def teacher_action(goal, state, previous_text=None):
    match=re.fullmatch(r"Search for (.+) movies directed by (.+) from year (\d+)\.",goal)
    if match is None: raise ValueError("Unexpected form instruction")
    for label, value in zip(("Genre","Director","Year"),match.groups()):
        target=next(t for t in state["targets"] if t["label"]==label and t["tag"]=="INPUT")
        if target["value"]==value: continue
        if not target["focused"]: return click_target(target)
        if target["value"] and previous_text!=action_text("press",key="Control+a"):
            return action_text("press",key="Control+a")
        return action_text("type",text=value)
    return click_target(next(t for t in state["targets"] if t["tag"]!="INPUT"))


def collect_demo(config, case, assets, *, recovery=False):
    from PIL import Image

    world=BrowserWorld(case,config.max_steps)
    observations, transitions = [], []
    def observe():
        path=f"images/{case['case_id']}/{world.steps:03d}.png"
        image=Image.fromarray(world.obs["screenshot"]).convert("RGB")
        assets[path]=image
        public=public_observation(world.obs,path)
        observations.append({**public,"pixel_sha256":hashlib.sha256(image.tobytes()).hexdigest()})
        return public
    try:
        initial=snapshot(world)
        history=initial_history(observe(),config.max_steps)
        while not (world.terminated or world.truncated):
            state=snapshot(world)
            injected=recovery and world.steps<2
            if injected and world.steps==0:
                inputs=[t for t in state["targets"] if t["tag"]=="INPUT"]
                text=click_target(inputs[case["seed"]%len(inputs)])
            elif injected:
                text=action_text("type",text="incorrect-training-value")
            else:
                text=teacher_action(world.obs["goal"],state,transitions[-1]["text"] if transitions else None)
            record={"step":world.steps+1,"messages":deepcopy(history),"text":text,
                    "generation":{"generated_ids":None,"token_source":"synthetic_perturbation" if injected else "privileged_oracle"},
                    **world.step(text)}
            transitions.append(record)
            public=observe()
            history.extend([{"role":"assistant","content":text},feedback_message(record,public["image"])])
        episode={"case":case,"policy":"recovery_teacher" if recovery else "oracle","max_steps":config.max_steps,
                 "observations":observations,"transitions":transitions,"messages":history,"steps":world.steps,
                 "success":bool(world.terminated and transitions[-1]["raw_reward"]==1.),
                 "return":sum(t["reward"] for t in transitions),
                 "invalid_actions":sum(not t["valid_action"] for t in transitions),
                 "execution_errors":sum(t["execution_error"] for t in transitions),
                 "end_reason":"task_terminal" if world.terminated else "action_limit",
                 "field_order":[t["label"] for t in initial["targets"] if t["tag"]=="INPUT"]}
        validate_episode(episode)
        if not episode["success"]: raise RuntimeError("Teacher failed "+case["case_id"])
        return episode
    finally: world.close()


def demonstrations(episodes):
    # Perturbations remain historical context; only corrective/current teacher
    # actions become targets. No diagnostic DOM fields enter an example.
    return [{"case_id":e["case"]["case_id"],"turn":t["step"],"messages":t["messages"],
             "target":t["text"],"source":"privileged_oracle",
             "kind":"recovery" if e["policy"]=="recovery_teacher" else "clean"}
            for e in episodes if e["case"]["split"]=="train" and e["success"]
            and e["policy"] in ("oracle","recovery_teacher") for t in e["transitions"]
            if t["generation"]["token_source"]=="privileged_oracle"]


def audit_episode(episode, assets):
    """Replay saved model actions and measure real pointer events independently."""
    from PIL import Image

    world=BrowserWorld(episode["case"],episode["max_steps"])
    rows=[]
    try:
        image=Image.fromarray(world.obs["screenshot"]).convert("RGB")
        if (hashlib.sha256(image.tobytes()).hexdigest()!=episode["observations"][0]["pixel_sha256"]
                or world.obs["goal"]!=episode["observations"][0]["goal"]):
            raise RuntimeError("Audit reset differs from recorded episode")
        page=world.env.unwrapped.page
        page.evaluate("""() => {window.__vlmClicks=[];
          window.addEventListener('pointerdown', e => window.__vlmClicks.push({x:e.clientX,y:e.clientY,
            tag:e.target.tagName,id:e.target.id}),true);} """)
        for turn in episode["transitions"]:
            state=snapshot(world)
            path=f"audit/{episode['policy']}-{episode['case']['case_id']}/{turn['step']:03d}.png"
            assets[path]=Image.fromarray(world.obs["screenshot"]).convert("RGB")
            row={"step":turn["step"],"text":turn["text"],"image":path,"state":state,
                 "screenshot_size":list(assets[path].size),
                 "processor_image_grid_thw":turn['generation'].get('image_grid_thw'),
                 "coordinate_contract":"Original screenshot pixels; processor resizing does not rescale actions"}
            call=turn["action"]
            is_click=call is not None and call["name"]=="click"
            if is_click:
                x,y=(call["arguments"][k] for k in ("x","y"))
                row.update(point={"x":x,"y":y}, target_hits=[t["label"] or t["id"] or t["tag"]
                    for t in state["targets"] if contains(t["screenshot_box"],x,y)])
            page.evaluate("window.__vlmClicks=[]")
            result=world.step(turn["text"])
            if any(result[k]!=turn[k] for k in result): raise RuntimeError("Audit replay status differs")
            if is_click:
                events=page.evaluate("window.__vlmClicks")
                if len(events)!=1: raise RuntimeError("Expected exactly one real pointerdown")
                css={k:call["arguments"][k]/state["scale"] for k in ("x","y")}
                delta=max(abs(events[0][k]-css[k]) for k in css)
                if delta>1.: raise RuntimeError("Screenshot-to-CSS execution mismatch")
                row.update(expected_css=css,actual_pointer=events[0],pointer_max_error_css=delta)
            rows.append(row)
        return {"case":episode["case"],"policy":episode["policy"],"success":episode["success"],"turns":rows}
    finally: world.close()


def summary(episodes, audit):
    clicks=[t for e in audit for t in e["turns"] if "point" in t]
    return {"clean_successes":sum(e["success"] for e in episodes if e["policy"]=="oracle"),
            "recovery_successes":sum(e["success"] for e in episodes if e["policy"]=="recovery_teacher"),
            "training_targets":len(demonstrations(episodes)),
            "clean_field_orders":dict(Counter(" / ".join(e["field_order"]) for e in episodes if e["policy"]=="oracle")),
            "audited_episodes":len(audit),"audited_clicks":len(clicks),
            "clicks_inside_form_controls":sum(bool(t["target_hits"]) for t in clicks),
            "clicks_outside_form_controls":sum(not t["target_hits"] for t in clicks),
            "max_pointer_error_css":max((t["pointer_max_error_css"] for t in clicks),default=0.),
            "optimizer_steps":0,"dev_and_final_unopened":True,
            "scope":"Pointer checks establish execution mapping on recorded actions, not model perception accuracy. "
                    "Recovery data uses synthetic mistakes, not learner-state DAgger. New seeds share one task template."}


def validate_audit(audit):
    expected={(policy,30000+i) for policy in ('sft','grpo') for i in range(8)}
    if (len(audit)!=16 or {(e['policy'],e['case']['seed']) for e in audit}!=expected
            or any(e['case']['task']!='multi-orderings' or e['case']['split']!='eval' for e in audit)):
        raise ValueError("Incomplete diagnostic audit; never substitute training or fresh test cases")
    for e in audit:
        if not e['turns'] or len(e['turns'])>12: raise ValueError("Invalid audit length")
        for i,t in enumerate(e['turns']):
            state=t['state'];scale=state['scale']
            if not math.isfinite(scale) or scale<=0 or t['step']!=i+1: raise ValueError("Invalid audit geometry")
            if any(abs(size-state['viewport'][key]*scale)>1.5 for size,key in zip(t['screenshot_size'],('width','height'))):
                raise ValueError("Screenshot and viewport scaling differ")
            for target in state['targets']:
                if target['screenshot_box']!=scaled_box(target['box'],scale): raise ValueError("Target scaling changed")
            from .browser import parse_action
            call=parse_action(t['text'],*t['screenshot_size'])
            is_click=call is not None and call['name']=='click'
            if is_click!=('point' in t): raise ValueError("Missing click audit")
            if is_click:
                point=call['arguments']
                css={k:point[k]/scale for k in ('x','y')}
                delta=max(abs(t['actual_pointer'][k]-css[k]) for k in css)
                hits=[b['label'] or b['id'] or b['tag'] for b in state['targets']
                      if contains(b['screenshot_box'],point['x'],point['y'])]
                if (t['point']!=point or t['expected_css']!=css or not math.isfinite(delta) or delta>1.
                        or t['pointer_max_error_css']!=delta or t['target_hits']!=hits):
                    raise ValueError("Pointer mapping or hit classification changed")


def report_html(audit, assets, stats):
    parts=['<!doctype html><meta charset="utf-8"><title>Form grounding audit</title>',
           '<style>body{font:16px system-ui;margin:30px}section{display:flex;gap:16px;overflow:auto}'
           'figure{margin:8px;flex-shrink:0}pre{white-space:pre-wrap;max-width:500px}svg{position:absolute;left:0;top:0}</style>',
           '<h1>Form grounding audit and dataset</h1><pre>'+escape(json.dumps(stats,indent=2))+'</pre>',
           '<p>Red cross: predicted click. Green boxes: real form controls. All DOM labels are audit-only.</p>']
    for episode in audit:
        parts.append('<details><summary>'+escape(episode['policy']+' / '+episode['case']['case_id'])+'</summary><section>')
        for row in episode['turns']:
            if 'point' not in row: continue
            buffer=io.BytesIO();assets[row['image']].save(buffer,format='PNG')
            width,height=row['screenshot_size']
            shapes=[]
            for target in row['state']['targets']:
                b=target['screenshot_box']
                shapes.append(f'<rect x="{b["x"]}" y="{b["y"]}" width="{b["width"]}" height="{b["height"]}" fill="none" stroke="limegreen"/>')
            x,y=row['point']['x'],row['point']['y']
            shapes.append(f'<path d="M{x-5},{y}h10 M{x},{y-5}v10" stroke="red" stroke-width="2"/>')
            parts.append(f'<figure><div style="position:relative;width:{width}px;height:{height}px">'
                         f'<img width="{width}" height="{height}" src="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}">'
                         f'<svg width="{width}" height="{height}">'+''.join(shapes)+'</svg></div><pre>'+
                         escape(json.dumps({k:v for k,v in row.items() if k not in ('state','image','screenshot_size')},indent=2))+'</pre></figure>')
        parts.append('</section></details>')
    return ''.join(parts)


def replay_saved_run(directory):
    verify_artifacts(directory,stage=13,artifact_files=FILES)
    config=FormDatasetConfig(**json.loads((directory/'config.json').read_text()))
    registered=pools(config)
    if json.loads((directory/'pools.json').read_text())!=registered: raise ValueError("Dataset pools changed")
    if (directory/'policy-prompt.txt').read_text()!=browser_prompt(12)+'\n': raise ValueError("Prompt changed")
    episodes=json.loads((directory/'episodes.json').read_text())
    expected=registered['clean']+registered['recovery']
    if [e['case'] for e in episodes]!=expected: raise ValueError("Training cases incomplete or held-out leakage")
    for e in episodes:
        validate_episode(e)
        recovery=e['case'] in registered['recovery']
        sources=[t['generation']['token_source'] for t in e['transitions']]
        expected_sources=['synthetic_perturbation']*2+['privileged_oracle']*(e['steps']-2) if recovery else ['privileged_oracle']*e['steps']
        if (sources!=expected_sources or not e['success'] or e['execution_errors'] or e['invalid_actions']
                or e['max_steps']!=12 or sorted(e['field_order'])!=['Director','Genre','Year']
                or e['policy']!=('recovery_teacher' if recovery else 'oracle')):
            raise ValueError("Invalid teacher or perturbation provenance")
    if demonstrations(episodes)!=json.loads((directory/'demonstrations.json').read_text()):
        raise ValueError("Training targets or assistant-only history changed")
    audit=json.loads((directory/'audit.json').read_text())
    validate_audit(audit)
    if summary(episodes,audit)!=json.loads((directory/'summary.json').read_text()): raise ValueError("Summary changed")
    index=json.loads((directory/'image-index.json').read_text())
    required={o['image'] for e in episodes for o in e['observations']} | {t['image'] for e in audit for t in e['turns']}
    with zipfile.ZipFile(directory/'images.zip') as archive:
        if set(archive.namelist())!=required or set(index)!=required or len(archive.namelist())!=len(required):
            raise ValueError("Missing or duplicate screenshot")
        for name in required:
            data=archive.read(name)
            if hashlib.sha256(data).hexdigest()!=index[name]['sha256']: raise ValueError("Screenshot hash changed")
            if data[:8]!=b'\x89PNG\r\n\x1a\n' or list(struct.unpack('>II',data[16:24]))!=index[name]['size']:
                raise ValueError("Screenshot dimensions changed")
    for e in episodes:
        for o in e['observations']:
            if index[o['image']]['size']!=[o['width'],o['height']]: raise ValueError("Observation size changed")
    return len(episodes)


def create_remote(run_id, raw_config, source, audit_run_id, persist):
    config=FormDatasetConfig(**raw_config)
    source_dir=run_path(RUNS_PATH,audit_run_id)
    stage12.replay_saved_run(source_dir)
    if (source_dir/'requirements-browser.txt').read_bytes()!=Path('/root/requirements-browser.txt').read_bytes():
        raise ValueError("Browser lock differs from audit source")
    source_episodes=json.loads((source_dir/'episodes.json').read_text())
    selected=[e for e in source_episodes if e['policy'] in ('sft','grpo') and e['case']['split']=='eval']
    if len(selected)!=16 or any(e['case']['task']!='multi-orderings' for e in selected):
        raise ValueError("Audit requires the completed paired eight-case Stage 12 form evaluation")
    registered=pools(config)
    old_pairs={(e['case']['task'],e['case']['seed']) for e in source_episodes}
    if any((c['task'],c['seed']) in old_pairs for cs in registered.values() for c in cs):
        raise ValueError("New pools overlap diagnostic instances")
    destination=run_path(RUNS_PATH,run_id);destination.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    # Reserve pools durably BEFORE any new training instance is opened.
    write_json(destination/'config.json',asdict(config));write_json(destination/'pools.json',registered)
    persist()
    assets, audit, episodes = {}, [], []
    for episode in selected:
        audit.append(audit_episode(episode,assets))
        print(f"Audit replay: {episode['policy']} / {episode['case']['case_id']}",flush=True)
    validate_audit(audit)
    for kind in ('clean','recovery'):
        for case in registered[kind]:
            episode=collect_demo(config,case,assets,recovery=kind=='recovery')
            stage8.reexecute_episode(episode)
            episodes.append(episode)
            write_json(destination/'progress.json',{'status':'collecting','complete_episodes':len(episodes),
                       'expected_episodes':120,'optimizer_steps':0})
            print(f"Form demo/replay {len(episodes)}/120: {case['case_id']}, {episode['steps']} actions",flush=True)
    stats=summary(episodes,audit)
    objects={'episodes.json':episodes,'demonstrations.json':demonstrations(episodes),'audit.json':audit,
             'summary.json':stats,'source-audit.json':{'run_id':audit_run_id,
                 'manifest_sha256':sha256_file(source_dir/'manifest.json'),'purpose':'diagnosis only; never training targets'},
             'checks.json':{'all_training_cases_solved_and_live_reexecuted':True,'heldout_instances_unopened':True,
                 'perturbations_excluded_from_targets':True,'dom_excluded_from_training_messages':True,
                 'real_pointer_events_checked':True,'pointer_tolerance_css_pixels':1.,'optimizer_steps':0},
             'environment.json':{**stage8.runtime(),'max_actions':12,'mode':'cpu',
                                  'seconds':time.monotonic()-started,'training_performed':False}}
    for name,value in objects.items(): write_json(destination/name,value)
    (destination/'images.zip').write_bytes(_images_zip(assets))
    with zipfile.ZipFile(destination/'images.zip') as archive:
        write_json(destination/'image-index.json',{name:{'sha256':hashlib.sha256(archive.read(name)).hexdigest(),
                   'size':list(assets[name].size)} for name in archive.namelist()})
    (destination/'report.html').write_text(report_html(audit,assets,stats))
    (destination/'requirements-browser.txt').write_bytes(Path('/root/requirements-browser.txt').read_bytes())
    (destination/'policy-prompt.txt').write_text(browser_prompt(12)+'\n')
    manifest=write_manifest(destination,{'mode':'cpu','source_sha256':source,'protocol':'browser-json-v1',
                                        'dataset_version':'forms13-v1'},stage=13,artifact_files=FILES)
    replay_saved_run(destination)
    write_json(destination/'progress.json',{'status':'complete','complete_episodes':120,'optimizer_steps':0})
    return {'manifest':manifest,'summary':stats,'files':{name:base64.b64encode((destination/name).read_bytes()).decode()
                                                       for name in (*FILES,'manifest.json')}}

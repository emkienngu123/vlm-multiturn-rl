"""Build the illustrated report from the repository and saved experiment artifacts.

Run with the project's light local environment. No training, Torch or browser
imports. Screenshots and code are embedded into one offline HTML deliverable.
"""
import ast
import base64
from html import escape
import hashlib
import json
from pathlib import Path
import re
import textwrap
import zipfile

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer, BashLexer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reports'
ART = ROOT/'artifacts'
WORKFLOW = 'stage18-20261019T171720Z-b1512977-workflow'
MANIFEST = {}


def read_json(path):
    return json.loads(path.read_text())


def remember(path):
    MANIFEST[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()


def data_image(path):
    remember(path)
    kind = 'svg+xml' if path.suffix == '.svg' else 'png'
    return 'data:image/'+kind+';base64,'+base64.b64encode(path.read_bytes()).decode()


def screenshot(directory, episode, index):
    archive = directory/'images.zip'; remember(archive)
    with zipfile.ZipFile(archive) as z:
        raw = z.read(episode['observations'][index]['image'])
    return 'data:image/png;base64,'+base64.b64encode(raw).decode()


def source_function(path, *names):
    path = ROOT/path; remember(path)
    text = path.read_text(); body = ast.parse(text).body
    for name in names:
        node = next(n for n in body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        body = node.body
    return textwrap.dedent(ast.get_source_segment(text,node))


def block(code, title, source, exact=True, language='python'):
    formatter=HtmlFormatter(nowrap=True)
    colored=highlight(code, BashLexer() if language=='bash' else PythonLexer(),formatter)
    kind='Repository excerpt' if exact else 'Condensed teaching example'
    return f'<div class="listing"><div class="listing-head">{escape(title)} <span>{kind}</span></div><pre><code>{colored}</code></pre><div class="source">Source: {escape(source)}</div></div>'


def main():
    OUT.mkdir(exist_ok=True)
    remember(Path(__file__).resolve())
    remember(OUT/'tools/render_report.mjs')
    workflow = read_json(ART/(WORKFLOW+'.json')); summary=read_json(ART/(WORKFLOW+'-summary.json'))
    assert workflow['status']=='complete' and summary['total_optimizer_steps']==7
    assert summary['phases']['after']['successes']==4 and summary['paired_changes']['success_to_failure']==0
    sft=ART/'stage14-20261019T102008Z-1255a893/training.json'
    remember(sft)
    assert read_json(sft)['completed_updates']==214 and read_json(sft)['examples_processed']==853
    for name in ('stage15-20261019T143704Z-7b6ed5ac-workflow-summary.json',): remember(ART/name)
    assets={}; replay={}; episodes={}
    for label in ('before','after'):
        for run in workflow['eval_phases'][label]:
            d=ART/run; remember(d/'manifest.json'); remember(d/'episodes.json')
            for e in read_json(d/'episodes.json'):
                if e['case']['case_id'] != 'forms13-dev-010': continue
                episodes[label]=e
                assets['case-'+label]=screenshot(d,e,6)
                assets['case-initial']=screenshot(d,e,0)
                replay[label]={'goal':e['observations'][0]['goal'],'success':e['success'],
                    'frames':[{'image':screenshot(d,e,i),'action':t['text'],
                        'reward':t['reward'],'valid':t['valid_action'],'execution_error':t['execution_error']}
                        for i,t in enumerate(e['transitions'])]}
    assert episodes['before']['observations'][0]['pixel_sha256']==episodes['after']['observations'][0]['pixel_sha256']
    # Actual screenshots of tasks used in the harder-task pilot.
    hard=ART/'stage11-20261019T072105Z-f87924c8'; remember(hard/'episodes.json')
    hard_episodes=read_json(hard/'episodes.json')
    for task,key in (('click-tab-2-hard','tabs'),('login-user-popup','popup')):
        e=next(e for e in hard_episodes if e['case']['task']==task and e['policy']=='oracle')
        assets[key]=screenshot(hard,e,0)
    assets['grid']=data_image(ART/'stage3-20261017T173649Z-67add07c/preview.png')
    assets['learning']=data_image(ART/(WORKFLOW+'-training.png'))
    assets['sft-learning']=data_image(ART/'stage14-20261019T100154Z-4b297337-workflow-training.png')
    snippets={
        'observation':block(source_function('src/vlm_multiturn_rl/browser.py','public_observation'),
            'Define the policy observation','browser.py::public_observation'),
        'processor':block(source_function('src/vlm_multiturn_rl/stage4.py','HistoryVLM','encode_history'),
            'Preserve image order and visual-token accounting','stage4.py::HistoryVLM.encode_history'),
        'labels':block(source_function('src/vlm_multiturn_rl/trajectories.py','assistant_labels'),
            'Assistant-only SFT labels','trajectories.py::assistant_labels'),
        'advantages':block(source_function('src/vlm_multiturn_rl/grpo.py','relative_advantages'),
            'Population-normalized episode advantages','grpo.py::relative_advantages'),
        'causal':block(source_function('src/vlm_multiturn_rl/grpo.py','selected_log_probs'),
            'The causal shift, explicitly','grpo.py::selected_log_probs'),
        'gradient-decode':block(source_function('src/vlm_multiturn_rl/browser_policy.py','gradient_decode'),
            'Enable gradients through the cached decoding path','browser_policy.py::gradient_decode'),
        'score-check':block(source_function('src/vlm_multiturn_rl/stage12.py','score_check'),
            'Reject drift before an update','stage12.py::score_check'),
        'clipping':block(source_function('src/vlm_multiturn_rl/grpo.py','clipped_token_terms'),
            'Clipped assistant-token surrogate','grpo.py::clipped_token_terms'),
    }
    examples={
    'modal':('Remote resources and persistent storage', 'modal_app.py; equivalent simplified H100 declaration', '''
cache = modal.Volume.from_name("vlm-multiturn-rl-cache", create_if_missing=True)
runs = modal.Volume.from_name("vlm-multiturn-rl-runs", create_if_missing=True)

@app.function(image=browser_image, gpu="H100!", timeout=1800,
              cpu=2, memory=65536,
              volumes={"/cache": cache, "/runs": runs})
def train_one_group(config, previous_checkpoint):
    # Import Torch here, not in local orchestration.
    import torch
    runs.reload()
    result = train_and_checkpoint(config, previous_checkpoint)
    runs.commit()
    return json.loads(json.dumps(result))
'''),
    'rollout':('A trajectory is a feedback loop', 'browser.py::collect_episode; protocol details omitted', '''
history = initial_history(first_observation, max_steps=12)
while not (world.terminated or world.truncated):
    decision = actor.sample(history, ordered_images(history))
    transition = world.step(decision["text"])
    save_turn(history, decision["generated_ids"], transition)
    next_observation = observe(world)
    history.extend([
        {"role": "assistant", "content": decision["text"]},
        feedback_message(transition, next_observation["image"]),
    ])
'''),
    'lora':('Restrict adaptation to language attention', 'stage4.py::train_adapter; shortened', '''
targets = [name for name, module in model.named_modules()
           if ".language_model." in name
           and name.endswith((".q_proj", ".v_proj"))
           and isinstance(module, torch.nn.Linear)]
model.requires_grad_(False)
model = get_peft_model(model, LoraConfig(
    task_type="CAUSAL_LM", r=8, lora_alpha=16,
    lora_dropout=0.0, target_modules=targets, bias="none"))
'''),
    'group':('Freeze, collect, score, then update', 'stage16.py::train_remote; simplified control flow', '''
behavior = adapter_fingerprint(actor.model)
members = []
for member in range(4):
    seed = trajectory_seed(config, group_index, member)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    episode = collect_episode(config, same_case, policy, assets,
                              actor=actor.sample)
    for turn in episode["transitions"]:
        score_check(actor, turn, assets, behavior, backward=False)
    members.append(episode)
group = strict_group(config, members, behavior)
update = update_group(actor, optimizer, members, group, assets)
# Only now may the next group use the updated policy.
'''),
    'update':('Normalize by trajectory, accumulate by action', 'stage12.py::update_group; checks and logging omitted', '''
optimizer.zero_grad(set_to_none=True)
if group["zero_variance"]:
    return {"optimizer_step": False}
for episode, advantage in zip(episodes, group["advantages"]):
    n = sum(len(t["generation"]["generated_ids"])
            for t in episode["transitions"])
    for turn in episode["transitions"]:
        current = actor.score_grad(turn, assets)
        previous = turn["generation"]["old_log_probs"]
        loss = clipped_token_terms(current, previous, advantage).sum() / (4*n)
        loss.backward()  # Release each action graph before the next.
torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
optimizer.step()
'''),
    'memory':('Saved values can move; the graph must remain connected', 'browser_policy.py::saved_activation_memory; shortened', '''
def pack(tensor):
    if not backward:
        return None  # Collection never uses backward.
    if is_large_cuda_activation(tensor) and not_parameter_storage(tensor):
        if budget_allows_retention(tensor):
            return (tensor.device, tensor.detach())
        return (tensor.device, tensor.detach().to("cpu", copy=True))
    return (tensor.device, tensor.detach())

def unpack(saved):
    device, tensor = saved
    return tensor.to(device)

with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
    scores = replay_exact_tokens_with_gradients(...)
'''),
    'checkpoint':('Save the learning process, not only its weights', 'stage12.py::save_checkpoint / reload_checkpoint; shortened', '''
save_adapter(actor.model, checkpoint / "adapter.zip")
torch.save({"optimizer": clone_cpu(optimizer.state_dict()),
            "rng": capture_rng()}, checkpoint / "training-state.pt")
write_metadata(parameter_names, optimizer_steps, adapter_fingerprint,
               optimizer_digest, rng_digest, file_sha256)

# In the next container: load adapter -> create Adam -> load state.
optimizer.load_state_dict(saved_state["optimizer"])
restore_rng(saved_state["rng"])
# Verify names/order, moments, step counters, RNG, and replayed scores.
'''),
    }
    for name,(title,source,code) in examples.items():
        snippets[name]=block(textwrap.dedent(code).strip(),title,source,exact=False)
    template=OUT/'source/tutorial.html'; remember(template)
    text=template.read_text()
    text=text.replace('{{PYGMENTS_CSS}}',HtmlFormatter().get_style_defs('.listing'))
    text=re.sub(r'\{\{CODE:([a-z-]+)\}\}',lambda m:snippets[m[1]],text)
    text=re.sub(r'\{\{IMG:([a-z-]+)\}\}',lambda m:assets[m[1]],text)
    text=text.replace('{{REPLAY_DATA}}',json.dumps(replay,separators=(',',':')).replace('</','<\\/'))
    text=re.sub(r'(<table\b.*?</table>)',r'<div class="table-wrap">\1</div>',text,flags=re.S)
    assert '{{' not in text, 'Unresolved template placeholder'
    output=OUT/'vlm-multiturn-rl-tutorial.html'; output.write_text(text)
    for path in (ART/(WORKFLOW+'.json'),ART/(WORKFLOW+'-summary.json'),
                 ART/'stage17-20261019T164758Z-5471fe18-workflow-summary.json',
                 ART/'stage16-20261019T152005Z-b83a0b3b-workflow-summary.json'):
        remember(path)
    manifest={'kind':'technical-tutorial-report','as_of_local_date':'2026-10-20',
        'timezone':'Asia/Ho_Chi_Minh','source_sha256':MANIFEST,
        'html_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'learning_checkpoint':workflow['last_checkpoint'],'final_test_unopened':True,
        'note':'Code excerpts reflect the current checkout; historical runs retain their own source hashes.'}
    (OUT/'report-provenance.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(output)


if __name__=='__main__': main()

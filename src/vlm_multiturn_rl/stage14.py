"""One-pass form SFT with true chunk continuation and paired development evaluation."""

import base64
from dataclasses import asdict, dataclass, fields
import io
import json
import math
from pathlib import Path
import random
import tempfile
import time
import zipfile

from . import stage8, stage9, stage11, stage13
from .browser import browser_prompt, collect_episode, validate_episode, summarize
from .checkpoints import adapter_zip, capture_rng, clone_cpu, restore_rng, tree_digest
from .stage4 import _images_zip, _probe
from .stage5 import adapter_fingerprint, unpack_adapter
from .stage9 import BrowserSFTActor, validate_mask
from .trajectories import image_paths
from .vision import Stage2Config
from .workspace import RUNS_PATH, run_path, sha256_file, verify_artifacts, write_json, write_manifest

TRAIN_FILES=('config.json','receipts.json','training.json','mask-inspection.json','adapter.zip',
             'training-state.pt','checkpoint.json','environment.json')
EVAL_FILES=('config.json','receipts.json','evaluation.json','episodes.json','images.zip','metrics.json',
            'environment.json','report.html','policy-prompt.txt')


@dataclass(frozen=True)
class FormSFTConfig(stage13.FormDatasetConfig):
    dataset_run_id: str = ''
    sft_run_id: str = ''
    epochs: int = 1
    gradient_accumulation: int = 4
    learning_rate: float = 5e-5
    lora_rank: int = 8

    def __post_init__(self):
        super().__post_init__()
        for value in (self.dataset_run_id,self.sft_run_id):
            if value: run_path(Path('.'),value)
        if (type(self.epochs) is not int or self.epochs not in (1,2)
                or type(self.gradient_accumulation) is not int or self.gradient_accumulation!=4
                or self.lora_rank!=8 or self.learning_rate!=5e-5):
            raise ValueError('Use one or two full passes, accumulation 4, rank 8, and lr=5e-5')


def schedule(config, count):
    if type(count) is not int or count<=0: raise ValueError('Empty training dataset')
    rows=[]
    for epoch in range(config.epochs):
        order=list(range(count));random.Random(config.seed+epoch).shuffle(order)
        for start in range(0,count,config.gradient_accumulation):
            rows.append({'update':len(rows)+1,'epoch':epoch+1,
                         'example_indices':order[start:start+config.gradient_accumulation]})
    return rows


def source_receipts(config, root=RUNS_PATH):
    if not config.dataset_run_id or not config.sft_run_id: raise ValueError('Supply dataset and initial SFT run IDs')
    data=run_path(root,config.dataset_run_id)
    stage13.replay_saved_run(data)
    expected={f.name:getattr(config,f.name) for f in fields(stage13.FormDatasetConfig)}
    if json.loads((data/'config.json').read_text())!=json.loads(json.dumps(expected)):
        raise ValueError('Dataset/model interface differs')
    adapter=stage11.source_receipt(config,root)
    if adapter['requirements_sha256']!=sha256_file(data/'requirements-browser.txt'):
        raise ValueError('Adapter and dataset dependency locks differ')
    examples=json.loads((data/'demonstrations.json').read_text())
    return {'dataset':{'run_id':data.name,'manifest_sha256':sha256_file(data/'manifest.json'),
                       'demonstrations_sha256':sha256_file(data/'demonstrations.json'),'targets':len(examples)},
            'initial_adapter':adapter}


def load_data(config):
    from PIL import Image

    directory=run_path(RUNS_PATH,config.dataset_run_id)
    examples=json.loads((directory/'demonstrations.json').read_text())
    required={p for e in examples for p in image_paths(e['messages'])}
    with zipfile.ZipFile(directory/'images.zip') as archive:
        assets={p:Image.open(io.BytesIO(archive.read(p))).convert('RGB').copy() for p in required}
    return examples,assets


def load_actor(config, adapter, *, trainable):
    from peft import PeftModel

    actor=BrowserSFTActor(config)
    actor.model.requires_grad_(False)
    with tempfile.TemporaryDirectory() as temporary:
        unpack_adapter(adapter,Path(temporary)/'adapter')
        actor.model=PeftModel.from_pretrained(actor.model,Path(temporary)/'adapter',is_trainable=trainable).eval()
    return actor


def validate_inspections(examples, inspections):
    if len(examples)!=len(inspections): raise ValueError('Every teacher target must have a mask audit')
    for e,i in zip(examples,inspections):
        validate_mask(i)
        if (i['case_id']!=e['case_id'] or i['turn']!=e['turn'] or i['target']!=e['target']
                or i['image_paths']!=image_paths(e['messages'])):
            raise ValueError('Target mask or ordered images differ from the verified teacher example')


def validate_training(config, training, count):
    planned=schedule(config,count)
    rows=training['updates']
    if not 0<len(rows)<=len(planned): raise ValueError('Invalid training cursor')
    for row,expected in zip(rows,planned):
        if any(row[k]!=v for k,v in expected.items()): raise ValueError('Examples skipped, repeated, or reordered')
        if (not all(math.isfinite(row[k]) and row[k]>=0 for k in ('mean_action_loss','gradient_norm','seconds'))
                or row['gradient_norm']==0 or row['target_tokens']<=0): raise ValueError('Invalid SFT update')
    processed=sum(len(r['example_indices']) for r in rows)
    if (training['examples_processed']!=processed or training['completed_updates']!=len(rows)
            or training['total_updates']!=len(planned) or training['dataset_passes']!=processed/count):
        raise ValueError('Incorrect dataset exposure or optimizer counters')


def replay_training(directory):
    manifest=verify_artifacts(directory,stage=14,artifact_files=TRAIN_FILES)
    if manifest.get('kind')!='form-sft-checkpoint': raise ValueError('Not a form SFT checkpoint')
    config=FormSFTConfig(**json.loads((directory/'config.json').read_text()))
    receipt=json.loads((directory/'receipts.json').read_text())
    if source_receipts(config,directory.parent)!=receipt: raise ValueError('Checkpoint input receipts changed')
    training=json.loads((directory/'training.json').read_text())
    validate_training(config,training,receipt['dataset']['targets'])
    inspections=json.loads((directory/'mask-inspection.json').read_text())
    if len(inspections)!=receipt['dataset']['targets']: raise ValueError('Incomplete masks')
    examples=json.loads((run_path(directory.parent,config.dataset_run_id)/'demonstrations.json').read_text())
    validate_inspections(examples,inspections)
    info=json.loads((directory/'checkpoint.json').read_text())
    if (info['completed_updates']!=training['completed_updates'] or info['config']!=json.loads(json.dumps(asdict(config)))
            or info['adapter_fingerprint']!=training['after_fingerprint']
            or any(sha256_file(directory/n)!=h for n,h in info['sha256'].items())
            or not all(training['verification'].values())):
        raise ValueError('Checkpoint/update proof differs')
    return training['completed_updates']


def restore_state(directory, actor, optimizer):
    import torch

    info=json.loads((directory/'checkpoint.json').read_text())
    names=[n for n,p in actor.model.named_parameters() if p.requires_grad]
    if names!=info['parameter_names'] or adapter_fingerprint(actor.model)!=info['adapter_fingerprint']:
        raise ValueError('Adapter or Adam parameter ordering differs')
    state=torch.load(directory/'training-state.pt',map_location='cpu',weights_only=True)
    optimizer.load_state_dict(state['optimizer'])
    if tree_digest(optimizer.state_dict())!=info['optimizer_digest']: raise ValueError('Adam restore mismatch')
    steps=[int(v['step']) for v in optimizer.state_dict()['state'].values()]
    if not steps or any(v!=info['completed_updates'] for v in steps): raise ValueError('Adam counter mismatch')
    restore_rng(state['rng'])
    if tree_digest(capture_rng())!=info['rng_digest']: raise ValueError('RNG restore mismatch')
    return {'adapter_restored':True,'adam_restored':True,'rng_restored':True,'restored_steps':info['completed_updates']}


def result_bundle(directory, files, **metadata):
    return {**metadata,'manifest':json.loads((directory/'manifest.json').read_text()),
            'files':{n:base64.b64encode((directory/n).read_bytes()).decode() for n in (*files,'manifest.json')}}


def train_chunk(run_id, raw_config, source, cache, previous_id, chunk_updates, persist):
    import torch

    config=FormSFTConfig(**raw_config)
    receipts=source_receipts(config)
    if (sha256_file(Path('/root/requirements-browser.txt'))!=receipts['initial_adapter']['requirements_sha256']
            or cache['revision']!=config.model_revision): raise ValueError('Dependency/model revision changed')
    if type(chunk_updates) is not int or not 1<=chunk_updates<=64: raise ValueError('Chunk must have 1–64 updates')
    examples,assets=load_data(config);plan=schedule(config,len(examples))
    previous=run_path(RUNS_PATH,previous_id) if previous_id else None
    if previous:
        cursor=replay_training(previous)
        if (json.loads((previous/'config.json').read_text())!=json.loads(json.dumps(asdict(config)))
                or json.loads((previous/'receipts.json').read_text())!=receipts
                or json.loads((previous/'manifest.json').read_text())['source_sha256']!=source):
            raise ValueError('Resume source/config/code differs')
        rows=json.loads((previous/'training.json').read_text())['updates']
        inspections=json.loads((previous/'mask-inspection.json').read_text())
        validate_inspections(examples,inspections)
        adapter=previous/'adapter.zip'
    else:
        cursor=0;rows=[];inspections=[]
        adapter=run_path(RUNS_PATH,config.sft_run_id)/'adapter.zip'
    if cursor>=len(plan): raise ValueError('Training pass already complete')
    started=time.monotonic()
    actor=load_actor(config,adapter,trainable=True)
    parameters={n:p for n,p in actor.model.named_parameters() if p.requires_grad}
    if (not parameters or any('lora_' not in n or '.language_model.' not in n or p.dtype!=torch.float32
                              for n,p in parameters.items())): raise RuntimeError('Expected FP32 language LoRA only')
    fingerprint=adapter_fingerprint(actor.model)
    optimizer=torch.optim.AdamW(parameters.values(),lr=config.learning_rate,weight_decay=0.)
    resume=restore_state(previous,actor,optimizer) if previous else {'fresh_adam':True,'initial_adapter_loaded':True}
    probes=[]
    if not previous:
        for e in examples:
            batch,inspection=actor.sft_batch(e,assets)
            inspection.pop('tokens',None)
            inspections.append(inspection);del batch
        validate_inspections(examples,inspections)
        longest=max(range(len(inspections)),key=lambda i:len(inspections[i]['input_ids']))
        for index in dict.fromkeys((0,longest)):
            loss,_=_probe(actor,examples[index],assets)
            probes.append({'example_index':index,'causal_shift_checked':True,'loss':loss,
                           'prompt_tokens':inspections[index]['prompt_tokens'],'images':len(inspections[index]['image_paths'])})
        print(f'Audited {len(inspections)} assistant-only targets; probes: {probes}',flush=True)
    actor.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    actor.model.config.use_cache=False
    if not previous:
        rng=capture_rng();tick=time.monotonic()
        actor.model.train();optimizer.zero_grad(set_to_none=True)
        batch,_=actor.sft_batch(examples[longest],assets)
        output=actor.model(**batch,use_cache=False);output.loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.,error_if_nonfinite=True)
        if float(norm)<=0: raise RuntimeError('Longest-history backward probe has no gradient')
        optimizer.zero_grad(set_to_none=True);del batch,output
        restore_rng(rng);torch.cuda.synchronize()
        if adapter_fingerprint(actor.model)!=fingerprint: raise RuntimeError('Probe changed adapter weights')
        probes.append({'longest_backward_checked':True,'example_index':longest,'gradient_norm':float(norm),
                       'seconds':time.monotonic()-tick,'weights_unchanged':True})
        print(f'Longest-history backward probe: {probes[-1]}',flush=True)
    end=min(cursor+chunk_updates,len(plan))
    for item in plan[cursor:end]:
        tick=time.monotonic();actor.model.train();optimizer.zero_grad(set_to_none=True)
        indices=item['example_indices'];losses=[];tokens=0
        for index in indices:
            batch,inspection=actor.sft_batch(examples[index],assets)
            outputs=actor.model(**batch,use_cache=False)
            loss=outputs.loss
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite teacher-action loss')
            (loss/len(indices)).backward()
            losses.append(float(loss.detach()));tokens+=inspection['target_tokens']
            del batch,outputs,loss
        norm=torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.,error_if_nonfinite=True)
        if float(norm)<=0 or any(p.grad is not None for n,p in actor.model.named_parameters() if n not in parameters):
            raise RuntimeError('Expected nonzero language-LoRA gradients and frozen base/vision')
        optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
        row={**item,'mean_action_loss':sum(losses)/len(losses),'gradient_norm':float(norm),
             'target_tokens':tokens,'seconds':time.monotonic()-tick}
        rows.append(row)
        if item['update']<=2 or item['update']%8==0 or item['update']==end:
            print(f"Form SFT {item['update']}/{len(plan)}: {row}",flush=True)
    after=adapter_fingerprint(actor.model)
    if after==fingerprint: raise RuntimeError('SFT did not change the adapter')
    actor.model.gradient_checkpointing_disable();actor.model.config.use_cache=True;actor.model.eval()
    actor.model.peft_config['default'].base_model_name_or_path=config.model_id
    actor.model.peft_config['default'].revision=config.model_revision
    directory=run_path(RUNS_PATH,run_id);directory.mkdir(parents=True,exist_ok=False)
    (directory/'adapter.zip').write_bytes(adapter_zip(actor.model))
    state={'optimizer':clone_cpu(optimizer.state_dict()),'rng':capture_rng()}
    torch.save(state,directory/'training-state.pt')
    info={'config':asdict(config),'completed_updates':end,'parameter_names':list(parameters),
          'adapter_fingerprint':after,'optimizer_digest':tree_digest(state['optimizer']),
          'rng_digest':tree_digest(state['rng']),
          'sha256':{n:sha256_file(directory/n) for n in ('adapter.zip','training-state.pt')}}
    # Verify the actual saved adapter + Adam + RNG before considering this chunk complete.
    actor.model=actor.model.unload().eval()
    from peft import PeftModel
    with tempfile.TemporaryDirectory() as temporary:
        unpack_adapter(directory/'adapter.zip',Path(temporary)/'adapter')
        actor.model=PeftModel.from_pretrained(actor.model,Path(temporary)/'adapter',is_trainable=True).eval()
    write_json(directory/'checkpoint.json',info)
    reloaded_optimizer=torch.optim.AdamW([p for p in actor.model.parameters() if p.requires_grad],lr=config.learning_rate,weight_decay=0.)
    reload_proof=restore_state(directory,actor,reloaded_optimizer)
    processed=sum(len(r['example_indices']) for r in rows)
    training={'updates':rows,'completed_updates':end,'total_updates':len(plan),'examples_processed':processed,
              'dataset_passes':processed/len(examples),'before_fingerprint':fingerprint,'after_fingerprint':after,
              'previous_checkpoint':previous_id,'resume_proof':resume,'saved_reload_proof':reload_proof,'causal_probes':probes,
              'verification':{'all_masks_checked':True,'language_lora_only':True,'base_vision_frozen':True,
                              'saved_adapter_adam_rng_reload':True,'adapter_changed':True}}
    validate_training(config,training,len(examples))
    environment={'gpu':torch.cuda.get_device_name(),'seconds':time.monotonic()-started,
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'trainable_parameters':sum(p.numel() for p in parameters.values()),
                 'optimizer':'AdamW; constant lr; no weight decay; clipping 1', 'cache':cache}
    for name,value in {'config.json':asdict(config),'receipts.json':receipts,'training.json':training,
                       'mask-inspection.json':inspections,'environment.json':environment}.items():write_json(directory/name,value)
    write_manifest(directory,{'kind':'form-sft-checkpoint','source_sha256':source,
        'complete_pass':end==len(plan)},stage=14,artifact_files=TRAIN_FILES)
    replay_training(directory);persist()
    return result_bundle(directory,TRAIN_FILES,training=training,environment=environment)


def development_metrics(episodes):
    if any(e['case']['split']!='dev' for e in episodes): raise ValueError('Only development episodes are allowed')
    return summarize([{**e,'case':{**e['case'],'split':'eval'}} for e in episodes])


def replay_evaluation(directory):
    manifest=verify_artifacts(directory,stage=14,artifact_files=EVAL_FILES)
    if manifest.get('kind')!='form-sft-dev': raise ValueError('Not a development evaluation')
    config=FormSFTConfig(**json.loads((directory/'config.json').read_text()))
    if source_receipts(config,directory.parent)!=json.loads((directory/'receipts.json').read_text()):
        raise ValueError('Evaluation input receipts changed')
    info=json.loads((directory/'evaluation.json').read_text())
    if info['policy'] not in ('source-sft','forms-sft'): raise ValueError('Unknown evaluated policy')
    expected=stage13.pools(config)['dev'][info['offset']:info['offset']+8]
    episodes=json.loads((directory/'episodes.json').read_text())
    if info['offset'] not in (0,8) or [e['case'] for e in episodes]!=expected: raise ValueError('Unexpected dev cases or final-test leakage')
    with zipfile.ZipFile(directory/'images.zip') as archive:
        names=set(archive.namelist())
        for e in episodes:
            validate_episode(e)
            if (e['policy']!=info['policy'] or e['max_steps']!=12
                    or any(o['image'] not in names for o in e['observations'])): raise ValueError('Incomplete evaluation trajectory')
    if development_metrics(episodes)!=json.loads((directory/'metrics.json').read_text()): raise ValueError('Development metrics differ')
    if (directory/'policy-prompt.txt').read_text()!=browser_prompt(12)+'\n': raise ValueError('Evaluation prompt changed')
    return len(episodes)


def evaluate_remote(run_id,raw_config,source,cache,checkpoint_id,offset):
    import torch

    config=FormSFTConfig(**raw_config);receipts=source_receipts(config)
    if offset not in (0,8) or cache['revision']!=config.model_revision: raise ValueError('Unexpected eval shard/model')
    if sha256_file(Path('/root/requirements-browser.txt'))!=receipts['initial_adapter']['requirements_sha256']:
        raise ValueError('Browser dependency lock changed')
    if checkpoint_id:
        cp=run_path(RUNS_PATH,checkpoint_id);cursor=replay_training(cp)
        if (json.loads((cp/'config.json').read_text())!=json.loads(json.dumps(asdict(config)))
                or json.loads((cp/'receipts.json').read_text())!=receipts
                or cursor!=len(schedule(config,receipts['dataset']['targets']))): raise ValueError('Evaluate the completed pass only')
        adapter=cp/'adapter.zip';policy='forms-sft'
        checkpoint_receipt={'run_id':checkpoint_id,'manifest_sha256':sha256_file(cp/'manifest.json'),
                            'adapter_sha256':sha256_file(adapter)}
    else:
        adapter=run_path(RUNS_PATH,config.sft_run_id)/'adapter.zip';policy='source-sft';checkpoint_receipt=None
    started=time.monotonic();actor=load_actor(config,adapter,trainable=False)
    fingerprint=adapter_fingerprint(actor.model);assets={};episodes=[]
    for case in stage13.pools(config)['dev'][offset:offset+8]:
        episode=collect_episode(config,case,policy,assets,actor=actor.act)
        validate_episode(episode);stage8.reexecute_episode(episode);episodes.append(episode)
        print(f"Development {policy} {case['case_id']}: success={episode['success']}, actions={episode['steps']}",flush=True)
    if adapter_fingerprint(actor.model)!=fingerprint: raise RuntimeError('Evaluation changed the adapter')
    metrics=development_metrics(episodes)
    environment={**stage8.runtime(),'max_actions':12,'gpu':torch.cuda.get_device_name(),
                 'seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    evaluation={'offset':offset,'policy':policy,'checkpoint':checkpoint_receipt,
                'adapter_fingerprint':fingerprint,'decoding':actor.generation.to_dict(),'final_test_unopened':True}
    directory=run_path(RUNS_PATH,run_id);directory.mkdir(parents=True,exist_ok=False)
    for name,value in {'config.json':asdict(config),'receipts.json':receipts,'evaluation.json':evaluation,
                       'episodes.json':episodes,'metrics.json':metrics,'environment.json':environment}.items():write_json(directory/name,value)
    (directory/'images.zip').write_bytes(_images_zip(assets));(directory/'policy-prompt.txt').write_text(browser_prompt(12)+'\n')
    (directory/'report.html').write_text(stage8.report_html({'episodes':episodes,'metrics':metrics,
        'checks':{'development_only':True,'actual_actions_live_reexecuted':True,'final_test_unopened':True}},assets,
        title='Form SFT development evaluation',description='Development instances only. Greedy decoding; final test remains closed.'))
    write_manifest(directory,{'kind':'form-sft-dev','source_sha256':source},stage=14,artifact_files=EVAL_FILES)
    replay_evaluation(directory)
    return result_bundle(directory,EVAL_FILES,metrics=metrics,environment=environment)


def paired_comparison(episodes, config):
    expected=stage13.pools(config)['dev']
    groups={p:[e for e in episodes if e['policy']==p] for p in ('source-sft','forms-sft')}
    for members in groups.values():
        if len(members)!=16 or {e['case']['case_id'] for e in members}!={c['case_id'] for c in expected}:
            raise ValueError('Paired evaluation requires all sixteen dev cases, once per policy')
        if any(e['case'] not in expected for e in members): raise ValueError('Unknown development instance')
    before={e['case']['case_id']:e for e in groups['source-sft']}
    for e in groups['forms-sft']:
        a=before[e['case']['case_id']]['observations'][0];b=e['observations'][0]
        if any(a[k]!=b[k] for k in ('goal','width','height','pixel_sha256')): raise ValueError('Paired initial observations differ')
    return {'development_metrics':development_metrics(episodes),'paired_cases':16,'final_test_unopened':True,
            'scope':'Same-template development comparison. Development is available for subsequent choices; final test remains closed.'}

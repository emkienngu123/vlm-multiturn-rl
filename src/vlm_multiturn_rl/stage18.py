"""Fresh H100 continuation of the Stage 16 learning checkpoint, not Stage 17 replay."""
from dataclasses import fields
from html import escape
import json
from pathlib import Path

from . import stage14, stage16
from .reporting import wilson_interval
from .vision import Stage2Config
from .workspace import RUNS_PATH, new_run_id, run_path, sha256_file, write_json

ORIGIN = 'stage16-20261019T155940Z-0927d223'
PHASE_SECONDS = 1800


def execution_receipt(root=RUNS_PATH):
    origin = run_path(root, ORIGIN)
    if stage16.replay_saved_run(origin) != 4: raise ValueError('Expected the four-round learning checkpoint')
    cp = json.loads((origin/'checkpoint/checkpoint.json').read_text())
    if cp['optimizer_steps'] != 2: raise ValueError('Expected two actual GRPO updates')
    return {'kind': 'h100-continuation-v1', 'origin_run_id': ORIGIN,
        'origin_manifest_sha256': sha256_file(origin/'manifest.json'),
        'origin_source_sha256': json.loads((origin/'manifest.json').read_text())['source_sha256'],
        'origin_adapter_fingerprint': cp['adapter_fingerprint'], 'starting_round': 4, 'starting_optimizer_steps': 2,
        'activation_gpu_budget_bytes': 40*1024**3, 'requested_gpu': 'H100!', 'phase_training_seconds': PHASE_SECONDS}


def validate_execution(execution, root=RUNS_PATH):
    if execution != execution_receipt(root): raise ValueError('H100 migration receipt changed')


def validate_source_transition(execution, parent_id, parent_source, current_source, parent_execution):
    """Permit exactly one explicit hardware/code boundary; later resumes stay strict."""
    if execution and parent_id == execution['origin_run_id'] and parent_execution is None:
        if parent_source != execution['origin_source_sha256']: raise ValueError('Migration origin source changed')
        return
    if parent_source != current_source or parent_execution != execution:
        raise ValueError('Resume core source or execution profile changed outside migration boundary')


def budget_decision(rows, starting_round=4):
    phase = rows[starting_round:]
    elapsed = sum(r['seconds'] for r in phase)
    predicted = max(r['seconds'] for r in phase)*1.25+30 if phase else 450.
    return {'elapsed': elapsed, 'predicted': predicted, 'can_start': elapsed+predicted <= PHASE_SECONDS}


def run_workflow(root, source, prepare, model, train, evaluate, save):
    """Local orchestration; all model imports and computation stay remote."""
    root = Path(root); execution = execution_receipt(root)
    config = json.loads((root/ORIGIN/'config.json').read_text())
    obj = stage16.FormGRPOConfig(**config)
    workflow = {'config': config, 'execution': execution, 'plan': stage16.plan(obj),
        'status': 'preflight', 'rounds': [], 'eval_phases': {}, 'last_checkpoint': ORIGIN,
        'phase_training_seconds_limit': PHASE_SECONDS, 'final_test_unopened': True}
    path = root/(new_run_id(stage=18)+'-workflow.json'); write_json(path, workflow)
    cpu_id = new_run_id(stage=18)
    save(prepare(cpu_id, config, source), root)
    workflow['cpu_run_id'] = cpu_id; write_json(path, workflow)
    cache = model({f.name: config[f.name] for f in fields(Stage2Config)})
    def phase(label, checkpoint):
        workflow['status'] = 'evaluating-'+label; workflow['eval_phases'][label] = []; write_json(path, workflow)
        for offset in (0, 8):
            next_id = new_run_id(stage=18)
            save(evaluate(next_id, config, source, cache, checkpoint, offset), root)
            workflow['eval_phases'][label].append(next_id); write_json(path, workflow)
    def fingerprint(checkpoint):
        return json.loads((root/checkpoint/'checkpoint/checkpoint.json').read_text())['adapter_fingerprint']
    def reuse(label, from_label, checkpoint):
        workflow['eval_phases'][label] = workflow['eval_phases'][from_label]
        workflow.setdefault('evaluation_reuse', {})[label] = {'from_phase': from_label,
            'reason': 'identical saved adapter; no additional independent evaluation',
            'adapter_fingerprint': fingerprint(checkpoint)}
        write_json(path, workflow)
    phase('before', ORIGIN)
    previous = ORIGIN
    rows = json.loads((root/ORIGIN/'training.json').read_text())['history']
    while len(rows) < obj.max_rounds:
        decision = budget_decision(rows)
        workflow['budget_decision'] = decision
        if not decision['can_start']:
            workflow['stop_reason'] = 'new_phase_training_time_budget'; break
        workflow['status'] = 'training'; write_json(path, workflow)
        next_id = new_run_id(stage=18)
        result = train(next_id, config, source, cache, cpu_id, previous, execution)
        save(result, root)
        rows = result['training']['history']; previous = next_id
        workflow['rounds'].append({'run_id': next_id, **rows[-1]})
        workflow['last_checkpoint'] = previous; write_json(path, workflow)
        if len(rows) == 6:
            if fingerprint(previous) == fingerprint(ORIGIN): reuse('smoke', 'before', previous)
            else: phase('smoke', previous)
    # Reuse only complete H100 evaluation of the exact final adapter.
    reused = False
    for label in ('smoke', 'before'):
        if label not in workflow['eval_phases']: continue
        infos = [json.loads((root/n/'evaluation.json').read_text()) for n in workflow['eval_phases'][label]]
        if all(i['adapter_fingerprint'] == fingerprint(previous) for i in infos):
            reuse('after', label, previous); reused = True; break
    if not reused: phase('after', previous)
    workflow.update(status='complete', completed_rounds=len(rows), optimizer_steps=rows[-1]['optimizer_steps'],
        new_rounds=len(rows)-4, new_optimizer_steps=rows[-1]['optimizer_steps']-2,
        recorded_phase_training_seconds=sum(r['seconds'] for r in rows[4:]),
        stop_reason=workflow.get('stop_reason', 'twenty_total_round_target'))
    write_json(path, workflow)
    summary = summarize(path)
    print(json.dumps({'workflow': str(path), 'new_rounds': summary['new_rounds'],
        'new_optimizer_steps': summary['new_optimizer_steps'], 'development': summary['phases'],
        'last_checkpoint': previous, 'stop_reason': summary['stop_reason']}, indent=2), flush=True)
    return path


def summarize(path):
    """Verify provenance, numerical guards, fresh groups, and paired H100 evaluation."""
    path = Path(path); root = path.parent; w = json.loads(path.read_text())
    if w['status'] != 'complete': raise ValueError('Workflow is incomplete')
    validate_execution(w['execution'], root)
    config = stage16.FormGRPOConfig(**w['config'])
    if w['plan'] != stage16.plan(config): raise ValueError('Schedule changed')
    cpu = root/w['cpu_run_id']; stage16.replay_saved_run(cpu)
    source = json.loads((cpu/'manifest.json').read_text())['source_sha256']
    previous = ORIGIN; phase_rows = []; checkpoints = []
    for record in w['rounds']:
        p = root/record['run_id']; stage16.replay_saved_run(p)
        manifest = json.loads((p/'manifest.json').read_text()); env = json.loads((p/'environment.json').read_text())
        training = json.loads((p/'training.json').read_text())
        if (manifest['source_sha256'] != source or env.get('execution') != w['execution']
                or training['previous']['run_id'] != previous
                or training['history'][-1] != {k:v for k,v in record.items() if k != 'run_id'}
                or json.loads((p/'cpu-source.json').read_text()) !=
                   {'run_id': cpu.name, 'manifest_sha256': sha256_file(cpu/'manifest.json')}):
            raise ValueError('H100 continuation chain changed')
        if not all(training['restore'][k] for k in ('adapter_matches','adam_matches','rng_matches')):
            raise ValueError('Missing restore proof')
        phase_rows.append(training['history'][-1]); checkpoints.append(p); previous = p.name
    if previous != w['last_checkpoint']: raise ValueError('Last checkpoint differs')
    last = root/previous
    rows = json.loads((last/'training.json').read_text())['history']
    if (w['new_rounds'] != len(phase_rows) or rows[4:] != phase_rows
            or w['new_optimizer_steps'] != rows[-1]['optimizer_steps']-2): raise ValueError('Counters changed')
    phases = {}; signatures = None; decoding = None; outcomes = {}
    for label, ids in w['eval_phases'].items():
        episodes = []; fp = None
        if len(ids) != 2: raise ValueError('Missing evaluation shards')
        for offset, name in zip((0,8), ids):
            p = root/name; stage16.replay_saved_run(p)
            env = json.loads((p/'environment.json').read_text()); info = json.loads((p/'evaluation.json').read_text())
            if ('H100' not in env['gpu'] or info['offset'] != offset
                    or json.loads((p/'manifest.json').read_text())['source_sha256'] != source):
                raise ValueError('Development GPU, offset, or code differs')
            cp = root/info['checkpoint']['run_id']; stage16.replay_saved_run(cp)
            expected_fp = json.loads((cp/'checkpoint/checkpoint.json').read_text())['adapter_fingerprint']
            if (info['checkpoint']['manifest_sha256'] != sha256_file(cp/'manifest.json')
                    or info['adapter_fingerprint'] != expected_fp or (fp is not None and fp != expected_fp)
                    or (decoding is not None and decoding != info['decoding'])): raise ValueError('Development policy differs')
            fp = expected_fp; decoding = info['decoding']; episodes.extend(json.loads((p/'episodes.json').read_text()))
        if [e['case'] for e in episodes] != stage16.plan(config)['dev']: raise ValueError('Development pool differs')
        initial = [(e['observations'][0]['goal'],e['observations'][0]['pixel_sha256']) for e in episodes]
        if signatures is not None and initial != signatures: raise ValueError('Paired screenshots/instructions differ')
        signatures = initial; outcomes[label] = [e['success'] for e in episodes]
        metrics = stage14.development_metrics(episodes)
        successes = sum(outcomes[label])
        phases[label] = {'successes': successes, 'episodes':16, 'wilson_95':wilson_interval(successes,16),
            'metrics':metrics, 'adapter_fingerprint':fp, 'run_ids':ids,
            'mean_actions':sum(e['steps'] for e in episodes)/16,
            'actions':sum(e['steps'] for e in episodes),
            'invalid_actions':sum(e['invalid_actions'] for e in episodes),
            'execution_errors':sum(e['execution_errors'] for e in episodes)}
    for label, cp in (('before',root/ORIGIN),('after',last)):
        if phases[label]['adapter_fingerprint'] != json.loads((cp/'checkpoint/checkpoint.json').read_text())['adapter_fingerprint']:
            raise ValueError('Before/after does not evaluate the correct checkpoint')
    summary = {'new_rounds':len(phase_rows),'new_optimizer_steps':w['new_optimizer_steps'],
        'total_rounds':len(rows),'total_optimizer_steps':rows[-1]['optimizer_steps'],
        'new_skipped_groups':sum(r['group']['zero_variance'] for r in phase_rows),
        'new_sampled_trajectories':4*len(phase_rows), 'phase_training_seconds':sum(r['seconds'] for r in phase_rows),
        'stop_reason':w['stop_reason'],'last_checkpoint':last.name,'phases':phases,'final_test_unopened':True,
        'paired_changes':{'failure_to_success':sum(not a and b for a,b in zip(outcomes['before'],outcomes['after'])),
            'success_to_failure':sum(a and not b for a,b in zip(outcomes['before'],outcomes['after']))},
        'peak_training_allocated_gib':max((json.loads((p/'environment.json').read_text())['peak_allocated_bytes']/1024**3 for p in checkpoints),default=0),
        'checkpoint_core_bytes':sum((last/'checkpoint'/n).stat().st_size for n in
                                    ('adapter.zip','training-state.pt','checkpoint.json')),
        'evaluation_reuse':w.get('evaluation_reuse',{}),'rounds':phase_rows}
    write_json(path.with_name(path.stem+'-summary.json'),summary)
    table=''.join('<tr>'+''.join(f'<td>{escape(str(x))}</td>' for x in
        (label,f"{data['successes']}/16",data['mean_actions'],
         f"{data['invalid_actions']}/{data['actions']}"))+'</tr>' for label,data in phases.items())
    links=''.join(f'<li><a href="{p.name}/report.html">Round {r["round"]}: {p.name}</a></li>' for p,r in zip(checkpoints,phase_rows))
    html='<!doctype html><meta charset="utf-8"><title>H100 GRPO continuation</title><style>body{font:16px system-ui;max-width:1100px;margin:35px auto}td,th{padding:10px;border:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
    html+='<h1>H100 GRPO continuation</h1><p>Fresh on-policy groups, original remaining schedule, real Adam/RNG continuation. Before and after use the same H100 hardware and 16 development layouts. Final test remains closed.</p>'
    html+=f'<p>{len(phase_rows)} new groups; {w["new_optimizer_steps"]} new optimizer steps. Sixteen development cases on one familiar template provide limited evidence of generalization.</p>'
    html+='<table><tr><th>Phase</th><th>Successes</th><th>Mean actions</th><th>Invalid/actions</th></tr>'+table+'</table><h2>Training replays</h2><ul>'+links+'</ul><h2>Verified results</h2><pre>'+escape(json.dumps(summary,indent=2))+'</pre>'
    path.with_suffix('.html').write_text(html)
    return summary

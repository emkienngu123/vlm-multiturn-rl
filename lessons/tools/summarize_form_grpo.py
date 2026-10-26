"""Verify a completed Stage 16 chain and pair development comparisons; no Torch."""
from html import escape
import argparse
import json
from pathlib import Path

from vlm_multiturn_rl import stage14, stage16
from vlm_multiturn_rl.reporting import wilson_interval
from vlm_multiturn_rl.workspace import sha256_file, write_json


def summarize(path):
    workflow = json.loads(path.read_text()); root = path.parent
    if workflow['status'] != 'complete': raise ValueError('Wait for a complete workflow')
    config = stage16.FormGRPOConfig(**workflow['config'])
    if workflow['plan'] != stage16.plan(config): raise ValueError('Study plan changed')
    cpu = root/workflow['cpu_run_id']; stage16.replay_saved_run(cpu)
    source = json.loads((cpu/'manifest.json').read_text())['source_sha256']
    checkpoints = []; cursor = workflow['last_checkpoint']
    while cursor:
        d = root/cursor
        if cursor in checkpoints: raise ValueError('Cyclic continuation chain')
        stage16.replay_saved_run(d); checkpoints.append(cursor)
        if json.loads((d/'manifest.json').read_text())['source_sha256'] != source: raise ValueError('Core source changed')
        training = json.loads((d/'training.json').read_text())
        receipt = json.loads((d/'cpu-source.json').read_text())
        if receipt != {'run_id': cpu.name, 'manifest_sha256': sha256_file(cpu/'manifest.json')}: raise ValueError('Preflight reference differs')
        cursor = training['previous']['run_id'] if training['previous'] else ''
    last = root/workflow['last_checkpoint']; rows = json.loads((last/'training.json').read_text())['history']
    if len(rows) != len(checkpoints) or workflow['completed_rounds'] != len(rows): raise ValueError('Incomplete round history')
    if workflow['optimizer_steps'] != rows[-1]['optimizer_steps']: raise ValueError('Step count differs')
    phases = {}; outcomes = {}; decoding = None; signatures = None
    def add_phase(label, directories, reader, *, full_sft=False):
        nonlocal decoding, signatures
        episodes = []; fingerprint = None
        if len(directories) != 2: raise ValueError('A phase requires two eight-case shards')
        for d in directories:
            reader(d)
            info = json.loads((d/'evaluation.json').read_text())
            if not full_sft:
                if info['checkpoint'] is None:
                    expected_fp = workflow['receipts']['early']['adapter_fingerprint']
                else:
                    cp = root/info['checkpoint']['run_id']; stage16.replay_saved_run(cp)
                    if info['checkpoint']['manifest_sha256'] != sha256_file(cp/'manifest.json'):
                        raise ValueError('Evaluation checkpoint receipt differs')
                    expected_fp = json.loads((cp/'checkpoint/checkpoint.json').read_text())['adapter_fingerprint']
                if info['adapter_fingerprint'] != expected_fp: raise ValueError('Wrong evaluation adapter')
            if decoding is not None and decoding != info['decoding']: raise ValueError('Decoding differs')
            decoding = info['decoding']
            if fingerprint is not None and fingerprint != info['adapter_fingerprint']: raise ValueError('Mixed phase policies')
            fingerprint = info['adapter_fingerprint']
            if not full_sft and json.loads((d/'manifest.json').read_text())['source_sha256'] != source:
                raise ValueError('Evaluation code differs')
            episodes.extend(json.loads((d/'episodes.json').read_text()))
        if (len(episodes) != 16 or [e['case'] for e in episodes] != stage16.plan(config)['dev']):
            raise ValueError('Phase has missing, duplicated or unexpected development cases')
        initial = [(e['observations'][0]['goal'], e['observations'][0]['pixel_sha256']) for e in episodes]
        if signatures is not None and initial != signatures: raise ValueError('Paired initial states differ')
        signatures = initial
        successes = sum(e['success'] for e in episodes); actions = sum(e['steps'] for e in episodes)
        outcomes[label] = [e['success'] for e in episodes]
        phases[label] = {'successes': successes, 'episodes': 16, 'success_rate': successes/16,
            'success_wilson_95': wilson_interval(successes, 16), 'mean_actions': actions/16,
            'actions': actions, 'invalid_actions': sum(e['invalid_actions'] for e in episodes),
            'execution_errors': sum(e['execution_errors'] for e in episodes), 'adapter_fingerprint': fingerprint,
            'run_ids': [d.name for d in directories]}
    for label, names in workflow['eval_phases'].items():
        add_phase(label, [root/n for n in names], stage16.replay_saved_run)
    last_fp = json.loads((last/'checkpoint/checkpoint.json').read_text())['adapter_fingerprint']
    if phases['after']['adapter_fingerprint'] != last_fp:
        raise ValueError('Final development results must evaluate the last saved policy')
    comparison = []
    for d in sorted(root.glob('stage14-*')):
        p = d/'evaluation.json'
        if not p.exists(): continue
        info = json.loads(p.read_text())
        if info.get('checkpoint', {}) and info['checkpoint']['run_id'] == config.comparison_run_id:
            comparison.append((info['offset'], d))
    if len(comparison) != 2: raise ValueError('Expected the two verified full-SFT comparison shards')
    add_phase('full-sft-reference', [d for _, d in sorted(comparison)], stage14.replay_evaluation, full_sft=True)
    if phases['full-sft-reference']['adapter_fingerprint'] != workflow['receipts']['final']['adapter_fingerprint']:
        raise ValueError('Wrong full-SFT comparator')
    report = {'development_only': True, 'final_test_unopened': True, 'attempted_rounds': len(rows),
        'optimizer_steps': rows[-1]['optimizer_steps'], 'skipped_groups': sum(r['group']['zero_variance'] for r in rows),
        'training_instances_visited': len({r['group']['case_id'] for r in rows}),
        'training_instances_with_updates': len({r['group']['case_id'] for r in rows if r['update']['optimizer_step']}),
        'sampled_trajectories': 4*len(rows), 'recorded_training_seconds': sum(r['seconds'] for r in rows),
        'stop_reason': workflow['stop_reason'], 'last_checkpoint': last.name, 'phases': phases,
        'evaluation_reuse': workflow.get('evaluation_reuse'),
        'peak_training_allocated_gib': max(json.loads((root/n/'environment.json').read_text())['peak_allocated_bytes'] for n in checkpoints)/1024**3,
        'rounds': rows}
    before_outcomes = outcomes['before']
    report['paired_changes'] = {label: {
        'paired_cases': 16, 'failure_to_success': sum(not a and b for a, b in zip(before_outcomes, values)),
        'success_to_failure': sum(a and not b for a, b in zip(before_outcomes, values)),
        'retained_success': sum(a and b for a, b in zip(before_outcomes, values))}
        for label, values in outcomes.items() if label != 'before'}
    train_initial = {(e['observations'][0]['goal'], e['observations'][0]['pixel_sha256'])
                     for e in json.loads((cpu/'episodes.json').read_text())}
    report['exact_training_development_initial_overlaps'] = len(train_initial & set(signatures))
    new_evaluations = {n for names in workflow['eval_phases'].values() for n in names}
    report['recorded_development_seconds'] = sum(json.loads((root/n/'environment.json').read_text())['seconds'] for n in new_evaluations)
    report['completed_model_episode_replays'] = 4*len(rows)+8*len(new_evaluations)
    report['completed_teacher_episode_replays'] = len(stage16.plan(config)['train'])
    report['checkpoint_core_bytes'] = sum((last/'checkpoint'/n).stat().st_size for n in
        ('adapter.zip', 'training-state.pt', 'checkpoint.json'))
    table = []
    display_names = ['before']
    if 'smoke' in phases and phases['smoke']['adapter_fingerprint'] != phases['after']['adapter_fingerprint']:
        display_names.append('smoke')
    display_names += ['after', 'full-sft-reference']
    for label in display_names:
        p = phases[label]
        table.append('<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in
            (label, f"{p['successes']}/16", p['mean_actions'], f"{p['invalid_actions']}/{p['actions']}"))+'</tr>')
    links = ''.join(f'<li><a href="{n}/report.html">Round {i+1}: {n}</a></li>' for i, n in enumerate(reversed(checkpoints)))
    plot = path.with_name(path.stem+'-training.png')
    diagnostics = f'<h2>Diagnostics</h2><img src="{escape(plot.name)}" style="width:100%" alt="Training returns, gradient norms, and paired development success">' if plot.exists() else ''
    html = ('<!doctype html><meta charset="utf-8"><title>Form GRPO learning</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:36px auto}table{border-collapse:collapse}'
        'th,td{border:1px solid #ccc;padding:10px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
        '<h1>Stage 16: form GRPO after short SFT</h1><p>Paired development comparison on a familiar template. '
        'Final-test cases remain closed. Training groups and actual optimizer updates are counted separately.</p>'
        + ('<p>The last policy is identical to the complete smoke evaluation after two skipped rounds. '
           'Final results reuse that evaluation; duplicate evaluation was stopped.</p>' if report['evaluation_reuse'] else '') +
        '<table><tr><th>Phase</th><th>Successes</th><th>Mean actions</th><th>Invalid actions</th></tr>'+''.join(table)+'</table>'
        + diagnostics + '<h2>Round reports and replays</h2><ul>'+links+'</ul><h2>Verified experiment</h2><pre>'+escape(json.dumps(report, indent=2))+'</pre>')
    write_json(path.with_name(path.stem+'-summary.json'), report); path.with_suffix('.html').write_text(html)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('workflow', type=Path)
    args = parser.parse_args(); result = summarize(args.workflow)
    print(json.dumps({k: v for k, v in result.items() if k != 'rounds'}, indent=2))

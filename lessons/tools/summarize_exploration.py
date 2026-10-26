"""Summarize all eight verified Stage 15 shards; standard library only."""
import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path

from vlm_multiturn_rl.stage15 import ExplorationConfig, readiness, replay_saved_run, study_plan
from vlm_multiturn_rl.workspace import sha256_file, write_json


def summarize_workflow(path):
    workflow = json.loads(path.read_text()); root = path.parent
    config = ExplorationConfig(**workflow['config']); plan = study_plan(config)
    if workflow['status'] != 'complete' or workflow['plan'] != plan:
        raise ValueError('A complete, unmodified eight-group workflow is required')
    if [{k: s[k] for k in ('index', 'candidate', 'case_id')} for s in workflow['shards']] != plan['shards']:
        raise ValueError('Missing, duplicated or reordered study shards')
    cpu = root/workflow['cpu_run_id']; replay_saved_run(cpu)
    source = json.loads((cpu/'manifest.json').read_text())['source_sha256']
    teachers = {e['case']['case_id']: e for e in json.loads((cpu/'episodes.json').read_text())}
    pools = defaultdict(lambda: {'episodes': [], 'groups': [], 'run_ids': []})
    seconds = peak = 0.; decoding = None; signatures = set(); actions = 0
    for entry in workflow['shards']:
        d = root/entry['run_id']; replay_saved_run(d)
        if json.loads((d/'manifest.json').read_text())['source_sha256'] != source: raise ValueError('Core source differs')
        if json.loads((d/'config.json').read_text()) != workflow['config']: raise ValueError('Study configuration differs')
        if json.loads((d/'sources.json').read_text()) != workflow['sources']: raise ValueError('Source receipts differ')
        if json.loads((d/'cpu-source.json').read_text()) != {'run_id': cpu.name, 'manifest_sha256': sha256_file(cpu/'manifest.json')}:
            raise ValueError('Preflight receipt differs')
        environment = json.loads((d/'environment.json').read_text())
        expected = {k: entry[k] for k in ('index', 'candidate', 'case_id')}
        if environment['shard'] != expected: raise ValueError('Shard identity differs')
        if decoding is not None and environment['sampling_decoding'] != decoding: raise ValueError('Sampling settings differ')
        decoding = environment['sampling_decoding']
        episodes = json.loads((d/'episodes.json').read_text()); groups = json.loads((d/'groups.json').read_text())
        if readiness(episodes, groups) != entry['readiness']: raise ValueError('Workflow summary differs')
        key = (entry['candidate'], episodes[0]['case']['task'])
        pool = pools[key]; pool['episodes'].extend(episodes); pool['groups'].extend(groups); pool['run_ids'].append(d.name)
        teacher = teachers[entry['case_id']]['observations'][0]
        for e in episodes:
            if (e['episode_id'] in signatures or any(e['observations'][0][k] != teacher[k] for k in ('goal', 'pixel_sha256'))):
                raise ValueError('Repeated trajectory or changed initial board')
            signatures.add(e['episode_id']); actions += e['steps']
        seconds += environment['seconds']; peak = max(peak, environment['peak_allocated_bytes'])
    report = {'workflow': path.name, 'training_only': True, 'optimizer_steps': 0,
              'trajectories': len(signatures), 'actions': actions, 'groups': 8,
              'gpu_function_seconds': seconds, 'peak_allocated_gib': peak/1024**3,
              'cpu_preflight_seconds': json.loads((cpu/'environment.json').read_text())['seconds'],
              'final_test_unopened': True, 'pools': []}
    for (candidate, task), pool in pools.items():
        report['pools'].append({'candidate': candidate, 'task': task,
            'checkpoint': workflow['sources'][candidate]['run_id'], 'run_ids': pool['run_ids'],
            'sampled_actions': sum(e['steps'] for e in pool['episodes']),
            'invalid_actions': sum(e['invalid_actions'] for e in pool['episodes']),
            'execution_errors': sum(e['execution_errors'] for e in pool['episodes']),
            'max_ratio_deviation': max(g['max_ratio_deviation'] for g in pool['groups']),
            'oracle_actions': [teachers[g['case_id']]['steps'] for g in pool['groups']],
            **readiness(pool['episodes'], pool['groups'])})
    rows = []
    for p in report['pools']:
        m = p['training_metrics']['all']
        links = ', '.join(f'<a href="{escape(n)}/report.html">group {i+1}</a>' for i, n in enumerate(p['run_ids']))
        rows.append('<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in
            (p['candidate'], p['task'], f"{m['successes']}/{m['episodes']}",
             p['mixed_return_groups'], p['returns'], m['mean_steps']))+f'<td>{links}</td></tr>')
    html = ('<!doctype html><meta charset="utf-8"><title>Exploration after SFT</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:36px auto}table{border-collapse:collapse}'
        'td,th{border:1px solid #ccc;padding:10px;text-align:left}pre{white-space:pre-wrap}</style>'
        '<h1>Stage 15: training-only exploration</h1><p>Fixed 32-trajectory study. No optimization. '
        'Final-test instances remain unopened. These are training-task selection measurements, not held-out performance.</p>'
        '<table><tr><th>Checkpoint</th><th>Task</th><th>Successes</th><th>Mixed groups</th><th>Group returns</th>'
        '<th>Mean actions</th><th>Replays</th></tr>'+''.join(rows)+'</table>'
        '<h2>Verified study details</h2><pre>'+escape(json.dumps(report, indent=2))+'</pre>')
    write_json(path.with_name(path.stem+'-summary.json'), report)
    path.with_suffix('.html').write_text(html)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('workflow', type=Path)
    args = parser.parse_args(); print(json.dumps(summarize_workflow(args.workflow), indent=2))

from collections import Counter
from dataclasses import asdict
from html import escape
import json
from pathlib import Path
import sys

from vlm_multiturn_rl.stage14 import FormSFTConfig, paired_comparison, replay_evaluation, replay_training
from vlm_multiturn_rl.reporting import wilson_interval
from vlm_multiturn_rl.workspace import source_hashes

path=Path(sys.argv[1]);workflow=json.loads(path.read_text());root=path.parent
assert workflow['status']=='complete'
config=FormSFTConfig(**workflow['config'])
last=root/workflow['last_checkpoint'];replay_training(last)
training=json.loads((last/'training.json').read_text())
assert training['dataset_passes']==config.epochs
hashes=source_hashes(Path.cwd());episodes=[];decoding=None;fps={};envs=[]
for item in workflow['chunks']:
 d=root/item['run_id'];replay_training(d)
 assert json.loads((d/'manifest.json').read_text())['source_sha256']==hashes
 envs.append(json.loads((d/'environment.json').read_text()))
for key in workflow['eval_shards']:
 d=root/key;replay_evaluation(d)
 assert json.loads((d/'manifest.json').read_text())['source_sha256']==hashes
 e=json.loads((d/'evaluation.json').read_text());policy=e['policy']
 if decoding is not None:assert decoding==e['decoding']
 decoding=e['decoding']
 if policy in fps:assert fps[policy]==e['adapter_fingerprint']
 fps[policy]=e['adapter_fingerprint']
 episodes.extend(json.loads((d/'episodes.json').read_text()))
 envs.append(json.loads((d/'environment.json').read_text()))
assert fps['forms-sft']==training['after_fingerprint']
if workflow['chunks'] and workflow['chunks'][0]['from_update']==0:
 first=json.loads((root/workflow['chunks'][0]['run_id']/'training.json').read_text())
 assert fps['source-sft']==first['before_fingerprint']
comparison=paired_comparison(episodes,config);assert comparison==workflow['comparison']
metrics=comparison['development_metrics']
for policy in metrics:
 m=metrics[policy]['all'];m['success_rate_95pct_wilson']=wilson_interval(m['successes'],m['episodes'])
before={e['case']['case_id']:e for e in episodes if e['policy']=='source-sft'}
after={e['case']['case_id']:e for e in episodes if e['policy']=='forms-sft'}
paired=[{'case_id':key,'before':before[key]['success'],'after':after[key]['success'],
         'before_steps':before[key]['steps'],'after_steps':after[key]['steps']} for key in sorted(before)]
train=json.loads((root/config.dataset_run_id/'episodes.json').read_text())
train_pairs={(e['observations'][0]['goal'],e['observations'][0]['pixel_sha256']) for e in train}
dev_pairs={(e['observations'][0]['goal'],e['observations'][0]['pixel_sha256']) for e in before.values()}
summary={'exact_initial_observation_pair_overlap':len(train_pairs & dev_pairs),'comparison':comparison,'paired_cases':paired,'optimizer_updates':training['completed_updates'],
 'examples_processed':training['examples_processed'],'dataset_passes':training['dataset_passes'],
 'gpu_function_seconds':sum(e['seconds'] for e in envs),
 'training_update_seconds':sum(r['seconds'] for r in training['updates']),
 'peak_allocated_gib':max(e['peak_allocated_bytes'] for e in envs)/2**30,
 'training_checkpoint_mib':sum((last/n).stat().st_size for n in ('adapter.zip','training-state.pt','checkpoint.json'))/2**20,
 'all_recorded_source_hashes_match':True,'final_test_unopened':True,
 'note':'Development within one familiar template. Recovery ability is not separately evaluated. GPU function time excludes startup/cache preparation.'}
output=path.with_name(path.stem+'-comparison.json');output.write_text(json.dumps(summary,indent=2)+'\n')
rows=''.join('<tr><th>'+escape(p)+'</th><td>'+str(m['all']['successes'])+'/16</td><td>'+str(round(m['all']['mean_steps'],3))+'</td><td>'+str(round(100*m['all']['invalid_action_rate'],2))+'%</td></tr>' for p,m in metrics.items())
links=''.join('<li><a href="'+escape(k+'/report.html')+'">'+escape(k)+'</a></li>' for k in workflow['eval_shards'])
html='<!doctype html><meta charset="utf-8"><title>Form SFT comparison</title><style>body{font:16px system-ui;max-width:1000px;margin:32px auto}table{border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #aaa;text-align:left}pre{white-space:pre-wrap}</style>'
html+='<h1>One-pass form SFT: paired development comparison</h1><p>Same 16 instances and greedy decoding. Final test remains closed.</p><table><tr><th>Policy</th><th>Success</th><th>Mean actions</th><th>Invalid actions</th></tr>'+rows+'</table><h2>Episode replays</h2><ul>'+links+'</ul><h2>Validation and measurements</h2><pre>'+escape(json.dumps(summary,indent=2))+'</pre>'
plot=path.with_name(path.stem+'-training.png')
if plot.is_file():
 html+='<h2>Training diagnostics</h2><p>Different teacher-action batches; these curves are not held-out performance.</p><img style="width:100%" src="'+escape(plot.name)+'">'
path.with_suffix('.html').write_text(html)
print(json.dumps(summary,indent=2))

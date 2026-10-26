"""Fixed-budget, training-only exploration after form SFT. No optimizer steps."""

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

from . import stage8, stage14
from .browser import browser_prompt, collect_episode, summarize, validate_episode
from .stage4 import _images_zip
from .stage5 import adapter_fingerprint, unpack_adapter
from .stage10 import _write_progress, group_record, trajectory_seed
from .stage11 import BrowserDifficultyConfig
from .trajectories import image_paths
from .vision import Stage2Config
from .workspace import RUNS_PATH, run_path, sha256_file, verify_artifacts, write_json, write_manifest

CPU_FILES = ('config.json', 'sources.json', 'plan.json', 'episodes.json', 'images.zip',
             'checks.json', 'environment.json', 'requirements-browser.txt', 'policy-prompt.txt', 'report.html')
GPU_FILES = (*CPU_FILES, 'groups.json', 'readiness.json', 'cpu-source.json', 'model-cache.json', 'chat-template.txt')


@dataclass(frozen=True)
class ExplorationConfig(BrowserDifficultyConfig):
    warmup_run_id: str = ''
    ratio_tolerance: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if self.episodes != 2 or self.ratio_tolerance != 1e-4:
            raise ValueError('Use two training seeds and the unchanged 1e-4 behavior-ratio tolerance')
        for value in (self.sft_run_id, self.warmup_run_id):
            if value: run_path(Path('.'), value)
        if self.sft_run_id and self.sft_run_id == self.warmup_run_id:
            raise ValueError('Final and early checkpoints must differ')


def study_plan(config):
    cases = [{'case_id': f'{task}-explore-train-{i:03d}', 'task': task, 'split': 'train', 'seed': 80000+i}
             for task in config.tasks for i in range(2)]
    pairs = [('final', c) for c in cases] + [('early', c) for c in cases if c['task'] == 'multi-orderings']
    return {'cases': cases, 'shards': [dict(index=i, candidate=p, case_id=c['case_id'])
                                     for i, (p, c) in enumerate(pairs)],
            'trajectories_per_group': 4, 'total_sampled_trajectories': 32,
            'optimizer_steps': 0, 'final_test_unopened': True,
            'scope': 'Training-only checkpoint/task selection; no performance evaluation or adaptive resampling.'}


def source_receipts(config, root=RUNS_PATH):
    if not config.sft_run_id or not config.warmup_run_id:
        raise ValueError('Supply both --sft-run-id (complete Stage 14 pass) and --warmup-run-id (update two)')
    result, histories = {}, {}
    for candidate, name in (('final', config.sft_run_id), ('early', config.warmup_run_id)):
        directory = run_path(root, name)
        cursor = stage14.replay_training(directory)
        previous = json.loads((directory/'config.json').read_text())
        for f in fields(Stage2Config):
            if previous[f.name] != getattr(config, f.name): raise ValueError(f'Model interface changed: {f.name}')
        if previous['max_steps'] != config.max_steps or previous['max_sequence_tokens'] != config.max_sequence_tokens:
            raise ValueError('History/action budget changed')
        training = json.loads((directory/'training.json').read_text())
        if cursor != (training['total_updates'] if candidate == 'final' else 2):
            raise ValueError('Expected a complete SFT pass and its update-two checkpoint')
        receipts = json.loads((directory/'receipts.json').read_text())
        histories[candidate] = (previous, training, receipts)
        result[candidate] = {'run_id': name, 'completed_updates': cursor,
            'manifest_sha256': sha256_file(directory/'manifest.json'),
            'adapter_sha256': sha256_file(directory/'adapter.zip'),
            'adapter_fingerprint': training['after_fingerprint'],
            'requirements_sha256': receipts['initial_adapter']['requirements_sha256']}
    final, early = histories['final'], histories['early']
    if final[0] != early[0] or final[2] != early[2] or final[1]['updates'][:2] != early[1]['updates']:
        raise ValueError('Candidates must belong to the same SFT continuation')
    # Verify the early checkpoint is actually an ancestor, not merely equal rows.
    cursor, seen = config.sft_run_id, set()
    while cursor != config.warmup_run_id:
        if not cursor or cursor in seen: raise ValueError('Early checkpoint is not an ancestor')
        seen.add(cursor)
        directory = run_path(root, cursor)
        verify_artifacts(directory, stage=14, artifact_files=stage14.TRAIN_FILES)
        cursor = json.loads((directory/'training.json').read_text())['previous_checkpoint']
    return result


def strict_group(config, members, fingerprint):
    group = group_record(config, 'new_train', members, fingerprint)
    ratio = 0.
    for episode in members:
        for turn in episode['transitions']:
            g = turn['generation']
            if g.get('policy_scoring') != 'grad-cache-v1': raise ValueError('Wrong GRPO scoring path')
            for old, sampled in zip(g['old_log_probs'], g['sampled_log_probs']):
                delta = old-sampled
                if abs(delta) > 1.: raise ValueError('Behavior ratio drift')
                ratio = max(ratio, abs(math.expm1(delta)))
    if ratio > config.ratio_tolerance: raise ValueError('Behavior ratio drift')
    return {**group, 'max_ratio_deviation': ratio}


def readiness(episodes, groups):
    metrics = summarize([{**e, 'policy': 'sampled', 'case': {**e['case'], 'split': 'eval'}} for e in episodes])
    mixed = sum(not g['zero_variance'] for g in groups)
    return {'optimizer_steps': 0, 'groups': len(groups), 'mixed_return_groups': mixed,
            'all_success_groups': sum(all(r == 1 for r in g['returns']) for g in groups),
            'all_failure_groups': sum(all(r == 0 for r in g['returns']) for g in groups),
            'training_metrics': metrics.get('sampled', {}),
            'returns': [g['returns'] for g in groups],
            'within_group_reward_variances': [g['std']**2 for g in groups],
            'ready_for_nonzero_grpo_signal': bool(mixed),
            'scope': 'Training-only exploration. Mixed returns indicate a possible relative-reward signal, not learning.'}


def validate_shard(config, shard, episodes, groups, fingerprint):
    plan = study_plan(config)
    if shard not in plan['shards']: raise ValueError('Unknown predeclared shard')
    case = next(c for c in plan['cases'] if c['case_id'] == shard['case_id'])
    if len(episodes) != 4 or len(groups) != 1: raise ValueError('Incomplete fixed-budget shard')
    if len({e['episode_id'] for e in episodes}) != 4: raise ValueError('Duplicate trajectories')
    for i, e in enumerate(episodes):
        validate_episode(e)
        if (e['case'] != case or e['max_steps'] != config.max_steps
                or e['policy'] != f"sample-{shard['candidate']}-{shard['index']}-{i}"
                or e['sampling_seed'] != trajectory_seed(config, shard['index'], i)):
            raise ValueError('Training case, candidate or independent RNG changed')
    if groups != [strict_group(config, episodes, fingerprint)]: raise ValueError('Group scores/advantages changed')


def replay_saved_run(directory):
    raw = json.loads((directory/'manifest.json').read_text())
    mode = raw.get('mode')
    if mode not in ('cpu', 'gpu'): raise ValueError('Unknown exploration mode')
    verify_artifacts(directory, stage=15, artifact_files=GPU_FILES if mode == 'gpu' else CPU_FILES)
    config = ExplorationConfig(**json.loads((directory/'config.json').read_text()))
    if json.loads((directory/'sources.json').read_text()) != source_receipts(config, directory.parent):
        raise ValueError('Source checkpoints changed')
    if (json.loads((directory/'plan.json').read_text()) != study_plan(config)
            or (directory/'policy-prompt.txt').read_text() != browser_prompt(config.max_steps)+'\n'):
        raise ValueError('Training-only plan/prompt changed')
    episodes = json.loads((directory/'episodes.json').read_text())
    with zipfile.ZipFile(directory/'images.zip') as archive:
        if archive.testzip() is not None: raise ValueError('Corrupt screenshot archive')
        for e in episodes:
            validate_episode(e)
            if e['case'] not in study_plan(config)['cases']: raise ValueError('Unexpected case; evaluation is prohibited')
            if any(o['image'] not in archive.namelist() for o in e['observations']): raise ValueError('Missing screenshot')
    environment = json.loads((directory/'environment.json').read_text())
    sources = json.loads((directory/'sources.json').read_text())
    if sha256_file(directory/'requirements-browser.txt') != sources['final']['requirements_sha256']:
        raise ValueError('Dependency lock changed')
    if mode == 'cpu':
        cases = study_plan(config)['cases']
        if (len(episodes) != 6 or [e['case'] for e in episodes] != cases
                or any(e['policy'] != 'oracle' or not e['success'] for e in episodes)):
            raise ValueError('Six successful training-only teacher episodes required')
    else:
        shard = environment['shard']; fp = sources[shard['candidate']]['adapter_fingerprint']
        groups = json.loads((directory/'groups.json').read_text())
        validate_shard(config, shard, episodes, groups, fp)
        if (environment['optimizer_steps'] != 0 or environment['before_fingerprint'] != fp
                or environment['after_fingerprint'] != fp or environment['policy_scoring'] != 'grad-cache-v1'):
            raise ValueError('Behavior policy must remain unchanged')
        if readiness(episodes, groups) != json.loads((directory/'readiness.json').read_text()):
            raise ValueError('Readiness report changed')
    if not all(json.loads((directory/'checks.json').read_text()).values()): raise ValueError('Failed verification')
    return len(episodes)


def save_run(run_id, config, source, sources, episodes, assets, checks, environment, extras=None):
    directory = run_path(RUNS_PATH, run_id)
    if (directory/'manifest.json').exists(): raise FileExistsError('Completed run exists')
    directory.mkdir(parents=True, exist_ok=True)
    objects = {'config.json': asdict(config), 'sources.json': sources, 'plan.json': study_plan(config),
               'episodes.json': episodes, 'checks.json': checks, 'environment.json': environment, **(extras or {})}
    for name, value in objects.items():
        if isinstance(value, str): (directory/name).write_text(value)
        else: write_json(directory/name, value)
    (directory/'requirements-browser.txt').write_bytes(Path('/root/requirements-browser.txt').read_bytes())
    (directory/'policy-prompt.txt').write_text(browser_prompt(config.max_steps)+'\n')
    (directory/'images.zip').write_bytes(_images_zip(assets))
    html = stage8.report_html({'episodes': episodes, 'metrics': {}, 'checks': checks}, assets,
        title='Exploration after form SFT', description=study_plan(config)['scope'])
    html = html.replace('Metrics — held-out instances only', 'Evaluation metrics — none collected')
    if extras: html += '<h2>Training-only signal</h2><pre>'+escape(json.dumps(extras['readiness.json'], indent=2))+'</pre>'
    (directory/'report.html').write_text(html)
    files = GPU_FILES if extras else CPU_FILES
    write_manifest(directory, {'mode': environment['mode'], 'protocol': 'browser-json-v1',
                   'source_sha256': source}, stage=15, artifact_files=files)
    replay_saved_run(directory)
    write_json(directory/'progress.json', {'status': 'complete', 'optimizer_steps': 0, 'episodes': len(episodes)})
    return stage14.result_bundle(directory, files, environment=environment,
                                 readiness=extras['readiness.json'] if extras else None)


def prepare_remote(run_id, raw_config, source):
    config = ExplorationConfig(**raw_config); sources = source_receipts(config)
    if sha256_file(Path('/root/requirements-browser.txt')) != sources['final']['requirements_sha256']:
        raise ValueError('Dependency lock changed')
    started = time.monotonic(); assets = {}; episodes = []
    for case in study_plan(config)['cases']:
        e = collect_episode(config, case, 'oracle', assets)
        e['episode_id'] = 'oracle/'+case['case_id']
        if not e['success']: raise RuntimeError('Teacher failed '+case['case_id'])
        stage8.reexecute_episode(e); episodes.append(e)
        print(f"Exploration teacher/replay: {case['case_id']} {e['steps']} actions", flush=True)
    return save_run(run_id, config, source, sources, episodes, assets,
                    {'all_teacher_cases_solvable': True, 'all_teacher_episodes_reexecuted': True,
                     'final_test_unopened': True, 'optimization_not_performed': True},
                    {**stage8.runtime(), 'mode': 'cpu', 'max_actions': 12, 'seconds': time.monotonic()-started})


def probe_remote(run_id, raw_config, source, cache, cpu_run_id, shard_index):
    import torch
    from PIL import Image
    from .browser_policy import BrowserTrainPolicy
    from .stage12 import score_check

    config = ExplorationConfig(**raw_config); sources = source_receipts(config)
    plan = study_plan(config)
    if type(shard_index) is not int or not 0 <= shard_index < 8: raise ValueError('Unknown shard index')
    shard = plan['shards'][shard_index]; case = next(c for c in plan['cases'] if c['case_id'] == shard['case_id'])
    cpu = run_path(RUNS_PATH, cpu_run_id); replay_saved_run(cpu)
    cpu_manifest = verify_artifacts(cpu, stage=15, artifact_files=CPU_FILES)
    if (json.loads((cpu/'config.json').read_text()) != json.loads(json.dumps(asdict(config)))
            or cpu_manifest['source_sha256'] != source or cache['revision'] != config.model_revision
            or sha256_file(Path('/root/requirements-browser.txt')) != sources['final']['requirements_sha256']):
        raise ValueError('Preflight config, code, model or dependency lock changed')
    teacher = next(e for e in json.loads((cpu/'episodes.json').read_text()) if e['case'] == case)
    assets = {}
    with zipfile.ZipFile(cpu/'images.zip') as archive:
        for name in image_paths(teacher['transitions'][0]['messages']) + image_paths(teacher['transitions'][min(2, teacher['steps']-1)]['messages']):
            with Image.open(io.BytesIO(archive.read(name))) as im: assets[name] = im.convert('RGB').copy()
    started = time.monotonic(); episodes = []; groups = []
    try:
        checkpoint = run_path(RUNS_PATH, sources[shard['candidate']]['run_id'])
        with tempfile.TemporaryDirectory() as temporary:
            unpack_adapter(checkpoint/'adapter.zip', Path(temporary)/'adapter')
            actor = BrowserTrainPolicy(config, Path(temporary)/'adapter')
        fp = adapter_fingerprint(actor.model)
        if fp != sources[shard['candidate']]['adapter_fingerprint']: raise RuntimeError('Adapter reload changed')
        smoke = []
        for turn in (teacher['transitions'][0], teacher['transitions'][min(2, teacher['steps']-1)]):
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(config.seed); torch.cuda.manual_seed_all(config.seed)
                decision = actor.sample(turn['messages'], [assets[p] for p in image_paths(turn['messages'])])
            record = {'messages': turn['messages'], 'generation': {k: v for k, v in decision.items() if k != 'text'}}
            scores, check = score_check(actor, record, assets, fp, backward=False)
            smoke.append({'images': len(image_paths(turn['messages'])), **check}); del scores
        print(f"Exploration {shard}: strict GRPO-path scoring smoke passed {smoke}", flush=True)
        for member in range(4):
            seed = trajectory_seed(config, shard_index, member)
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            e = collect_episode(config, case, f"sample-{shard['candidate']}-{shard_index}-{member}", assets, actor=actor.sample)
            e.update(episode_id=f"sample-{shard_index}-{member}/"+case['case_id'], sampling_seed=seed)
            episodes.append(e)
            for turn in e['transitions']:
                scores, check = score_check(actor, turn, assets, fp, backward=False); del scores
            stage8.reexecute_episode(e)
            _write_progress(run_id, config, {'episodes': episodes}, assets, groups)
            print(f"Exploration {shard['candidate']} {case['case_id']} {member+1}/4: return={e['return']} actions={e['steps']} elapsed={time.monotonic()-started:.1f}s", flush=True)
        groups = [strict_group(config, episodes, fp)]
        after = adapter_fingerprint(actor.model)
        if fp != after or any(p.grad is not None for p in actor.model.parameters()):
            raise RuntimeError('Collection changed parameters or accumulated gradients')
        environment = {**stage8.runtime(), 'mode': 'gpu', 'max_actions': 12, 'shard': shard,
            'seconds': time.monotonic()-started, 'gpu': torch.cuda.get_device_name(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'optimizer_steps': 0,
            'before_fingerprint': fp, 'after_fingerprint': after, 'policy_scoring': 'grad-cache-v1',
            'sampling_decoding': actor.rollout_generation.to_dict(), 'scoring_smoke': smoke}
        return save_run(run_id, config, source, sources, episodes, assets,
            {'frozen_behavior_adapter': True, 'exact_assistant_tokens_scored': True,
             'strict_behavior_ratios_passed': True, 'all_sampled_episodes_reexecuted': True,
             'independent_trajectory_rng': True, 'optimization_not_performed': True, 'final_test_unopened': True},
            environment, {'groups.json': groups, 'readiness.json': readiness(episodes, groups),
             'cpu-source.json': {'run_id': cpu_run_id, 'manifest_sha256': sha256_file(cpu/'manifest.json')},
             'model-cache.json': cache, 'chat-template.txt': actor.processor.chat_template})
    except Exception as exc:
        _write_progress(run_id, config, {'episodes': episodes}, assets, groups, error=f'{type(exc).__name__}: {exc}')
        raise

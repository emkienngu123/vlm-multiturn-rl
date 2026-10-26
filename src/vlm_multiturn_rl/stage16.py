"""Checkpointed form GRPO after the short SFT warmup; bounded learning study."""

from dataclasses import asdict, dataclass, fields
from html import escape
import io
import json
import math
from pathlib import Path
import tempfile
import time
import zipfile

from . import stage8, stage12, stage13, stage14, stage15
from .browser import browser_prompt, collect_episode, validate_episode
from .browser_policy import BrowserTrainPolicy
from .grpo import relative_advantages, tensor_checks
from .stage4 import _images_zip
from .stage5 import adapter_fingerprint, unpack_adapter
from .stage10 import trajectory_seed
from .vision import Stage2Config
from .workspace import RUNS_PATH, run_path, sha256_file, verify_artifacts, write_json, write_manifest

COMMON = ('config.json', 'receipts.json', 'plan.json', 'episodes.json', 'images.zip', 'environment.json',
          'checks.json', 'requirements-browser.txt', 'policy-prompt.txt', 'report.html')
CPU_FILES = COMMON
TRAIN_FILES = (*COMMON, 'training.json', 'group.json', 'cpu-source.json', 'model-cache.json', 'chat-template.txt',
               'checkpoint/adapter.zip', 'checkpoint/training-state.pt', 'checkpoint/checkpoint.json')
EVAL_FILES = (*COMMON, 'evaluation.json', 'metrics.json')


@dataclass(frozen=True)
class FormGRPOConfig(stage13.FormDatasetConfig):
    sft_run_id: str = ''
    comparison_run_id: str = ''
    readiness_run_id: str = ''
    group_size: int = 4
    sampling_logp_tolerance: float = .1
    learning_rate: float = 1e-5
    clip_epsilon: float = .2
    ratio_tolerance: float = 1e-4
    kl_beta: float = 0.
    max_rounds: int = 20
    training_seconds: int = 1800

    def __post_init__(self):
        super().__post_init__()
        for name in (self.sft_run_id, self.comparison_run_id, self.readiness_run_id):
            if name: run_path(Path('.'), name)
        fixed = (self.group_size, self.sampling_logp_tolerance, self.learning_rate, self.clip_epsilon,
                 self.ratio_tolerance, self.kl_beta, self.max_rounds, self.training_seconds)
        if fixed != (4, .1, 1e-5, .2, 1e-4, 0., 20, 1800):
            raise ValueError('This study fixes four samples, lr=1e-5, clip=.2, strict ratios, and a 20-round/1800-second cap')


def plan(config):
    train = [dict(case_id=f'forms16-train-{i:03d}', task='multi-orderings', split='train', seed=80000+i)
             for i in range(10)]
    # Smoke on the previously mixed training state, then nine other states twice.
    order = [0, 0, *range(1, 10), *range(1, 10)]
    return {'train': train, 'dev': stage13.pools(config)['dev'],
            'round_case_ids': [train[i]['case_id'] for i in order],
            'first_sampling_group_index': 8, 'max_rounds': 20, 'training_seconds': 1800,
            'final_test_unopened': True, 'reward': 'binary episode success, no shaping or format bonus'}


def source_receipts(config, root=RUNS_PATH):
    interface = {f.name: getattr(config, f.name) for f in fields(Stage2Config)}
    explorer = stage15.ExplorationConfig(**interface, sft_run_id=config.comparison_run_id,
                                        warmup_run_id=config.sft_run_id)
    sources = stage15.source_receipts(explorer, root)
    if not config.readiness_run_id: raise ValueError('Supply the completed mixed Stage 15 group')
    ready = run_path(root, config.readiness_run_id); stage15.replay_saved_run(ready)
    env = json.loads((ready/'environment.json').read_text())
    groups = json.loads((ready/'groups.json').read_text())
    if (env['shard']['candidate'] != 'early' or env['shard']['case_id'] != 'multi-orderings-explore-train-000'
            or len(groups) != 1 or groups[0]['zero_variance']
            or groups[0]['behavior_fingerprint'] != sources['early']['adapter_fingerprint']):
        raise ValueError('Readiness must establish a mixed group for the selected early form policy')
    if json.loads((ready/'config.json').read_text()) != json.loads(json.dumps(asdict(explorer))):
        raise ValueError('Readiness model/decoding interface differs')
    return {**sources, 'readiness': {'run_id': ready.name, 'manifest_sha256': sha256_file(ready/'manifest.json')}}


def validate_history(config, rows, initial_fingerprint):
    if not 1 <= len(rows) <= config.max_rounds: raise ValueError('Invalid round cursor')
    fp, steps = initial_fingerprint, 0
    for index, row in enumerate(rows):
        group = row['group']; update = row['update']
        stats = relative_advantages(group['returns'])
        if any(group[k] != v for k, v in stats.items()): raise ValueError('Historical advantages differ')
        if (row['round'] != index+1 or group['case_id'] != plan(config)['round_case_ids'][index]
                or group['behavior_fingerprint'] != fp
                or group['trajectory_seeds'] != [trajectory_seed(config, 8+index, i) for i in range(4)]):
            raise ValueError('Historical round, case, behavior or RNG differs')
        if (len(group['returns']) != 4 or any(r not in (0., 1.) for r in group['returns'])
                or group['max_ratio_deviation'] > config.ratio_tolerance
                or update['optimizer_step'] != (not group['zero_variance'])
                or not math.isfinite(row['seconds']) or row['seconds'] <= 0):
            raise ValueError('Invalid return, ratio, skip or timing')
        if (update['max_ratio_deviation'] > config.ratio_tolerance
                or not all(math.isfinite(update[k]) for k in ('loss', 'gradient_norm', 'clipped_fraction'))):
            raise ValueError('Invalid optimization metrics')
        if update['optimizer_step']:
            if update['gradient_norm'] <= 0 or update['after_fingerprint'] == fp: raise ValueError('Missing parameter update')
            steps += 1
        elif update['after_fingerprint'] != fp or update['gradient_norm'] != 0:
            raise ValueError('Zero-variance group changed weights')
        if row['optimizer_steps'] != steps: raise ValueError('Optimizer counter differs from attempted rounds')
        fp = update['after_fingerprint']
    return fp, steps


def replay_saved_run(directory):
    manifest = json.loads((directory/'manifest.json').read_text()); kind = manifest.get('kind')
    files = {'form-grpo-preflight': CPU_FILES, 'form-grpo-checkpoint': TRAIN_FILES, 'form-grpo-dev': EVAL_FILES}.get(kind)
    if files is None: raise ValueError('Unknown form GRPO bundle')
    verify_artifacts(directory, stage=16, artifact_files=files)
    config = FormGRPOConfig(**json.loads((directory/'config.json').read_text()))
    receipts = json.loads((directory/'receipts.json').read_text())
    if receipts != source_receipts(config, directory.parent): raise ValueError('Source receipts changed')
    if (json.loads((directory/'plan.json').read_text()) != plan(config)
            or (directory/'policy-prompt.txt').read_text() != browser_prompt(12)+'\n'
            or sha256_file(directory/'requirements-browser.txt') != receipts['early']['requirements_sha256']):
        raise ValueError('Plan, prompt or lock changed')
    episodes = json.loads((directory/'episodes.json').read_text())
    with zipfile.ZipFile(directory/'images.zip') as archive:
        if archive.testzip() is not None: raise ValueError('Screenshot archive failed CRC')
        for e in episodes:
            validate_episode(e)
            if e['max_steps'] != 12 or any(o['image'] not in archive.namelist() for o in e['observations']):
                raise ValueError('Missing screenshot or changed action cap')
    if not all(json.loads((directory/'checks.json').read_text()).values()): raise ValueError('Failed check')
    if kind == 'form-grpo-preflight':
        if ([e['case'] for e in episodes] != plan(config)['train']
                or any(e['policy'] != 'oracle' or not e['success'] for e in episodes)):
            raise ValueError('Every training state requires a successful teacher preflight')
        return len(episodes)
    if kind == 'form-grpo-dev':
        evaluation = json.loads((directory/'evaluation.json').read_text())
        offset = evaluation['offset']
        if offset not in (0, 8) or [e['case'] for e in episodes] != plan(config)['dev'][offset:offset+8]:
            raise ValueError('Unexpected development pool; final test is prohibited')
        if any(e['policy'] != evaluation['policy'] for e in episodes): raise ValueError('Mixed evaluation policies')
        if stage14.development_metrics(episodes) != json.loads((directory/'metrics.json').read_text()):
            raise ValueError('Development metrics changed')
        return len(episodes)
    training = json.loads((directory/'training.json').read_text()); rows = training['history']
    fp, steps = validate_history(config, rows, receipts['early']['adapter_fingerprint'])
    row = rows[-1]; group = json.loads((directory/'group.json').read_text())
    case = next(c for c in plan(config)['train'] if c['case_id'] == plan(config)['round_case_ids'][len(rows)-1])
    if (len(episodes) != 4 or any(e['case'] != case for e in episodes)
            or [e['sampling_seed'] for e in episodes] != row['group']['trajectory_seeds']
            or len({e['episode_id'] for e in episodes}) != 4
            or group != stage15.strict_group(config, episodes, row['group']['behavior_fingerprint'])
            or row['group'] != group): raise ValueError('Saved group does not match actual trajectories')
    cp = json.loads((directory/'checkpoint/checkpoint.json').read_text())
    if (cp['round'] != len(rows) or cp['optimizer_steps'] != steps or cp['adapter_fingerprint'] != fp
            or any(sha256_file(directory/'checkpoint'/n) != h for n, h in cp['sha256'].items())
            or not all(training['reload'][k] for k in ('adapter_matches', 'adam_matches', 'rng_matches'))
            or training['reload']['max_logp_delta'] > 1e-5): raise ValueError('Checkpoint/reload proof differs')
    previous = training['previous']
    if len(rows) == 1:
        if previous is not None: raise ValueError('Unexpected initial parent')
    else:
        parent = run_path(directory.parent, previous['run_id'])
        pm = verify_artifacts(parent, stage=16, artifact_files=TRAIN_FILES)
        if (previous['manifest_sha256'] != sha256_file(parent/'manifest.json')
                or json.loads((parent/'training.json').read_text())['history'] != rows[:-1]):
            raise ValueError('Continuation chain changed')
        environment = json.loads((directory/'environment.json').read_text())
        execution = environment.get('execution')
        from .stage18 import validate_source_transition, validate_execution
        if execution:
            validate_execution(execution, directory.parent)
            if ('H100' not in environment['gpu'] or environment['activation_gpu_budget_bytes'] != 40*1024**3):
                raise ValueError('H100 execution profile differs')
        parent_execution = json.loads((parent/'environment.json').read_text()).get('execution')
        validate_source_transition(execution, parent.name, pm['source_sha256'],
                                   manifest['source_sha256'], parent_execution)
    return len(rows)


def save_bundle(directory, config, receipts, source, kind, episodes, assets, environment, checks, extra=None):
    directory.mkdir(parents=True, exist_ok=True)
    objects = {'config.json': asdict(config), 'receipts.json': receipts, 'plan.json': plan(config),
               'episodes.json': episodes, 'environment.json': environment, 'checks.json': checks, **(extra or {})}
    for name, value in objects.items():
        if isinstance(value, str): (directory/name).write_text(value)
        else: write_json(directory/name, value)
    (directory/'images.zip').write_bytes(_images_zip(assets))
    (directory/'requirements-browser.txt').write_bytes(Path('/root/requirements-browser.txt').read_bytes())
    (directory/'policy-prompt.txt').write_text(browser_prompt(12)+'\n')
    html = stage8.report_html({'episodes': episodes, 'metrics': objects.get('metrics.json', {}), 'checks': checks}, assets,
        title='Form GRPO after short SFT', description='Bounded training and development measurements. Final test remains closed.')
    if kind != 'form-grpo-dev': html = html.replace('Metrics — held-out instances only', 'Evaluation metrics — none in this training shard')
    if kind == 'form-grpo-checkpoint': html += '<h2>Training</h2><pre>'+escape(json.dumps(objects['training.json'], indent=2))+'</pre>'
    (directory/'report.html').write_text(html)
    files = {'form-grpo-preflight': CPU_FILES, 'form-grpo-checkpoint': TRAIN_FILES, 'form-grpo-dev': EVAL_FILES}[kind]
    write_manifest(directory, {'kind': kind, 'source_sha256': source}, stage=16, artifact_files=files)
    try: replay_saved_run(directory)
    except Exception:
        (directory/'manifest.json').unlink(); raise
    write_json(directory/'progress.json', {'status': 'complete', 'kind': kind})
    return stage14.result_bundle(directory, files, environment=environment,
        training=objects.get('training.json'), metrics=objects.get('metrics.json'))


def prepare_remote(run_id, raw, source):
    config = FormGRPOConfig(**raw); receipts = source_receipts(config)
    started = time.monotonic(); assets = {}; episodes = []
    for case in plan(config)['train']:
        e = collect_episode(config, case, 'oracle', assets); e['episode_id'] = 'oracle/'+case['case_id']
        if not e['success']: raise RuntimeError('Training oracle failed '+case['case_id'])
        stage8.reexecute_episode(e); episodes.append(e)
        print(f"Form GRPO teacher {case['case_id']}: {e['steps']} actions", flush=True)
    return save_bundle(run_path(RUNS_PATH, run_id), config, receipts, source, 'form-grpo-preflight', episodes, assets,
        {**stage8.runtime(), 'seconds': time.monotonic()-started, 'max_actions': 12},
        {'all_training_states_solvable': True, 'teacher_actions_reexecuted': True, 'final_test_unopened': True})


def train_remote(run_id, raw, source, cache, cpu_run_id, previous_id, *, execution=None):
    import torch
    from PIL import Image

    config = FormGRPOConfig(**raw); receipts = source_receipts(config)
    cpu = run_path(RUNS_PATH, cpu_run_id); replay_saved_run(cpu)
    if (json.loads((cpu/'config.json').read_text()) != json.loads(json.dumps(asdict(config)))
            or json.loads((cpu/'manifest.json').read_text())['source_sha256'] != source
            or cache['revision'] != config.model_revision
            or sha256_file(Path('/root/requirements-browser.txt')) != receipts['early']['requirements_sha256']):
        raise ValueError('Preflight config/code, cache or lock differs')
    previous = run_path(RUNS_PATH, previous_id) if previous_id else None
    rows = []
    if previous:
        cursor = replay_saved_run(previous)
        if json.loads((previous/'config.json').read_text()) != json.loads(json.dumps(asdict(config))):
            raise ValueError('Resume config changed')
        from .stage18 import validate_source_transition, validate_execution
        if execution: validate_execution(execution)
        validate_source_transition(execution, previous.name,
            json.loads((previous/'manifest.json').read_text())['source_sha256'], source,
            json.loads((previous/'environment.json').read_text()).get('execution'))
        rows = json.loads((previous/'training.json').read_text())['history']
    else: cursor = 0
    if execution and (not previous or 'H100' not in torch.cuda.get_device_name()):
        raise ValueError('H100 continuation needs its parent and the requested GPU')
    if cursor >= config.max_rounds: raise ValueError('Round cap reached')
    number = cursor+1; case = next(c for c in plan(config)['train'] if c['case_id'] == plan(config)['round_case_ids'][cursor])
    teacher = next(e for e in json.loads((cpu/'episodes.json').read_text()) if e['case'] == case)
    assets = {}
    with zipfile.ZipFile(cpu/'images.zip') as archive:
        for o in teacher['observations']:
            with Image.open(io.BytesIO(archive.read(o['image']))) as im: assets[o['image']] = im.convert('RGB').copy()
    directory = run_path(RUNS_PATH, run_id); directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); members = []
    def progress(error=None):
        write_json(directory/'progress.json', {'status': 'incomplete', 'round': number, 'error': error})
        write_json(directory/'partial-episodes.json', members)
        (directory/'partial-images.zip').write_bytes(_images_zip(assets))
    try:
        adapter = previous/'checkpoint/adapter.zip' if previous else run_path(RUNS_PATH, config.sft_run_id)/'adapter.zip'
        with tempfile.TemporaryDirectory() as temporary:
            unpack_adapter(adapter, Path(temporary)/'adapter'); actor = BrowserTrainPolicy(config, Path(temporary)/'adapter')
        if previous:
            optimizer, restore = stage12.reload_checkpoint(previous/'checkpoint', actor)
            smoke = []; tensor = {}
            if execution:
                actor.activation_gpu_budget_bytes = execution['activation_gpu_budget_bytes']
                if previous_id == execution['origin_run_id']:
                    from .gpu_benchmark import activation_check
                    tensor = {**tensor_checks(), **activation_check(actor.model, actor.activation_gpu_budget_bytes)}
                    smoke = stage12.gradient_smoke(actor, teacher, assets)
        else:
            if adapter_fingerprint(actor.model) != receipts['early']['adapter_fingerprint']: raise RuntimeError('Initial adapter differs')
            tensor = {**tensor_checks(), **stage12.activation_memory_checks()}
            smoke = stage12.gradient_smoke(actor, teacher, assets)
            optimizer = torch.optim.AdamW([p for p in actor.model.parameters() if p.requires_grad], lr=config.learning_rate, weight_decay=0.)
            restore = None
        fp = adapter_fingerprint(actor.model)
        for member in range(4):
            seed = trajectory_seed(config, 8+cursor, member)
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            policy = f'grpo-round-{number:03d}-sample-{member}'
            e = collect_episode(config, case, policy, assets, actor=actor.sample)
            e.update(episode_id=policy+'/'+case['case_id'], sampling_seed=seed)
            members.append(e)
            for turn in e['transitions']:
                scores, _ = stage12.score_check(actor, turn, assets, fp, backward=False); del scores
            stage8.reexecute_episode(e); progress()
            print(f"Form GRPO round {number} sample {member+1}/4: return={e['return']} actions={e['steps']}", flush=True)
        group = stage15.strict_group(config, members, fp)
        if any(p.grad is not None for p in actor.model.parameters()): raise RuntimeError('Collection accumulated gradients')
        timing = {} if execution else None
        update = stage12.update_group(actor, optimizer, members, group, assets, timing=timing)
        steps = (rows[-1]['optimizer_steps'] if rows else 0)+int(update['optimizer_step'])
        # Save and reload the actual updated policy, Adam and RNG, then compare scores.
        probe = members[0]['transitions'][0]
        before = actor.score_grad(probe, assets, backward=False).detach().cpu()
        stage12.save_checkpoint(directory/'checkpoint', actor, optimizer, number, steps)
        del optimizer
        optimizer, reload = stage12.reload_checkpoint(directory/'checkpoint', actor)
        after = actor.score_grad(probe, assets, backward=False).detach().cpu()
        reload['max_logp_delta'] = float((before-after).abs().max())
        if reload['max_logp_delta'] > 1e-5: raise RuntimeError('Saved adapter changed scores')
        row = {'round': number, 'group': group, 'update': update, 'optimizer_steps': steps, 'seconds': time.monotonic()-started}
        rows = [*rows, row]; validate_history(config, rows, receipts['early']['adapter_fingerprint'])
        training = {'history': rows, 'previous': {'run_id': previous_id, 'manifest_sha256': sha256_file(previous/'manifest.json')} if previous else None,
                    'restore': restore, 'reload': reload, 'gradient_smoke': smoke, 'tensor_checks': tensor,
                    'normalization': 'mean of per-trajectory assistant-token means', 'kl_beta': 0.}
        if execution: training['timing'] = timing
        environment = {**stage8.runtime(), 'seconds': row['seconds'], 'max_actions': 12,
            'gpu': torch.cuda.get_device_name(), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'policy_scoring': 'grad-cache-v1', 'sampling_decoding': actor.rollout_generation.to_dict(),
            'activation_memory': 'exact saved-tensor CPU offload for backward; collection tensors discarded'}
        if execution:
            environment.update(execution=execution, activation_gpu_budget_bytes=actor.activation_gpu_budget_bytes,
                activation_memory='bounded GPU retention; exact CPU offload for overflow; collection tensors discarded',
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
        print(f"Form GRPO round {number}: returns={group['returns']}, optimizer_steps={steps}, update={update}, seconds={row['seconds']:.1f}", flush=True)
        return save_bundle(directory, config, receipts, source, 'form-grpo-checkpoint', members, assets, environment,
            {'assistant_only_scores': True, 'strict_ratios_passed': True, 'fresh_group': True,
             'model_actions_reexecuted': True, 'adapter_adam_rng_reloaded': True, 'final_test_unopened': True},
            {'training.json': training, 'group.json': group, 'cpu-source.json': {'run_id': cpu_run_id, 'manifest_sha256': sha256_file(cpu/'manifest.json')},
             'model-cache.json': cache, 'chat-template.txt': actor.processor.chat_template})
    except Exception as exc:
        progress(f'{type(exc).__name__}: {exc}'); raise


def evaluate_remote(run_id, raw, source, cache, checkpoint_id, offset):
    import torch

    config = FormGRPOConfig(**raw); receipts = source_receipts(config)
    if offset not in (0, 8) or cache['revision'] != config.model_revision: raise ValueError('Unexpected dev offset/model')
    if sha256_file(Path('/root/requirements-browser.txt')) != receipts['early']['requirements_sha256']: raise ValueError('Dependency lock changed')
    if checkpoint_id:
        cp = run_path(RUNS_PATH, checkpoint_id); cursor = replay_saved_run(cp)
        if json.loads((cp/'config.json').read_text()) != json.loads(json.dumps(asdict(config))): raise ValueError('Checkpoint config differs')
        adapter = cp/'checkpoint/adapter.zip'; policy = f'grpo-round-{cursor:03d}'
        receipt = {'run_id': checkpoint_id, 'manifest_sha256': sha256_file(cp/'manifest.json')}
        expected_fp = json.loads((cp/'checkpoint/checkpoint.json').read_text())['adapter_fingerprint']
    else:
        adapter = run_path(RUNS_PATH, config.sft_run_id)/'adapter.zip'; policy = 'early-sft'; receipt = None
        expected_fp = receipts['early']['adapter_fingerprint']
    started = time.monotonic(); actor = stage14.load_actor(config, adapter, trainable=False)
    if adapter_fingerprint(actor.model) != expected_fp: raise RuntimeError('Evaluation loaded the wrong adapter')
    assets = {}; episodes = []
    for case in plan(config)['dev'][offset:offset+8]:
        e = collect_episode(config, case, policy, assets, actor=actor.act)
        stage8.reexecute_episode(e); episodes.append(e)
        print(f"Form GRPO dev {policy} {case['case_id']}: success={e['success']} actions={e['steps']}", flush=True)
    if adapter_fingerprint(actor.model) != expected_fp: raise RuntimeError('Evaluation changed weights')
    metrics = stage14.development_metrics(episodes)
    return save_bundle(run_path(RUNS_PATH, run_id), config, receipts, source, 'form-grpo-dev', episodes, assets,
        {**stage8.runtime(), 'max_actions': 12, 'gpu': torch.cuda.get_device_name(), 'seconds': time.monotonic()-started,
         'peak_allocated_bytes': torch.cuda.max_memory_allocated()},
        {'development_only': True, 'actual_actions_reexecuted': True, 'adapter_unchanged': True, 'final_test_unopened': True},
        {'metrics.json': metrics, 'evaluation.json': {'policy': policy, 'checkpoint': receipt, 'offset': offset,
         'adapter_fingerprint': expected_fp, 'decoding': actor.generation.to_dict()}})

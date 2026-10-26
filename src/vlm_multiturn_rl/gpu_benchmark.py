"""Fixed-work GPU/offload benchmark; disposable updates, never learning evidence.

Replay a previously collected mixed group so every device processes identical
histories and exact token IDs. Rebase old scores on each device: BF16 kernels
can differ across architectures. This is a timing experiment, not fresh GRPO.
"""
import base64
from copy import deepcopy
import gc
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile

from . import stage8, stage12, stage16
from .browser_policy import BrowserTrainPolicy, saved_activation_memory
from .stage5 import adapter_fingerprint, unpack_adapter
from .trajectories import image_paths
from .workspace import RUNS_PATH, run_path, sha256_file, verify_artifacts, write_json, write_manifest

FILES = ('benchmark.json', 'source.json')
GPU_RATES = {'L40S': .000542, 'H100': .001097}


def verify_source(directory):
    stage16.replay_saved_run(directory)
    history = json.loads((directory/'training.json').read_text())['history']
    group = json.loads((directory/'group.json').read_text())
    if len(history) != 1 or group['zero_variance']:
        raise ValueError('Use the first mixed group, with the original SFT behavior adapter')
    return group


def activation_check(model, budget):
    """Compare exact saved values and gradients under both memory policies."""
    import torch

    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(123)
        x = torch.randn(512, 256, device='cuda', requires_grad=True)
        layer = torch.nn.Linear(256, 16, bias=False, device='cuda')
        expected = torch.autograd.grad(layer(x).sin().square().mean(), layer.weight)[0]
        stats = {}
        with saved_activation_memory(layer, backward=True, gpu_budget_bytes=budget, stats=stats):
            loss = layer(x).sin().square().mean()
        actual = torch.autograd.grad(loss, layer.weight)[0]
        delta = float((actual-expected).abs().max())
        if delta > 1e-6: raise RuntimeError('Saved activation policy changed gradients')
    return {'max_gradient_delta': delta, **stats}


def benchmark_remote(run_id, source_id, requested_gpu, budget_gib, source):
    import torch
    from PIL import Image

    started = time.monotonic()
    directory = run_path(RUNS_PATH, run_id); directory.mkdir(parents=True, exist_ok=False)
    source_dir = run_path(RUNS_PATH, source_id)
    group = verify_source(source_dir)
    config = stage16.FormGRPOConfig(**json.loads((source_dir/'config.json').read_text()))
    episodes = json.loads((source_dir/'episodes.json').read_text())
    gpu = torch.cuda.get_device_name()
    if requested_gpu not in GPU_RATES or requested_gpu not in gpu:
        raise RuntimeError(f'Expected {requested_gpu}, received {gpu}')
    if budget_gib != {'L40S': 16, 'H100': 40}[requested_gpu]: raise ValueError('Unexpected activation budget')
    if sha256_file(Path('/root/requirements-browser.txt')) != sha256_file(source_dir/'requirements-browser.txt'):
        raise ValueError('Dependency lock changed')
    receipt = {'run_id': source_id, 'manifest_sha256': sha256_file(source_dir/'manifest.json'),
        'episodes_sha256': sha256_file(source_dir/'episodes.json'), 'images_sha256': sha256_file(source_dir/'images.zip'),
        'adapter_run_id': config.sft_run_id,
        'adapter_sha256': sha256_file(run_path(RUNS_PATH, config.sft_run_id)/'adapter.zip')}
    write_json(directory/'source.json', receipt)
    result = {'status': 'incomplete', 'requested_gpu': requested_gpu, 'gpu': gpu, 'source': receipt,
        'scope': 'fixed-history forced-token replay and disposable update; not fresh on-policy training',
        'final_test_unopened': True, 'modes': []}
    def progress(): write_json(directory/'progress.json', result)
    progress()
    assets = {}
    with zipfile.ZipFile(source_dir/'images.zip') as archive:
        for e in episodes:
            for obs in e['observations']:
                with Image.open(io.BytesIO(archive.read(obs['image']))) as im:
                    assets[obs['image']] = im.convert('RGB').copy()
    t = time.monotonic()
    for e in episodes: stage8.reexecute_episode(e)
    result['browser_reexecution_seconds'] = time.monotonic()-t
    from .browser import close_browser_runtime
    close_browser_runtime()
    t = time.monotonic()
    with tempfile.TemporaryDirectory() as temporary:
        unpack_adapter(run_path(RUNS_PATH, config.sft_run_id)/'adapter.zip', Path(temporary)/'adapter')
        actor = BrowserTrainPolicy(config, Path(temporary)/'adapter')
    torch.cuda.synchronize()
    result['model_load_seconds'] = time.monotonic()-t
    fp = adapter_fingerprint(actor.model)
    if fp != group['behavior_fingerprint']: raise RuntimeError('Wrong behavior adapter')
    parameters = [p for p in actor.model.parameters() if p.requires_grad]
    initial = [p.detach().cpu().clone() for p in parameters]
    turns = [turn for e in episodes for turn in e['transitions']]
    result.update(actions=len(turns), assistant_tokens=sum(len(t['generation']['generated_ids']) for t in turns),
                  returns=group['returns'], behavior_fingerprint=fp,
                  torch_version=torch.__version__, gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
    # One fresh sample on short and long recorded histories validates the real
    # sampler against the scorer on this GPU. These tokens are not optimized.
    result['fresh_sampling_checks'] = []
    for index in (0, max(range(len(turns)), key=lambda i: len(image_paths(turns[i]['messages'])))):
        turn = turns[index]
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(4242+index); torch.cuda.manual_seed_all(4242+index)
            sample = actor.sample(turn['messages'], [assets[p] for p in image_paths(turn['messages'])])
        check = {'messages': turn['messages'], 'generation': sample}
        scores, stats = stage12.score_check(actor, check, assets, fp, backward=False)
        result['fresh_sampling_checks'].append({'images': len(image_paths(turn['messages'])), **stats})
        del scores
    # Per-device BF16 behavior scores, preserving the historical ones in source.
    torch.cuda.synchronize(); t = time.monotonic(); device_delta = 0.
    for turn in turns:
        scores = actor.score_grad(turn, assets, backward=False).detach().cpu()
        old = torch.tensor(turn['generation']['sampled_log_probs'])
        device_delta = max(device_delta, float((scores-old).abs().max()))
        turn['generation']['old_log_probs'] = scores.tolist()
        turn['generation']['behavior_fingerprint'] = fp
    torch.cuda.synchronize()
    result['forced_replay_seconds'] = time.monotonic()-t
    result['historical_device_max_logp_delta'] = device_delta
    print(f'GPU benchmark {gpu}: {len(turns)} actions, device logp delta={device_delta}', flush=True)
    reference_grad = None
    try:
        for name, budget in (('cpu-offload', 0), ('budgeted', budget_gib*1024**3)):
            with torch.no_grad():
                for p, value in zip(parameters, initial): p.copy_(value)
            actor.model.zero_grad(set_to_none=True)
            if adapter_fingerprint(actor.model) != fp: raise RuntimeError('Benchmark did not reset initial weights')
            actor.activation_gpu_budget_bytes = budget
            checks = activation_check(actor.model, budget)
            optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=0.)
            captured = {}
            # Capture the real accumulated, clipped LoRA gradient immediately
            # before Adam; compare the entire vector across offload modes.
            def capture(opt, args, kwargs):
                captured['gradient'] = torch.cat([p.grad.detach().flatten().cpu() for p in parameters])
            handle = optimizer.register_step_pre_hook(capture)
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            timing = {}; t = time.monotonic()
            update = stage12.update_group(actor, optimizer, episodes, group, assets, timing=timing)
            torch.cuda.synchronize(); seconds = time.monotonic()-t
            handle.remove()
            if reference_grad is None:
                reference_grad = captured['gradient']; grad_delta = 0.
                reference_after = update['after_fingerprint']
            else:
                grad_delta = float((reference_grad-captured['gradient']).abs().max())
                torch.testing.assert_close(reference_grad, captured['gradient'], rtol=1e-5, atol=1e-6)
                if update['after_fingerprint'] != reference_after:
                    raise RuntimeError('Offload mode changed the updated adapter')
            row = {'mode': name, 'gpu_budget_gib': budget/1024**3, 'seconds': seconds,
                'timing': timing, 'update': update, 'activation_check': checks,
                'max_clipped_gradient_delta': grad_delta,
                'peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3,
                'peak_reserved_gib': torch.cuda.max_memory_reserved()/1024**3,
                'gpu_only_update_cost_usd': seconds*GPU_RATES[requested_gpu]}
            result['modes'].append(row); progress()
            print(f'GPU benchmark {gpu} {name}: {json.dumps(row)}', flush=True)
            del optimizer, captured
        result['status'] = 'complete'
    except torch.cuda.OutOfMemoryError:
        result['status'] = 'budgeted_oom'
        result['error'] = 'Activation cap is not a total VRAM cap; baseline retained. No silent retry.'
        actor.model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    result['seconds'] = time.monotonic()-started
    result['gpu_only_function_cost_estimate_usd'] = result['seconds']*GPU_RATES[requested_gpu]
    result['pricing'] = {'url': 'https://modal.com/pricing', 'checked_date': '2026-10-19',
        'gpu_usd_per_second': GPU_RATES[requested_gpu],
        'excludes': 'CPU, host memory, queue/image startup, volume storage, and idle shutdown; not an invoice'}
    write_json(directory/'benchmark.json', result)
    write_manifest(directory, {'kind': 'gpu-offload-benchmark', 'source_sha256': source}, stage=17, artifact_files=FILES)
    return {'run_id': run_id, 'result': result,
            'files': {name: base64.b64encode((directory/name).read_bytes()).decode() for name in (*FILES, 'manifest.json')}}


def save_result(bundle, root):
    directory = run_path(Path(root), bundle['run_id']); directory.mkdir(parents=True, exist_ok=False)
    if set(bundle['files']) != {*FILES, 'manifest.json'}: raise ValueError('Unexpected benchmark files')
    for name, value in bundle['files'].items(): (directory/name).write_bytes(base64.b64decode(value, validate=True))
    verify_artifacts(directory, stage=17, artifact_files=FILES)
    return directory


def comparison_report(paths, output):
    """Standard library report; report uncertainty from one group explicitly."""
    from html import escape
    rows = [json.loads((p/'benchmark.json').read_text()) for p in paths]
    if len(rows) != 2 or rows[0]['source'] != rows[1]['source']: raise ValueError('Unpaired benchmark sources')
    if {r['requested_gpu'] for r in rows} != {'L40S', 'H100'}: raise ValueError('Missing GPU')
    table = ''
    for r in rows:
        for m in r['modes']:
            table += '<tr>'+''.join(f'<td>{escape(str(x))}</td>' for x in
                (r['gpu'], m['mode'], f"{m['seconds']:.2f}", f"{m['timing']['scoring_forward_seconds']:.2f}",
                 f"{m['timing']['backward_seconds']:.2f}", f"{m['peak_allocated_gib']:.2f}",
                 f"${m['gpu_only_update_cost_usd']:.4f}", m['update']['max_ratio_deviation']))+'</tr>'
    html = '<!doctype html><meta charset="utf-8"><title>GPU/offload benchmark</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto}td,th{padding:10px;border:1px solid #ddd}table{border-collapse:collapse}</style>'
    html += '<h1>Fixed-group GPU and activation offloading benchmark</h1><p>Identical recorded histories, screenshots, token IDs, returns, initial adapter, and fresh Adam state. Disposable updates; no learning or generalization claim.</p>'
    html += '<p>BF16 scores are measured on each device before timing; strict unchanged-policy ratios are checked locally. CPU and budgeted modes must produce matching full LoRA gradients and identical updated adapter fingerprints on the same device.</p>'
    html += '<table><tr><th>GPU</th><th>Mode</th><th>Update seconds</th><th>Forward seconds</th><th>Backward seconds</th><th>Peak allocated GiB</th><th>GPU-only update cost</th><th>Ratio drift</th></tr>'+table+'</table>'
    for p, r in zip(paths, rows):
        html += f'<p><a href="{escape(p.name)}/benchmark.json">{escape(r["gpu"])}</a>: load {r["model_load_seconds"]:.1f}s; forced replay {r["forced_replay_seconds"]:.1f}s; browser action replay {r["browser_reexecution_seconds"]:.1f}s. Status: {escape(r["status"])}.</p>'
    html += '<p>One group per configuration, baseline first: preliminary timings include warmup/order effects. Forced replay approximates generation work but does not sample a fresh trajectory. Browser replay differs from live interaction. Update cost excludes host resources and startup; see pricing and per-function estimates in JSON.</p>'
    output.write_text(html)
    return rows

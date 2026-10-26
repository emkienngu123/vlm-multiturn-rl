"""Export diagnostics from the verified Stage 16 summary on a small CPU image."""
import base64
import io
import json
from pathlib import Path

import modal

image = modal.Image.debian_slim(python_version='3.12').pip_install(
    'contourpy==1.4.0', 'cycler==0.12.1', 'fonttools==4.67.0', 'kiwisolver==1.5.1',
    'matplotlib==3.11.2', 'numpy==2.5.4', 'packaging==26.3', 'pillow==13.0.0',
    'pyparsing==3.3.3', 'python-dateutil==2.9.0.post0', 'six==1.17.0')
app = modal.App('vlm-form-grpo-diagnostics')


@app.function(image=image, cpu=1, memory=2048, timeout=180)
def render(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = summary['rounds']; x = [r['round'] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    for r in rows:
        for j, reward in enumerate(r['group']['returns']):
            axes[0].scatter(r['round']+(j-1.5)*.11, reward, s=22,
                            color='#16a34a' if reward else '#94a3b8', alpha=.8)
    axes[0].plot(x, [r['group']['mean'] for r in rows], marker='D', color='#2563eb', label='Four-trajectory mean')
    axes[0].set(xlabel='Attempted round (training states vary)', ylabel='Binary training return',
                title='Training exploration, not held-out success', ylim=(-.08, 1.08), xticks=x)
    axes[0].legend(fontsize=8, loc='center right')
    axes[1].plot(x, [r['update']['gradient_norm'] for r in rows], marker='o', color='#d97706')
    axes[1].axhline(1., color='#64748b', ls='--', lw=1., label='Gradient clipping threshold')
    axes[1].set(xlabel='Attempted round (zero = skipped group)', ylabel='LoRA norm before clipping', title='Actual optimization')
    axes[1].legend(fontsize=8)
    phases = summary['phases']; names = ['before']
    if 'smoke' in phases and phases['smoke']['adapter_fingerprint'] != phases['after']['adapter_fingerprint']:
        names.append('smoke')
    names += ['after', 'full-sft-reference']
    labels = {'before': 'Early SFT', 'smoke': 'After smoke', 'after': 'After GRPO', 'full-sft-reference': 'Full SFT'}
    counts = [phases[n]['successes'] for n in names]
    bars = axes[2].bar([labels[n] for n in names], counts, color=['#94a3b8']+['#2563eb']*(len(names)-2)+['#16a34a'])
    axes[2].bar_label(bars, labels=[f'{n}/16' for n in counts], padding=3)
    axes[2].set(ylabel='Successful development episodes', title='Paired greedy development cases', ylim=(0, 18), yticks=[0, 4, 8, 12, 16])
    axes[2].tick_params(axis='x', labelrotation=15)
    for ax in axes: ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.suptitle(f"{summary['attempted_rounds']} groups / {summary['optimizer_steps']} optimizer steps / "
                 f"{summary['skipped_groups']} skips — final test unopened", fontsize=13)
    outputs = {}
    for fmt in ('png', 'svg'):
        stream = io.BytesIO(); fig.savefig(stream, format=fmt, dpi=160)
        outputs[fmt] = base64.b64encode(stream.getvalue()).decode()
    plt.close(fig); return outputs


@app.local_entrypoint()
def main(workflow: str):
    path = Path(workflow); summary = json.loads(path.with_name(path.stem+'-summary.json').read_text())
    outputs = render.remote(summary)
    for fmt, data in outputs.items():
        output = path.with_name(path.stem+'-training.'+fmt)
        output.write_bytes(base64.b64decode(data, validate=True)); print(output)

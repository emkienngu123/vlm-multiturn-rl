"""Export fresh H100 training diagnostics on a CPU-only Modal function."""
import base64
import io
import json
from pathlib import Path
import modal

image = modal.Image.debian_slim(python_version='3.12').pip_install(
    'contourpy==1.4.0', 'cycler==0.12.1', 'fonttools==4.67.0', 'kiwisolver==1.5.1',
    'matplotlib==3.11.2', 'numpy==2.5.4', 'packaging==26.3', 'pillow==13.0.0',
    'pyparsing==3.3.3', 'python-dateutil==2.9.0.post0', 'six==1.17.0')
app = modal.App('vlm-h100-grpo-diagnostics')


@app.function(image=image, cpu=1, memory=2048, timeout=180)
def render(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows=summary['rounds']; x=[r['round'] for r in rows]
    fig, axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    for r in rows:
        for j,reward in enumerate(r['group']['returns']):
            axes[0].scatter(r['round']+(j-1.5)*.1,reward,s=20,
                            color='#16a34a' if reward else '#94a3b8',alpha=.8)
    axes[0].plot(x,[r['group']['mean'] for r in rows],marker='D',color='#2563eb',label='Four-trajectory mean')
    axes[0].set(xlabel='Overall round (training states vary)',ylabel='Binary training return',
        title='Fresh H100 exploration',ylim=(-.08,1.08),xticks=x)
    axes[0].legend(fontsize=8,loc='center right')
    axes[1].plot(x,[r['update']['gradient_norm'] for r in rows],marker='o',color='#d97706')
    axes[1].axhline(1.,color='#64748b',ls='--',lw=1.,label='Gradient clipping threshold')
    axes[1].set(xlabel='Overall round (zero = skipped group)',ylabel='LoRA norm before clipping',
        title='Real optimizer updates')
    axes[1].legend(fontsize=8)
    phases=summary['phases']; names=['before']
    if 'smoke' in phases and phases['smoke']['adapter_fingerprint'] != phases['after']['adapter_fingerprint']:
        names.append('smoke')
    names.append('after')
    labels={'before':'H100 before','smoke':'After two groups','after':'H100 after'}
    bars=axes[2].bar([labels[n] for n in names],[phases[n]['successes'] for n in names],
        color=['#94a3b8']+['#2563eb']*(len(names)-1))
    axes[2].bar_label(bars,labels=[f"{phases[n]['successes']}/16" for n in names],padding=3)
    axes[2].set(ylabel='Successful development episodes',title='Paired H100 greedy evaluation',
        ylim=(0,18),yticks=[0,4,8,12,16])
    axes[2].tick_params(axis='x',labelrotation=12)
    for ax in axes: ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    fig.suptitle(f"{summary['new_rounds']} fresh groups / {summary['new_optimizer_steps']} new updates / "
        f"{summary['new_skipped_groups']} skips — final test unopened",fontsize=13)
    outputs={}
    for fmt in ('png','svg'):
        stream=io.BytesIO(); fig.savefig(stream,format=fmt,dpi=160)
        outputs[fmt]=base64.b64encode(stream.getvalue()).decode()
    plt.close(fig); return outputs


@app.local_entrypoint()
def main(workflow: str):
    path=Path(workflow); summary=json.loads(path.with_name(path.stem+'-summary.json').read_text())
    for fmt,data in render.remote(summary).items():
        output=path.with_name(path.stem+'-training.'+fmt)
        output.write_bytes(base64.b64decode(data,validate=True)); print(output)
    report=path.with_suffix('.html')
    if report.exists():
        content=report.read_text()
        if '<h2>Diagnostics</h2>' not in content:
            name=path.with_name(path.stem+'-training.png').name
            content=content.replace('<h2>Training replays</h2>',
                f'<h2>Diagnostics</h2><img src="{name}" style="width:100%" alt="H100 training returns, gradient norms, and paired development success"><h2>Training replays</h2>')
            report.write_text(content)

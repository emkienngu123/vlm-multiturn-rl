import base64
import io
import json
from pathlib import Path

import modal

# Standalone CPU presentation image. Keep its plotting dependencies identical
# to the tested GPU lock without importing the GPU entry point remotely.
image=(modal.Image.debian_slim(python_version='3.12').pip_install(
    'contourpy==1.4.0','cycler==0.12.1','fonttools==4.67.0','kiwisolver==1.5.1',
    'matplotlib==3.11.2','numpy==2.5.4','packaging==26.3','pillow==13.0.0',
    'pyparsing==3.3.3','python-dateutil==2.9.0.post0','six==1.17.0'))

app=modal.App('vlm-form-sft-diagnostics')

@app.function(image=image,cpu=1,memory=2048,timeout=180)
def render_diagnostics(rows,boundaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=[r['update'] for r in rows]
    loss=[r['mean_action_loss'] for r in rows]
    smooth=[sum(loss[max(0,i-9):i+1])/len(loss[max(0,i-9):i+1]) for i in range(len(loss))]
    norms=[r['gradient_norm'] for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    axes[0].plot(x,loss,color='#64748b',alpha=.45,lw=.8,label='Four-action batch mean (last batch: one)')
    axes[0].plot(x,smooth,color='#2563eb',lw=1.8,label='Trailing 10-update mean')
    axes[0].set(xlabel='Optimizer update',ylabel='Mean teacher-action cross entropy',yscale='log',title='Training loss')
    axes[0].legend(fontsize=8)
    axes[1].plot(x,norms,color='#d97706',lw=1.,label='Gradient norm before clipping')
    axes[1].axhline(1.,color='#2563eb',ls='--',lw=1.,label='Clipping threshold')
    axes[1].set(xlabel='Optimizer update',ylabel='Language-LoRA gradient norm',title='Gradient diagnostics')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(True,alpha=.15)
        for boundary in boundaries[:-1]:ax.axvline(boundary,color='#94a3b8',alpha=.3,lw=.8)
    fig.suptitle('One data pass: 853 teacher actions / 214 optimizer updates',fontsize=13)
    outputs={}
    for fmt in ('png','svg'):
        buffer=io.BytesIO();fig.savefig(buffer,format=fmt,dpi=160)
        outputs[fmt]=base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return outputs

@app.local_entrypoint()
def main(workflow: str):
    path=Path(workflow);data=json.loads(path.read_text())
    if data['status']!='complete':raise ValueError('Wait for the completed comparison')
    root=path.parent
    rows=json.loads((root/data['last_checkpoint']/'training.json').read_text())['updates']
    outputs=render_diagnostics.remote(rows,[r['to_update'] for r in data['chunks']])
    for fmt,value in outputs.items():
        output=path.with_name(path.stem+'-training.'+fmt)
        output.write_bytes(base64.b64decode(value,validate=True));print(output)

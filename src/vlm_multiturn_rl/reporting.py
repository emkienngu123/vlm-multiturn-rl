"""Paired evaluations and uncertainty; plotting is remote-only and lazy."""

import csv
from html import escape
import math
from pathlib import Path

from .gridworld import summarize


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if type(total) is not int or type(successes) is not int or not 0 <= successes <= total or total == 0:
        raise ValueError("Need 0 <= successes <= total with total > 0")
    p = successes/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0.0, center-half), min(1.0, center+half)]


def compare_policies(episodes: list[dict], policies: dict[str, str], eval_layouts: list[dict]) -> dict:
    expected = [x["layout_id"] for x in eval_layouts]
    rows, lookups = {}, {}
    for label, policy in policies.items():
        selected = [e for e in episodes if e["policy"] == policy]
        if len(selected) != len(expected) or {e["layout_id"] for e in selected} != set(expected):
            raise ValueError("Each policy must evaluate each held-out board exactly once")
        lookups[label] = {e["layout_id"]: e for e in selected}
        metrics = summarize(selected)
        returns = [e["return"] for e in selected]
        mean = sum(returns)/len(returns)
        rows[label] = {**metrics, "return_variance": sum((r-mean)**2 for r in returns)/len(returns),
                       "success_wilson_95": wilson_interval(int(metrics["successes"]), len(selected))}
    if set(policies) != {"base", "sft", "grpo"}:
        raise ValueError("Comparison requires base, SFT, and GRPO")
    pairs = [{"layout_id": key, **{label: {"success": lookup[key]["success"], "steps": lookup[key]["steps"],
                                          "return": lookup[key]["return"]} for label, lookup in lookups.items()}}
             for key in expected]
    wins = sum(row["grpo"]["success"] and not row["sft"]["success"] for row in pairs)
    losses = sum(row["sft"]["success"] and not row["grpo"]["success"] for row in pairs)
    return {"policies": rows, "paired_boards": pairs,
            "grpo_vs_sft": {"wins": wins, "losses": losses, "ties": len(pairs)-wins-losses,
                            "success_rate_delta": (wins-losses)/len(pairs)},
            "uncertainty_note": "Wilson intervals assume independent sampled boards; these few fixed mazes are diagnostic."}


def write_curves(directory: Path, groups: list[dict], evaluations: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [g["round"] for g in groups]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(rounds, [g["mean"] for g in groups], "o-", color="#2563eb")
    axes[0, 0].set(title="Sampled training return (one board/group)", ylabel="Mean episode return")
    x, y, lo, hi = [], [], [], []
    for row in evaluations:
        m = row["metrics"]
        interval = wilson_interval(int(m["successes"]), int(m["episodes"]))
        x.append(row["round"]); y.append(m["success_rate"])
        lo.append(max(0, m["success_rate"]-interval[0])); hi.append(max(0, interval[1]-m["success_rate"]))
    axes[0, 1].errorbar(x, y, yerr=[lo, hi], fmt="o-", color="#16a34a", capsize=4)
    axes[0, 1].set(title="Held-out greedy success (Wilson 95%)", ylabel="Success rate", ylim=(-.05, 1.05))
    axes[1, 0].plot(rounds, [g["std"]**2 for g in groups], "o-", color="#9333ea")
    axes[1, 0].set(title="Within-group reward variance", ylabel="Population variance")
    axes[1, 1].plot(rounds, [g["update"]["gradient_norm"] for g in groups], "o-", color="#ea580c")
    axes[1, 1].set(title="Gradient norm before clipping (0 = skipped)", ylabel="Gradient norm")
    for ax in axes.flat:
        ax.set_xlabel("Completed collection round")
        ax.grid(alpha=.2)
        ax.xaxis.get_major_locator().set_params(integer=True)
    fig.suptitle("Stage 6: training diagnostics and held-out evaluation")
    fig.savefig(directory / "learning-curves.svg")
    fig.savefig(directory / "learning-curves.png", dpi=140)
    plt.close(fig)
    with (directory / "learning-curves.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["round", "mean_train_return", "reward_variance", "gradient_norm", "optimizer_step",
                         "collection_seconds", "update_seconds", "rollout_tokens_per_second", "peak_allocated_bytes"])
        for g in groups:
            writer.writerow([g["round"], g["mean"], g["std"]**2, g["update"]["gradient_norm"],
                             g["update"]["optimizer_step"], g["collection_seconds"], g["update_seconds"],
                             g["rollout_tokens_per_second"], g["peak_allocated_bytes"]])


def report_html(comparison: dict, training: dict, environment: dict) -> str:
    rows = []
    for name, m in comparison["policies"].items():
        lo, hi = m["success_wilson_95"]
        rows.append(f"<tr><th>{escape(name)}</th><td>{m['successes']}/{m['episodes']}</td>"
                    f"<td>{lo:.1%}–{hi:.1%}</td><td>{m['mean_return']:.3f}</td>"
                    f"<td>{m['mean_path_efficiency']:.3f}</td><td>{m['mean_steps']:.2f}</td>"
                    f"<td>{m['invalid_action_rate']:.1%}</td><td>{m['blocked_move_rate']:.1%}</td></tr>")
    paired = comparison["grpo_vs_sft"]
    return f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage 6 comparison</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 16px;color:#172033}}
table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #ccd3dd}}
img{{width:100%;height:auto}}p{{line-height:1.6}}</style>
<h1>Stage 6: base → SFT → GRPO</h1>
<p>Identical held-out layouts, full-history observations, tool format, and greedy decoding.</p>
<table><tr><th>Policy</th><th>Success</th><th>Wilson 95%</th><th>Return</th><th>Efficiency</th>
<th>Actions</th><th>Invalid</th><th>Blocked</th></tr>{''.join(rows)}</table>
<p>GRPO vs SFT: {paired['wins']} board wins, {paired['losses']} losses, {paired['ties']} ties;
success-rate difference {paired['success_rate_delta']:+.1%}. {escape(comparison['uncertainty_note'])}</p>
<p>{training['collection_rounds']} rounds; {training['optimizer_steps']} optimizer steps;
{training['zero_variance_groups']} equal-return groups skipped. Current job compute time:
{environment['total_seconds']:.1f} s; peak allocated GPU memory: {environment['peak_allocated_bytes']/2**30:.2f} GiB.</p>
<img src="learning-curves.svg" alt="Training return, held-out success, reward variance, and gradient norms">
<p>Training returns use sampled actions on training boards. Evaluation uses greedy actions on fixed held-out boards.
Each training point can use a different maze; these curves alone do not establish improvement.</p>
<p><a href="replay.html">Inspect every episode</a> · <a href="learning-curves.csv">Download curve data</a>
· <a href="comparison.json">Paired board outcomes</a> · <a href="training.json">Resume verification</a></p></html>'''

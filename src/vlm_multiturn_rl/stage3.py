"""Stage 3 baseline suite, image-only VLM adapter, and self-contained replays."""

import base64
from dataclasses import asdict, dataclass
from html import escape
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import time

from .gridworld import (
    GRID_PROMPT, Layout, make_layouts, render_observation, render_svg, replay_episode,
    run_episode, shortest_path, summarize,
)
from .vision import Stage2Config, generated_suffix
from .workspace import RUNS_PATH, run_path, verify_artifacts, write_json, write_manifest


ARTIFACT_FILES = (
    "config.json", "environment.json", "layouts.json", "episodes.json",
    "metrics.json", "replay.html", "policy-prompt.txt",
)
GPU_ARTIFACT_FILES = (*ARTIFACT_FILES, "model-cache.json", "preview.png")


@dataclass(frozen=True)
class Stage3Config(Stage2Config):
    episodes: int = 4
    dataset_seed: int = 2026
    max_steps: int = 8

    def __post_init__(self):
        super().__post_init__()
        if type(self.episodes) is not int or not 1 <= self.episodes <= 16:
            raise ValueError("episodes must be an integer between 1 and 16")
        if type(self.dataset_seed) is not int or not 0 <= self.dataset_seed < 2**32:
            raise ValueError("dataset_seed must be in [0, 2**32)")
        if type(self.max_steps) is not int or self.max_steps != 8:
            raise ValueError("Stage 3 fixes the collection cap at eight actions")


def evaluate(config: Stage3Config, actor=None) -> dict:
    train = make_layouts("train", 8, config.dataset_seed)
    evaluation = make_layouts("eval", config.episodes, config.dataset_seed)
    if {x.maze_id for x in train} & {x.maze_id for x in evaluation}:
        raise RuntimeError("Train and evaluation mazes overlap")
    if any(not 2 <= len(shortest_path(x)) <= 6 for x in (*train, *evaluation)):
        raise RuntimeError("Generated layout violates the path-length constraint")
    episodes, metrics = [], {}
    for policy in ("random", "oracle", "vlm") if actor is not None else ("random", "oracle"):
        rows = []
        for index, layout in enumerate(evaluation):
            episode = run_episode(layout, policy, seed=config.seed, max_steps=config.max_steps, actor=actor)
            replay_episode(layout, episode)
            rows.append(episode)
            print(f"{policy} {index+1}/{len(evaluation)}: success={episode['success']} "
                  f"steps={episode['steps']} return={episode['return']:.2f}", flush=True)
        episodes.extend(rows)
        metrics[policy] = summarize(rows)
    if metrics["oracle"]["success_rate"] != 1.0 or metrics["oracle"]["mean_path_efficiency"] != 1.0:
        raise RuntimeError("The shortest-path oracle failed a reachable evaluation task")
    return {
        "layouts": {"train": [x.to_dict() for x in train], "eval": [x.to_dict() for x in evaluation]},
        "episodes": episodes, "metrics": metrics,
        "verification": {
            "train_eval_mazes_disjoint": True, "all_tasks_reachable_within_cap": True,
            "oracle_solves_all_eval_tasks": True, "all_episodes_replayed": True,
            "paired_evaluation_layouts": True,
        },
    }


def replay_html(data: dict, title: str = "Stage 3: environment baselines") -> str:
    layouts = {x["layout_id"]: Layout.from_dict(x) for group in data["layouts"].values() for x in group}
    rows = "".join(
        f"<tr><th>{escape(name)}</th><td>{m['successes']}/{m['episodes']}</td>"
        f"<td>{m['mean_return']:.3f}</td><td>{m['mean_steps']:.2f}</td>"
        f"<td>{m['mean_path_efficiency']:.3f}</td><td>{m['invalid_action_rate']:.1%}</td>"
        f"<td>{m['blocked_move_rate']:.1%}</td></tr>"
        for name, m in data["metrics"].items()
    )
    sections = []
    for episode in data["episodes"]:
        layout = layouts[episode["layout_id"]]
        frames = [f"<figure>{render_svg(layout, layout.start)}<figcaption>Initial observation</figcaption></figure>"]
        for step in episode["transitions"]:
            caption = (f"Step {step['step']}: {step['text']!r}; r={step['reward']:.2f}; "
                       f"blocked={step['blocked']}; terminated={step['terminated']}; truncated={step['truncated']}")
            frames.append(f"<figure>{render_svg(layout, step['after'])}<figcaption>{escape(caption)}</figcaption></figure>")
        sections.append(
            f"<details><summary>{escape(episode['policy'])} · layout {layout.layout_id[:10]} · "
            f"{episode['end_reason']} · return {episode['return']:.2f}</summary>"
            f"<div class=frames>{''.join(frames)}</div></details>"
        )
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 16px;color:#172033}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{padding:10px;border-bottom:1px solid #ccd3dd;text-align:left}}
details{{border:1px solid #ccd3dd;border-radius:8px;margin:12px 0}}summary{{padding:14px;cursor:pointer}}
.frames{{display:flex;gap:16px;overflow-x:auto;padding:16px}}figure{{margin:0;flex:0 0 230px}}
svg{{width:230px;height:230px}}figcaption{{font-size:13px;line-height:1.5;margin-top:6px}}</style>
<h1>{escape(title)}</h1><p>Blue circle: agent. Green square: goal. Dark cells: walls.</p>
<p>Open an episode and scroll its frames from left to right. Frames reconstruct recorded states as SVG;
the VLM receives raster images. The oracle has privileged state access; random and VLM have no action mask.</p>
<table><thead><tr><th>Policy</th><th>Success</th><th>Mean return</th><th>Mean actions</th>
<th>Efficiency*</th><th>Invalid</th><th>Blocked</th></tr></thead><tbody>{rows}</tbody></table>
<p>*Shortest-path length / actions on success; zero on failure. Small diagnostic sample, not a generalization claim.</p>
{''.join(sections)}</html>"""


def save_report(directory: Path, config: Stage3Config, data: dict, environment: dict,
                source_sha256: dict, extra_files: dict[str, bytes] | None = None) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    for filename, content in {
        "config.json": asdict(config), "environment.json": environment,
        "layouts.json": data["layouts"], "episodes.json": {"episodes": data["episodes"]},
        "metrics.json": {"metrics": data["metrics"], "verification": data["verification"]},
    }.items():
        write_json(directory / filename, content)
    (directory / "replay.html").write_text(replay_html(data))
    (directory / "policy-prompt.txt").write_text(GRID_PROMPT + "\n")
    for name, content in (extra_files or {}).items():
        (directory / name).write_bytes(content)
    artifact_files = GPU_ARTIFACT_FILES if extra_files is not None else ARTIFACT_FILES
    manifest = write_manifest(directory, {
        "source_sha256": source_sha256, "environment_version": "grid-v1",
        "mode": environment["mode"],
    }, stage=3, artifact_files=artifact_files)
    verify_artifacts(directory, stage=3, artifact_files=artifact_files)
    return manifest


def replay_saved_run(directory: Path) -> int:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("environment_version") != "grid-v1" or manifest.get("mode") not in {"cpu", "gpu"}:
        raise ValueError("Unknown grid environment version or report mode")
    files = GPU_ARTIFACT_FILES if manifest["mode"] == "gpu" else ARTIFACT_FILES
    verify_artifacts(directory, stage=3, artifact_files=files)
    layouts = json.loads((directory / "layouts.json").read_text())
    lookup = {x["layout_id"]: Layout.from_dict(x) for x in layouts["eval"]}
    episodes = json.loads((directory / "episodes.json").read_text())["episodes"]
    for episode in episodes:
        replay_episode(lookup[episode["layout_id"]], episode)
    return len(episodes)


class GridVLM:
    """Greedy, current-image-only baseline; no coordinate or environment argument."""

    def __init__(self, config: Stage3Config):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor, GenerationConfig, Qwen2_5_VLForConditionalGeneration
        from .stage1 import seed_everything
        from .stage2 import SNAPSHOT_PATTERNS

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("The VLM baseline requires a BF16-capable CUDA GPU")
        seed_everything(config.seed)
        snapshot = snapshot_download(config.model_id, revision=config.model_revision,
                                     allow_patterns=SNAPSHOT_PATTERNS, local_files_only=True)
        if Path(snapshot).name != config.model_revision:
            raise RuntimeError("Loaded snapshot differs from the pinned revision")
        self.config = config
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            snapshot, local_files_only=True, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation="sdpa",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        self.prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": GRID_PROMPT}]}],
            tokenize=False, add_generation_prompt=True,
        )
        self.generation = GenerationConfig(
            do_sample=False, max_new_tokens=config.max_new_tokens, use_cache=True,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            temperature=1.0, top_p=1.0, top_k=50, repetition_penalty=1.0, num_beams=1,
        )

    def __call__(self, image) -> dict:
        import torch

        inputs = self.processor(
            text=[self.prompt], images=[image], return_tensors="pt", add_special_tokens=False,
            images_kwargs={"size": {"shortest_edge": self.config.min_pixels, "longest_edge": self.config.max_pixels}},
        )
        prompt_ids = inputs["input_ids"][0].tolist()
        grids = inputs["image_grid_thw"].tolist()
        expected = sum(t*h*w // self.model.config.vision_config.spatial_merge_size**2 for t, h, w in grids)
        if prompt_ids.count(self.model.config.image_token_id) != expected:
            raise RuntimeError("Image tokens and grid do not align")
        gpu_inputs = {k: v.to("cuda:0", dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                      for k, v in inputs.items()}
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.inference_mode():
            outputs = self.model.generate(**gpu_inputs, generation_config=self.generation)
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        ids = generated_suffix(prompt_ids, outputs[0].cpu().tolist())
        eos_ids = self.generation.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
        return {
            "text": self.processor.tokenizer.decode(ids, skip_special_tokens=True),
            "generated_ids": ids, "prompt_token_count": len(prompt_ids),
            "image_grid_thw": grids, "image_token_count": expected,
            "finish_reason": "eos" if ids and ids[-1] in eos_ids else "length", "seconds": seconds,
        }


def evaluate_remote(run_id: str, raw_config: dict, source_sha256: dict, cache_receipt: dict) -> dict:
    import torch

    config = Stage3Config(**raw_config)
    if cache_receipt["revision"] != config.model_revision:
        raise RuntimeError("Cache receipt differs from the requested model revision")
    actor = GridVLM(config)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    data = evaluate(config, actor=actor)
    environment = {
        "mode": "gpu", "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__), "transformers": importlib.metadata.version("transformers"),
        "model_revision": config.model_revision, "observation_mode": "current_image_only",
        "decoding": "greedy", "dtype": str(actor.model.dtype), "attention_implementation": "sdpa",
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "producer_task_id": os.environ["MODAL_TASK_ID"], "modal_image_id": os.environ["MODAL_IMAGE_ID"],
        "evaluation_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "generation_config": actor.generation.to_dict(),
    }
    preview = io.BytesIO()
    layout = Layout.from_dict(data["layouts"]["eval"][0])
    render_observation(layout, layout.start).save(preview, format="PNG")
    directory = run_path(RUNS_PATH, run_id)
    manifest = save_report(directory, config, data, environment, source_sha256, {
        "model-cache.json": (json.dumps(cache_receipt, indent=2) + "\n").encode(),
        "preview.png": preview.getvalue(),
    })
    return {
        "manifest": manifest, "config": asdict(config), "environment": environment,
        "metrics": data["metrics"], "verification": data["verification"],
        "files": {name: base64.b64encode((directory/name).read_bytes()).decode("ascii")
                  for name in (*GPU_ARTIFACT_FILES, "manifest.json")},
    }

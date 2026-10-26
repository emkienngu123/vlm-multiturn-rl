"""Pinned TRL integration; heavy dependencies are imported inside GPU functions."""

import base64
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from html import escape
import importlib.metadata
import inspect
import itertools
import json
import math
from pathlib import Path
import time
import types
import zipfile

from .gridworld import Layout, make_layouts, render_observation, shortest_path
from .stage3 import replay_html
from .stage4 import _evaluate, _images_zip
from .stage5 import GRPOPolicy, Stage5Config, adapter_fingerprint, source_receipt, unpack_adapter
from .trajectories import TOOL_PROMPT, move_call, replay_trajectory
from .trl_environment import CHAT_TEMPLATE, GridToolEnvironment, dispatch_response
from .workspace import RUNS_PATH, run_path, sha256_file, verify_artifacts, write_json, write_manifest


TRL_VERSION = "1.16.0"
ARTIFACT_FILES = (
    "config.json", "environment.json", "history.json", "training.json", "integration.json",
    "metrics.json", "report.html", "replay.html", "images.zip", "adapter.zip",
    "source-adapter.json", "model-cache.json", "chat-template.txt", "policy-prompt.txt",
    "requirements-trl.txt",
)


@dataclass(frozen=True)
class Stage7Config(Stage5Config):
    updates: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.updates != 1 or self.train_layouts < 2:
            raise ValueError("Stage 7 is a one-update integration check and needs at least two train layouts")


def dataset_row(layout: Layout) -> dict:
    # reset() supplies image then text. Private layout JSON is never templated.
    return {"prompt": [{"role": "system", "content": [{"type": "text", "text": TOOL_PROMPT}]},
                       {"role": "user", "content": []}], "layout": json.dumps(layout.to_dict())}


def install_grid_protocol(processor):
    """Keep the tutorial's strict action parser at TRL's tokenizer boundary."""
    from trl.chat_template_utils import (is_chat_template_prefix_preserving, qwen3_template,
                                         supports_tool_calling)

    processor.chat_template = processor.tokenizer.chat_template = CHAT_TEMPLATE
    processor.tokenizer.response_template = qwen3_template

    def parse_response(tokenizer, ids, **kwargs):
        if isinstance(ids, str):
            text = ids.removesuffix(tokenizer.eos_token)
        else:
            ids = list(ids)
            if ids and ids[-1] == tokenizer.eos_token_id:
                ids = ids[:-1]
            text = tokenizer.decode(ids, skip_special_tokens=False)
        return dispatch_response(text)

    processor.tokenizer.parse_response = types.MethodType(parse_response, processor.tokenizer)
    capability = supports_tool_calling(processor)
    prefix = is_chat_template_prefix_preserving(processor)
    if not capability or not prefix:
        raise RuntimeError(f"Unsafe grid template: supports_tool_calling={capability}, prefix_preserving={prefix}")


def make_trainer_class():
    """Only generation/lifecycle hooks change; native TRL owns tool masks and loss.

    Private hooks are version-pinned and their installed source hashes are saved.
    In particular, this class does not implement _tool_call_loop or _compute_loss.
    """
    import torch
    from trl import GRPOTrainer

    class GridGRPOTrainer(GRPOTrainer):
        _scripted = False

        def _generate_single_turn(self, prompt_ids, images, multimodal_fields,
                                  num_generations, has_tool_images=False):
            if num_generations not in (1, 4):
                raise RuntimeError("The tutorial uses four repeated rows, then one continuation per history")
            results = []
            for index, ids in enumerate(prompt_ids):
                row_images = images[index]
                env = self._image_owners[id(row_images[-1])] if has_tool_images else self.environments[index]
                if env._env.terminated or env._env.truncated:
                    # Native TRL already appended/masked the final board. Empty
                    # output stops the tool loop without inventing a final action.
                    results.append([])
                    continue
                inputs = self._actor.encode_history(env._history, row_images)
                if env._transitions:
                    previous = env._transitions[-1]["generation"]
                    message = deepcopy(env._history[-1])
                    message["content"][-1]["image"] = row_images[-1]
                    # Independently encode just the public tool response. Native
                    # TRL adds a newline, not a synthetic EOS, after sampled IDs.
                    tool_ids = self.processing_class.apply_chat_template(
                        [message], tokenize=True, return_dict=False, add_generation_prompt=True)[0]
                    expected = (previous["prompt_ids"] + previous["generated_ids"]
                                + self._tokenizer.encode("\n", add_special_tokens=False) + tool_ids)
                else:
                    expected = inputs["input_ids"][0].tolist()
                if expected != ids:
                    raise RuntimeError("Native generation did not preserve the exact sampled history and tool suffix")
                if len(ids)+self._actor.config.max_new_tokens > self._actor.config.max_sequence_tokens:
                    raise RuntimeError("Native history exceeds the context guard")
                # Obtain image features through the same reference processor,
                # but condition on actual packed IDs, including length cutoffs.
                inputs["input_ids"] = torch.tensor([ids], dtype=torch.long)
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
                inputs["mm_token_type_ids"] = (inputs["input_ids"] == self._image_pad_token_id).long()
                fields = {k: v for k, v in inputs.items() if k not in {"input_ids", "attention_mask"}}
                if self._scripted:
                    text = move_call(shortest_path(env._layout, env._env.position)[0])
                    generated = self._tokenizer.encode(text, add_special_tokens=False) + [self._tokenizer.eos_token_id]
                else:
                    # Native generation, one history at a time; no group-batch KV cache.
                    generated = super()._generate_single_turn(
                        [ids], [row_images], fields, 1, has_tool_images=has_tool_images)[0][0]
                    text_ids = generated[:-1] if generated[-1] in self.eos_token_ids else generated
                    text = self._tokenizer.decode(text_ids, skip_special_tokens=False)
                if not generated or self._image_pad_token_id in generated:
                    raise RuntimeError("Expected text action IDs, not generated image placeholders")
                env._set_decision(text, {
                    "generated_ids": generated, "prompt_ids": ids,
                    "image_paths": [o["image"] for o in env._observations],
                    "image_grid_thw": inputs["image_grid_thw"].tolist(),
                    "token_source": "solver_integration_check" if self._scripted else "model_generation",
                })
                results.append(generated)
            return results, None

        def _generate(self, prompts):
            before = adapter_fingerprint(self.model)
            output = super()._generate(prompts)
            prompt_ids, completion_ids, masks, _, _, _, images, tool_images = output
            rows = []
            for env, pids, cids, mask, imgs, new_imgs in zip(
                    self.environments, prompt_ids, completion_ids, masks, images, tool_images, strict=True):
                episode = env._episode()  # Also rejects context-limit rollbacks/early exits.
                sampled = [token for t in episode["transitions"] for token in t["generation"]["generated_ids"]]
                if [token for token, include in zip(cids, mask, strict=True) if include] != sampled:
                    raise RuntimeError("Native TRL loss mask did not select exactly sampled assistant IDs")
                if len(imgs) != episode["steps"]+1 or len(new_imgs) != episode["steps"]:
                    raise RuntimeError("A tool observation was dropped")
                if any(a.tobytes() != b.tobytes() for a, b in zip(imgs, env._images, strict=True)):
                    raise RuntimeError("Images were reordered or shared across environments")
                if any(include and token == self._image_pad_token_id for token, include in zip(cids, mask)):
                    raise RuntimeError("Image observations entered the policy loss")
                # All rollout images are square and use the same processor budget.
                visual_per_image = env._transitions[0]["generation"]["image_grid_thw"][0]
                visual_per_image = math.prod(visual_per_image) // 4
                if (pids+cids).count(self._image_pad_token_id) != len(imgs)*visual_per_image:
                    raise RuntimeError("Image features and the exact packed token stream disagree")
                rows.append({"layout_id": episode["layout_id"], "return": episode["return"],
                             "steps": episode["steps"], "image_count": len(imgs),
                             "image_hashes": [__import__('hashlib').sha256(im.tobytes()).hexdigest() for im in imgs],
                             "assistant_tokens": sum(mask), "tool_tokens": len(mask)-sum(mask),
                             "prompt_ids": pids, "completion_ids": cids, "tool_mask": mask,
                             "observation_and_exact_token_checks": True})
            if adapter_fingerprint(self.model) != before:
                raise RuntimeError("The behavior policy changed during group collection")
            self._audits.append({"scripted": self._scripted, "behavior_fingerprint": before, "rows": rows})
            return output

        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            masks = output["completion_mask"] * output["tool_mask"]
            if masks.sum().item() != sum(r["assistant_tokens"] for r in self._audits[-1]["rows"]):
                raise RuntimeError("Padding or truncation changed the native assistant loss mask")
            if output["image_grid_thw"].shape[0] != sum(r["image_count"] for r in self._audits[-1]["rows"]):
                raise RuntimeError("Native scoring dropped an image")
            self._audits[-1]["advantages"] = output["advantages"].detach().cpu().tolist()
            if not torch.isfinite(output["advantages"]).all():
                raise RuntimeError("Nonfinite native TRL advantages")
            return output

    return GridGRPOTrainer


def protocol_checks(processor):
    """Exercise native suffix concatenation against a full re-encoding on CPU."""
    from trl import GRPOTrainer

    assets = {}
    env = GridToolEnvironment(renderer=render_observation, assets=assets, episodes=[], image_owners={},
                              next_key=lambda: "cpu-protocol")
    layout = make_layouts("train", 2)[1]
    env.reset(**dataset_row(layout))
    def encode():
        text = processor.apply_chat_template(env._history, tokenize=False, add_generation_prompt=True)
        return processor(text=[text], images=env._images, return_tensors="pt", add_special_tokens=False,
                         images_kwargs={"size": dict(processor.image_processor.size)})["input_ids"][0].tolist()
    before = encode()
    text = move_call(shortest_path(layout)[0])
    target = processor.tokenizer.encode(text, add_special_tokens=False)+[processor.tokenizer.eos_token_id]
    env._set_decision(text, {"generated_ids": target})
    content = env.move(**dispatch_response(text)["tool_calls"][0]["function"]["arguments"])
    stub = object.__new__(GRPOTrainer)
    stub.processing_class, stub._tokenizer, stub._is_vlm = processor, processor.tokenizer, True
    stub.chat_template, stub.chat_template_kwargs = None, {}
    suffix = stub._get_tool_suffix_ids([{"role": "tool", "name": "move", "content": content}])
    native, reference = before+target+suffix, encode()
    if native != reference:
        index = next((i for i, (a,b) in enumerate(zip(native, reference)) if a != b), min(len(native),len(reference)))
        raise RuntimeError(json.dumps({"prefix_mismatch_index": index, "native_length": len(native),
                                      "reference_length": len(reference), "tokenizer_eos": processor.tokenizer.eos_token,
                                      "native_near_mismatch": processor.tokenizer.decode(native[max(0,index-8):index+12]),
                                      "reference_near_mismatch": processor.tokenizer.decode(reference[max(0,index-8):index+12])}))
    tool_ids = processor.apply_chat_template([{"role": "tool", "content": content}],
                                             tokenize=True, return_dict=False, add_generation_prompt=True)[0]
    independent_suffix = processor.tokenizer.encode("\n", add_special_tokens=False)+tool_ids
    if suffix != independent_suffix:
        raise RuntimeError("Native tool suffix differs from independent tool-message encoding")
    # A length-cutoff action must NOT acquire an invented EOS when continued.
    cutoff = processor.tokenizer.encode('<tool_call>{"name":', add_special_tokens=False)
    native_cutoff = before+cutoff+suffix
    if native_cutoff != before+cutoff+independent_suffix or native_cutoff[len(before)+len(cutoff)-1] != cutoff[-1]:
        raise RuntimeError("Cutoff continuation changed sampled tokens")
    return {"two_turn_native_prefix_matches_reference_when_eos_present": True,
            "native_suffix_matches_independent_encoding": True, "length_cutoff_ids_preserved_without_invented_eos": True}


def objective_comparison():
    """Compare actual TRL loss/gradients with the reference, including masked tokens."""
    import torch
    from trl import GRPOTrainer
    from .grpo import clipped_token_terms, reference_loss

    class LocalAccelerator:
        num_processes = 1
        def gather(self, value): return value
        def reduce(self, value, **kwargs): return value

    trainer = object.__new__(GRPOTrainer)
    for key, value in dict(aux_loss_enabled=False, top_entropy_quantile=1., off_policy_mask_threshold=None,
                           importance_sampling_level="token", beta=0., loss_type="grpo", epsilon_low=.2,
                           epsilon_high=.2, use_vllm=False, _entropy_bonus_enabled=False,
                           current_gradient_accumulation_steps=1, accelerator=LocalAccelerator(),
                           _metrics={"eval": defaultdict(list), "train": defaultdict(list)},
                           args=types.SimpleNamespace(delta=None)).items():
        setattr(trainer, key, value)
    model = torch.nn.Identity().eval()
    trainer.model = model
    new = torch.tensor([[math.log(1.5), math.log(.5), 2., 0.],
                        [math.log(.5), math.log(1.5), -2., 0.]], requires_grad=True)
    old = torch.zeros_like(new)
    masks = torch.tensor([[1, 1, 0, 1], [1, 1, 0, 0]])
    advantages = torch.tensor([1., -1.])
    trainer._get_per_token_logps_and_entropies = lambda *a, **kw: (new, torch.zeros_like(new), None)
    inputs = {"prompt_ids": torch.ones(2, 2, dtype=torch.long), "prompt_mask": torch.ones(2, 2),
              "completion_ids": torch.ones(2, 4, dtype=torch.long),
              "completion_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
              "tool_mask": torch.tensor([[1, 1, 0, 1], [1, 1, 0, 1]]),
              "advantages": advantages, "old_per_token_logps": old}
    native = trainer._compute_loss(model, inputs)
    native_grad = torch.autograd.grad(native, new)[0]
    reference = torch.stack([clipped_token_terms(new[i], old[i], advantages[i], loss_mask=masks[i].bool()).mean()
                             for i in range(2)]).mean()
    reference_grad = torch.autograd.grad(reference, new)[0]
    scalar = sum(reference_loss(new[i].tolist(), old[i].tolist(), masks[i].bool().tolist(), float(advantages[i]))
                 for i in range(2))/2
    if not (torch.allclose(native, reference) and torch.allclose(native_grad, reference_grad)
            and math.isclose(native.item(), scalar, abs_tol=1e-6)):
        raise RuntimeError("Pinned TRL surrogate differs from reference with matching inputs/settings")
    zero_inputs = {**inputs, "advantages": torch.zeros(2)}
    zero = trainer._compute_loss(model, zero_inputs)
    if zero.item() != 0 or torch.autograd.grad(zero, new)[0].abs().sum().item() != 0:
        raise RuntimeError("Zero-variance advantage should produce zero policy gradient")
    return {"native_loss": native.item(), "reference_loss": scalar, "values_match": True,
            "gradients_match": True, "masked_positions_have_zero_gradient": bool((native_grad[masks == 0] == 0).all()),
            "zero_advantage_has_zero_loss_and_gradient": True,
            "settings": {"loss_type": "grpo", "epsilon": .2, "beta": 0., "importance_sampling_level": "token"},
            "advantage_difference": "TRL: sample std + 1e-4; reference: population std + 1e-6. Comparison supplies identical advantages."}


def replay_saved_run(directory: Path) -> int:
    verify_artifacts(directory, stage=7, artifact_files=ARTIFACT_FILES)
    history = json.loads((directory / "history.json").read_text())
    lookup = {x["layout_id"]: Layout.from_dict(x) for group in history["layouts"].values() for x in group}
    with zipfile.ZipFile(directory / "images.zip") as archive:
        for episode in history["episodes"]:
            replay_trajectory(lookup[episode["layout_id"]], episode)
            if any(o["image"] not in archive.namelist() for o in episode["observations"]):
                raise ValueError("Missing observation asset")
    integration = json.loads((directory / "integration.json").read_text())
    for audit in integration["native_rollouts"]:
        if len(audit["rows"]) != 4 or len({r["layout_id"] for r in audit["rows"]}) != 1:
            raise ValueError("TRL grouping mixed different initial boards")
        for row in audit["rows"]:
            if (len(row["completion_ids"]) != len(row["tool_mask"])
                    or sum(row["tool_mask"]) != row["assistant_tokens"]
                    or row["image_count"] != row["steps"]+1):
                raise ValueError("Corrupt packed trajectory audit")
    return len(history["episodes"])


def train_remote(run_id: str, raw_config: dict, source_hashes: dict, cache_receipt: dict, source: dict):
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer
    from .checkpoints import adapter_zip

    config = Stage7Config(**raw_config)
    if importlib.metadata.version("trl") != TRL_VERSION:
        raise RuntimeError("Private hooks require the exact tested TRL version")
    if source_receipt(config) != source or cache_receipt["revision"] != config.model_revision:
        raise RuntimeError("Source adapter or model cache changed")
    directory = run_path(RUNS_PATH, run_id)
    directory.mkdir(parents=True, exist_ok=False)
    temporary = run_path(Path("/tmp"), run_id)
    unpack_adapter(run_path(RUNS_PATH, config.sft_run_id)/"adapter.zip", temporary/"adapter")
    started = time.monotonic()
    actor = GRPOPolicy(config, temporary/"adapter")
    assets, rollout_episodes, owners, audits = {}, [], {}, []
    history = {"layouts": {split: [x.to_dict() for x in make_layouts(split, count, config.dataset_seed)]
                           for split, count in (("train", config.train_layouts), ("eval", config.episodes))},
               "episodes": [], "metrics": {}}
    install_grid_protocol(actor.processor)
    actor.processor.image_processor.size = {"shortest_edge": config.min_pixels, "longest_edge": config.max_pixels}
    counter = itertools.count()
    def next_key(): return f"trl-{next(counter):05d}"
    factory = lambda: GridToolEnvironment(renderer=render_observation, assets=assets,
                                          episodes=rollout_episodes, image_owners=owners,
                                          next_key=next_key, seed=config.seed)
    # A fixed training row, selected by index before sampling; no held-out selection.
    training_layout = Layout.from_dict(history["layouts"]["train"][1])
    args = GRPOConfig(
        output_dir=str(temporary/"trainer"), max_steps=1, per_device_train_batch_size=1,
        gradient_accumulation_steps=4, num_generations=4, num_iterations=1,
        max_completion_length=3584, max_tool_calling_iterations=8,
        loss_type="grpo", importance_sampling_level="token", beta=0., epsilon=.2,
        scale_rewards="group", temperature=1., top_p=1., top_k=0,
        generation_kwargs={"max_new_tokens": config.max_new_tokens, "use_cache": False, "suppress_tokens": None},
        learning_rate=config.learning_rate, weight_decay=0., lr_scheduler_type="constant",
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        disable_dropout=True, remove_unused_columns=False, mask_truncated_completions=False,
        report_to="none", save_strategy="no", logging_steps=1, disable_tqdm=True,
        seed=config.seed, data_seed=config.seed,
    )
    trainer_type = make_trainer_class()
    trainer = trainer_type(model=actor.model, args=args, processing_class=actor.processor,
                           train_dataset=Dataset.from_list([dataset_row(training_layout)]),
                           environment_factory=factory)
    trainer._actor, trainer._image_owners, trainer._audits = actor, owners, audits
    comparison = objective_comparison()
    # Native tool loop with a known two-action path isolates integration from model ability.
    probe_layout = next(x for x in make_layouts("train", 32, config.dataset_seed) if len(shortest_path(x)) == 2)
    if probe_layout.layout_id not in {x["layout_id"] for x in history["layouts"]["train"]}:
        history["layouts"]["train"].append(probe_layout.to_dict())
    trainer._scripted = True
    actor.model.eval()
    trainer._generate_and_score_completions([deepcopy(dataset_row(probe_layout)) for _ in range(4)])
    for episode in rollout_episodes:
        episode["policy"] = "oracle_integration_check"
    probe_rows = audits[-1]["rows"]
    if any(row["steps"] != 2 or len(set(row["image_hashes"])) != 3 for row in probe_rows):
        raise RuntimeError("The two-turn changing-image check failed")
    print("Native two-turn image/mask check passed", flush=True)
    trainer._scripted = False
    # Validate the native boundary before spending GPU time on paired evaluation.
    with actor.model.disable_adapter():
        _evaluate(actor, history, assets, "base")
    _evaluate(actor, history, assets, "sft")
    before = adapter_fingerprint(actor.model)
    set_seed(config.seed)
    events = []
    class UpdateRecorder(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            grads = [p.grad.detach().float() for p in actor.model.parameters() if p.requires_grad and p.grad is not None]
            norm = math.sqrt(sum(g.square().sum().item() for g in grads))
            if not math.isfinite(norm): raise RuntimeError("Nonfinite gradient norm")
            events.append({"step": state.global_step+1, "clipped_gradient_norm": norm})
    trainer.add_callback(UpdateRecorder())
    trainer.train()
    actor.model.eval()
    after = adapter_fingerprint(actor.model)
    if trainer.state.global_step != 1 or len(events) != 1:
        raise RuntimeError("Expected exactly one trainer optimizer step")
    history["episodes"].extend(rollout_episodes)
    _evaluate(actor, history, assets, "trl")
    training_rows = audits[-1]["rows"]
    training = {"optimizer_steps": trainer.state.global_step, "adapter_changed": before != after,
                "before_adapter": before, "after_adapter": after, "gradient_events": events,
                "returns": [r["return"] for r in training_rows], "advantages": audits[-1]["advantages"],
                "zero_variance_group": len({r["return"] for r in training_rows}) == 1,
                "trainer_log_history": trainer.state.log_history,
                "training_layout_index": 1, "training_layout_id": training_layout.layout_id,
                "note": "TRL counts zero-gradient optimizer steps; inspect adapter_changed and reward variance."}
    integration = {"trl_version": TRL_VERSION, "objective_comparison": comparison,
                   "two_turn_changing_images": True, "native_rollouts": audits,
                   "native_source_sha256": {name: __import__('hashlib').sha256(inspect.getsource(getattr(GRPOTrainer, name)).encode()).hexdigest()
                                             for name in ("_tool_call_loop", "_compute_loss", "_generate_and_score_completions")},
                   "args": args.to_dict(), "extension": "strict parser; sequential native generation; terminal empty output. Native tool loop, loss and advantages unchanged."}
    environment = {"mode": "gpu", "gpu": torch.cuda.get_device_name(), "total_seconds": time.monotonic()-started,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate")},
                   "evaluation_decoding": actor.generation.to_dict()}
    for name, value in {"config.json": raw_config, "environment.json": environment, "history.json": history,
                        "training.json": training, "integration.json": integration, "metrics.json": history["metrics"],
                        "source-adapter.json": source, "model-cache.json": cache_receipt}.items():
        write_json(directory/name, value)
    (directory/"images.zip").write_bytes(_images_zip(assets))
    (directory/"adapter.zip").write_bytes(adapter_zip(actor.model))
    (directory/"policy-prompt.txt").write_text(TOOL_PROMPT+"\n")
    (directory/"chat-template.txt").write_text(CHAT_TEMPLATE)
    (directory/"requirements-trl.txt").write_bytes(Path("/root/requirements-trl.txt").read_bytes())
    (directory/"replay.html").write_text(replay_html(history, "Stage 7: native TRL image tools"))
    summary = {"training": training, "metrics": history["metrics"],
               "objective_comparison": comparison, "environment": environment}
    (directory/"report.html").write_text('<!doctype html><meta charset="utf-8"><title>Stage 7</title>'
        '<style>body{max-width:1000px;margin:40px auto;font:16px system-ui}pre{white-space:pre-wrap}</style>'
        '<h1>Stage 7: TRL integration</h1><p><a href="replay.html">Replay all episodes</a></p>'
        '<p>Four scripted oracle episodes verify image plumbing; they are separate from learned rollouts and evaluation.</p>'
        '<pre>'+escape(json.dumps(summary, indent=2))+'</pre>')
    manifest = write_manifest(directory, {"mode": "gpu", "protocol": "move-json-v1", "source_sha256": source_hashes},
                              stage=7, artifact_files=ARTIFACT_FILES)
    replay_saved_run(directory)
    return {"config": raw_config, "environment": environment, "metrics": history["metrics"], "training": training,
            "manifest": manifest, "verification": {"two_turn_changing_images": True, "native_mask_checked": True,
                                                     "objective_and_gradient_match": True, "all_episodes_replayed": True},
            "files": {name: base64.b64encode((directory/name).read_bytes()).decode("ascii")
                      for name in (*ARTIFACT_FILES, "manifest.json")}}

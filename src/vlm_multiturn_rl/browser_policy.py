"""A readable browser policy with identical sampling and scoring forward paths.

The prefix loop is intentionally slower than scoring a whole completion at once.
It avoids comparing different BF16 forward shapes when validating behavior scores.
"""

from .stage5 import GRPOPolicy, extend_inputs
from .trajectories import image_paths
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class ActivationBudget:
    """Per-action cap on large saved tensor references retained on the GPU.

    Count each save, even shared storage, conservatively. Small tensors and
    parameters are outside this budget; leave headroom for forward/backward.
    """
    limit_bytes: int = 0
    retained_bytes: int = 0
    offloaded_bytes: int = 0

    def __post_init__(self):
        if type(self.limit_bytes) is not int or self.limit_bytes < 0:
            raise ValueError('Activation budget must be a nonnegative integer')

    def keep(self, size, free_bytes, reserve_bytes):
        if (self.retained_bytes + size <= self.limit_bytes
                and free_bytes >= reserve_bytes + size):
            self.retained_bytes += size
            return True
        self.offloaded_bytes += size
        return False


def walk_prefixes(next_log_probs, *, choose=None, target_ids=None, max_tokens=64, eos_ids=()):
    """Shared causal traversal, also testable with plain Python probability lists."""
    if (choose is None) == (target_ids is None):
        raise ValueError("Provide either a sampling callback or exact target IDs")
    if target_ids is not None and not target_ids:
        raise ValueError("Need at least one generated target token")
    count = len(target_ids) if target_ids is not None else max_tokens
    if count <= 0: raise ValueError("Need a positive token budget")
    ids, selected = [], []
    for index in range(count):
        log_probs = next_log_probs(tuple(ids))
        token = target_ids[index] if target_ids is not None else choose(log_probs)
        if type(token) is not int or not 0 <= token < len(log_probs):
            raise ValueError("Generated ID is outside the vocabulary")
        selected.append(log_probs[token])
        ids.append(token)
        if token in eos_ids:
            if target_ids is not None and index != count-1:
                raise ValueError("A recorded continuation contains tokens after EOS")
            break
    return ids, selected


class BrowserPrefixPolicy(GRPOPolicy):
    """Stage 10's correctness path; grid reference policy remains unchanged."""

    def _next_log_probs(self, inputs, prefix):
        batch = extend_inputs(inputs, list(prefix)) if prefix else inputs
        # Last-position-only projection in BOTH paths. Positions/vision inputs
        # are recomputed from the same full prefix, without KV state.
        outputs = self.model(**batch, use_cache=False, logits_to_keep=1)
        return outputs.logits[0, -1].float().log_softmax(-1)

    def sample(self, messages, images):
        import torch

        inputs = self.encode_history(messages, images)
        prompt_ids = inputs["input_ids"][0].tolist()
        gpu_inputs = self.to_gpu(inputs)
        eos = self.rollout_generation.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        self.model.eval()
        with torch.no_grad():
            ids, selected = walk_prefixes(
                lambda prefix: self._next_log_probs(gpu_inputs, prefix),
                choose=lambda log_probs: int(torch.multinomial(log_probs.exp(), 1).item()),
                max_tokens=self.config.max_new_tokens, eos_ids=eos)
            logps = [float(value) for value in selected]
        ended = ids[-1] in eos
        return {"text": self.processor.tokenizer.decode(ids[:-1] if ended else ids, skip_special_tokens=False),
                "generated_ids": ids, "prompt_ids": prompt_ids, "sampled_log_probs": logps,
                "token_source": "model_generation", "policy_scoring": "prefix-v1",
                "image_paths": image_paths(messages), "image_grid_thw": inputs["image_grid_thw"].tolist(),
                "finish_reason": "eos" if ended else "length"}

    def score(self, record, assets):
        import torch

        generation = record["generation"]
        paths = image_paths(record["messages"])
        inputs = self.encode_history(record["messages"], [assets[path] for path in paths])
        if (inputs["input_ids"][0].tolist() != generation["prompt_ids"]
                or paths != generation["image_paths"]
                or inputs["image_grid_thw"].tolist() != generation["image_grid_thw"]):
            raise RuntimeError("Prefix scoring changed the sampled history or image order")
        ids = generation["generated_ids"]
        if self.model.config.image_token_id in ids:
            raise RuntimeError("Generated image placeholders cannot be scored as text actions")
        eos = self.rollout_generation.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        gpu_inputs = self.to_gpu(inputs)
        self.model.eval()
        _, selected = walk_prefixes(lambda prefix: self._next_log_probs(gpu_inputs, prefix),
                                    target_ids=ids, eos_ids=eos)
        return torch.stack(selected)

    def score_batched_diagnostic(self, record, assets):
        """Retain the old fast path as a diagnostic, never as behavior scores."""
        return super().score(record, assets)


class ReplayCursor:
    """Choose exact recorded IDs after checking the actual causal prefix."""

    def __init__(self, target_ids, eos_ids=()):
        self.ids = list(target_ids)
        if not self.ids or any(type(i) is not int or i < 0 for i in self.ids):
            raise ValueError("Need nonempty nonnegative token IDs")
        if any(i in eos_ids for i in self.ids[:-1]):
            raise ValueError("Recorded tokens continue after EOS")
        self.index = 0

    def next(self, actual_prefix):
        if list(actual_prefix) != self.ids[:self.index] or self.index >= len(self.ids):
            raise RuntimeError("Cached replay changed the exact causal token prefix")
        token = self.ids[self.index]
        self.index += 1
        return token


class BrowserCachedPolicy(GRPOPolicy):
    """Fast no-grad probes: sample and replay with identical cached generate calls.

    Replay records the UNCONSTRAINED policy score before forcing each actual ID.
    The forced distribution controls traversal only and never supplies old logps.
    This is deliberately not an autograd scoring API for a later GRPO update.
    """

    def __init__(self, config, adapter_directory):
        super().__init__(config, adapter_directory)
        self.rollout_generation.use_cache = True

    def sample(self, messages, images):
        result = super().sample(messages, images)
        result["policy_scoring"] = "cached-replay-v1"
        return result

    def score(self, record, assets):
        import torch

        if torch.is_grad_enabled():
            raise RuntimeError("Cached generate replay is no-grad only; validate a gradient scorer before training")
        return self._replay(record, assets)

    def _replay(self, record, assets, *, differentiable=False):
        from copy import deepcopy
        import torch
        from transformers import LogitsProcessorList
        from .vision import generated_suffix

        g = record["generation"]
        paths = image_paths(record["messages"])
        inputs = self.encode_history(record["messages"], [assets[p] for p in paths])
        if (inputs["input_ids"][0].tolist() != g["prompt_ids"] or paths != g["image_paths"]
                or inputs["image_grid_thw"].tolist() != g["image_grid_thw"]):
            raise RuntimeError("Cached replay changed prompt tokens or image order")
        if self.model.config.image_token_id in g["generated_ids"]:
            raise RuntimeError("Generated image placeholders are not text actions")
        eos = self.rollout_generation.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        cursor = ReplayCursor(g["generated_ids"], eos)
        prompt_length = len(g["prompt_ids"])
        captured = []

        class ReplayProcessor:
            def __call__(self, input_ids, scores):
                token = cursor.next(input_ids[0,prompt_length:].tolist())
                # Capture BEFORE masking; forced logits would give a bogus logp 0.
                captured.append(scores[0].float().log_softmax(-1)[token])
                forced = torch.full_like(scores, -float("inf"))
                forced[0,token] = 0
                return forced

        self.model.eval()
        replay_config = deepcopy(self.rollout_generation)
        replay_config.max_new_tokens = len(cursor.ids)
        with torch.random.fork_rng(devices=[0]):
            output = self.model.generate(**self.to_gpu(inputs), generation_config=replay_config,
                logits_processor=LogitsProcessorList([ReplayProcessor()]),
                **({"custom_generate": gradient_decode} if differentiable else {}))
        if (generated_suffix(g["prompt_ids"],output.sequences[0].tolist()) != cursor.ids
                or len(captured) != len(cursor.ids)):
            raise RuntimeError("Cached replay did not traverse every recorded assistant token")
        return torch.stack(captured)

    def score_batched_diagnostic(self, record, assets):
        return super().score(record, assets)


def gradient_decode(model, input_ids, logits_processor, stopping_criteria, generation_config,
                    synced_gpus=False, streamer=None, **model_kwargs):
    """Pinned Transformers callback: identical _sample implementation, autograd on.

    generate() normally applies no_grad. Its supported custom callable receives
    prepared inputs/caches; enable gradients for prefill AND all decoding steps.
    No KV tensors are detached: later scores differentiate through their history.
    """
    import torch

    with torch.enable_grad():
        result = model._sample(input_ids, logits_processor=logits_processor,
            stopping_criteria=stopping_criteria, generation_config=generation_config,
            synced_gpus=synced_gpus, streamer=streamer, **model_kwargs)
        values = [layer.values for layer in result.past_key_values.layers]
        if not values or any(v is None or not v.requires_grad or v.grad_fn is None for v in values):
            raise RuntimeError("Cached values lost their autograd history")
        return result


class BrowserTrainPolicy(BrowserCachedPolicy):
    """Sample and score under the same gradient-enabled cached forward kernels."""

    def sample(self, messages, images):
        import torch
        from .vision import generated_suffix

        inputs = self.encode_history(messages,images)
        prompt_ids = inputs["input_ids"][0].tolist()
        self.model.eval()
        # Graphs exist only for this action and are released when output leaves
        # scope. Collection never calls backward or steps an optimizer.
        with saved_activation_memory(self.model,backward=False):
            output = self.model.generate(**self.to_gpu(inputs), generation_config=self.rollout_generation,
                                         custom_generate=gradient_decode)
        ids = generated_suffix(prompt_ids,output.sequences[0].tolist())
        if not ids or len(output.scores)!=len(ids): raise RuntimeError("Missing sampled token scores")
        with torch.no_grad():
            logps = [float(row[0].float().log_softmax(-1)[token]) for row,token in zip(output.scores,ids)]
        eos = self.rollout_generation.eos_token_id
        eos = [eos] if isinstance(eos,int) else eos
        ended = ids[-1] in eos
        return {"text":self.processor.tokenizer.decode(ids[:-1] if ended else ids,skip_special_tokens=False),
                "generated_ids":ids,"prompt_ids":prompt_ids,"sampled_log_probs":logps,
                "token_source":"model_generation","policy_scoring":"grad-cache-v1",
                "image_paths":image_paths(messages),"image_grid_thw":inputs["image_grid_thw"].tolist(),
                "finish_reason":"eos" if ended else "length"}

    def score_grad(self, record, assets, *, backward=True):
        import torch

        stats = {}
        with saved_activation_memory(self.model,backward=backward,
                gpu_budget_bytes=getattr(self, 'activation_gpu_budget_bytes', 0), stats=stats):
            scores = self._replay(record,assets,differentiable=True)
        self.last_activation_stats = stats
        if not scores.requires_grad: raise RuntimeError("Replay silently detached the policy graph")
        return scores


@contextmanager
def saved_activation_memory(model, *, backward, gpu_budget_bytes=0, reserve_bytes=12*1024**3, stats=None):
    """Preserve forward grad kernels while bounding saved-tensor GPU memory.

    Collection needs forward values only, so saved backward tensors can be
    discarded. Training copies large activations to CPU exactly. Parameter
    storages stay on GPU: offloading a frozen 3B weight for each decoded token
    would otherwise create hundreds of GB of redundant CPU copies.
    KV tensors themselves are never detached from the computational graph.
    """
    import torch

    budget = ActivationBudget(gpu_budget_bytes)
    parameter_storages={p.untyped_storage().data_ptr() for p in model.parameters()}
    def pack(tensor):
        if not backward:return None
        if (tensor.is_cuda and tensor.numel()>=65536
                and tensor.untyped_storage().data_ptr() not in parameter_storages):
            size = tensor.numel()*tensor.element_size()
            free = torch.cuda.mem_get_info(tensor.device)[0] if gpu_budget_bytes else 0
            if budget.keep(size, free, reserve_bytes):
                return (tensor.device,tensor.detach())
            return (tensor.device,tensor.detach().to('cpu',copy=True))
        return (tensor.device,tensor.detach())
    def unpack(saved):
        if saved is None:raise RuntimeError("Collection-only graph cannot be backpropagated")
        device,tensor=saved
        return tensor.to(device)
    with torch.autograd.graph.saved_tensors_hooks(pack,unpack):
        yield
    if stats is not None:
        stats.update(gpu_budget_bytes=gpu_budget_bytes, retained_bytes=budget.retained_bytes,
                     offloaded_bytes=budget.offloaded_bytes)

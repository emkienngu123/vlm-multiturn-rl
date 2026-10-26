"""Round-boundary checkpoints. CPU metadata inspection needs no torch/Pillow."""

from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
import zipfile

from .stage4 import _images_zip
from .stage5 import adapter_fingerprint
from .workspace import verify_artifacts, write_json, write_manifest


FILES = ("checkpoint.json", "adapter.zip", "training-state.pt", "history.json", "images.zip")
RESUME_MUTABLE_FIELDS = {"updates", "resume_run_id", "resume_round", "verify_resume"}


def validate_resume_config(previous: dict, current: dict, completed_round: int) -> None:
    if set(previous) != set(current):
        raise ValueError("Checkpoint configuration schema changed")
    changed = [key for key in previous if key not in RESUME_MUTABLE_FIELDS and previous[key] != current[key]]
    if changed:
        raise ValueError(f"Resume changes training/evaluation settings: {', '.join(changed)}")
    if current["updates"] <= completed_round:
        raise ValueError("--updates is the total round target and must exceed the checkpoint round")


def metadata(directory: Path) -> dict:
    manifest = verify_artifacts(directory, stage=6, artifact_files=FILES)
    if manifest.get("kind") != "checkpoint" or manifest.get("protocol") != "move-json-v1":
        raise ValueError("Not a Stage 6 training checkpoint")
    return json.loads((directory / "checkpoint.json").read_text())


def clone_cpu(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone_cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_cpu(v) for v in value)
    return value


def tree_digest(value) -> str:
    """Content identity independent of tensor device and torch.save ZIP timestamps."""
    import torch

    digest = hashlib.sha256()
    def visit(item):
        digest.update(type(item).__name__.encode())
        if isinstance(item, torch.Tensor):
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for entry in item:
                visit(entry)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def capture_rng() -> dict:
    import numpy as np
    import torch

    name, keys, position, has_gauss, cached_gauss = np.random.get_state()
    return {"python": random.getstate(), "numpy": [name, keys.tolist(), position, has_gauss, cached_gauss],
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state: dict) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    name, keys, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, has_gauss, cached_gauss))
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def rng_probe() -> dict:
    """Read the next draws, then put every generator back exactly where it was."""
    import numpy as np
    import torch

    state = capture_rng()
    result = {"python": random.random(), "numpy": float(np.random.random()),
              "torch_cpu": torch.rand(3).tolist(), "torch_cuda": torch.rand(3, device="cuda:0").tolist()}
    restore_rng(state)
    return result


def adapter_zip(model) -> bytes:
    with tempfile.TemporaryDirectory() as temporary:
        model.save_pretrained(temporary, safe_serialization=True, save_embedding_layers=False)
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(Path(temporary).iterdir()):
                if path.is_file():
                    archive.write(path, arcname=f"adapter/{path.name}")
        return output.getvalue()


def save_checkpoint(directory: Path, actor, optimizer, history: dict, assets: dict,
                    source: dict, source_hashes: dict, runtime: dict, proof: dict | None = None) -> dict:
    import torch

    directory.mkdir(parents=True, exist_ok=False)
    names = [name for name, p in actor.model.named_parameters() if p.requires_grad]
    state = {"optimizer": clone_cpu(optimizer.state_dict()), "rng": capture_rng()}
    groups = history["groups"]
    info = {"config": asdict(actor.config), "completed_round": len(groups),
            "optimizer_steps": sum(g["update"]["optimizer_step"] for g in groups),
            "parameter_names": names, "adapter_fingerprint": adapter_fingerprint(actor.model),
            "optimizer_digest": tree_digest(state["optimizer"]), "rng_probe": rng_probe(),
            "source": source, "source_sha256": source_hashes, "runtime": runtime,
            "continuation_reference": proof}
    (directory / "adapter.zip").write_bytes(adapter_zip(actor.model))
    torch.save(state, directory / "training-state.pt")
    write_json(directory / "history.json", history)
    (directory / "images.zip").write_bytes(_images_zip(assets))
    write_json(directory / "checkpoint.json", info)
    write_manifest(directory, {"kind": "checkpoint", "protocol": "move-json-v1"}, stage=6, artifact_files=FILES)
    metadata(directory)
    return info


def restore_training(directory: Path, actor, optimizer) -> dict:
    import torch

    info = metadata(directory)
    names = [name for name, p in actor.model.named_parameters() if p.requires_grad]
    if names != info["parameter_names"] or adapter_fingerprint(actor.model) != info["adapter_fingerprint"]:
        raise ValueError("Restored adapter or optimizer parameter ordering differs")
    state = torch.load(directory / "training-state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    if tree_digest(optimizer.state_dict()) != info["optimizer_digest"]:
        raise ValueError("Optimizer state changed while restoring")
    steps = [int(s["step"]) for s in optimizer.state_dict()["state"].values()]
    if any(step != info["optimizer_steps"] for step in steps) or (info["optimizer_steps"] > 0 and not steps):
        raise ValueError("Adam step counters do not match the checkpoint")
    restore_rng(state["rng"])
    if tree_digest(capture_rng()) != tree_digest(state["rng"]) or rng_probe() != info["rng_probe"]:
        raise ValueError("RNG restoration failed")
    return {"adapter_matches": True, "optimizer_matches": True, "rng_matches": True,
            "optimizer_states_restored": len(steps), "optimizer_steps_restored": info["optimizer_steps"]}


def load_history(directory: Path) -> tuple[dict, dict]:
    from PIL import Image

    data = json.loads((directory / "history.json").read_text())
    with zipfile.ZipFile(directory / "images.zip") as archive:
        assets = {name: Image.open(io.BytesIO(archive.read(name))).convert("RGB") for name in archive.namelist()}
    return data, assets

"""GPU workspace smoke check and independent CPU reader (no training yet)."""

from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time

from .workspace import (
    RUNS_PATH,
    Stage1Config,
    run_path,
    sha256_file,
    verify_artifacts,
    write_json,
    write_manifest,
)


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def cached_model_metadata(config: Stage1Config, *, offline: bool) -> dict:
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

    # Resolve an immutable snapshot through the public Hub API. Do not rely on
    # private Transformers config attributes to discover the loaded revision.
    snapshot = snapshot_download(
        config.model_id,
        revision=config.model_revision,
        allow_patterns=["*.json", "*.txt", "*.jinja", "*.model"],
        local_files_only=offline,
    )
    model_config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    if model_config.model_type != "qwen2_5_vl":
        raise RuntimeError(f"Unexpected model type: {model_config.model_type}")
    # Downloads config/tokenizer/processor files, never model weights in Stage 1.
    return {
        "revision": Path(snapshot).name,
        "model_type": model_config.model_type,
        "model_class": Qwen2_5_VLForConditionalGeneration.__name__,
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(processor.tokenizer).__name__,
    }


def create_run(run_id: str, raw_config: dict, source_sha256: dict) -> dict:
    import numpy as np
    import torch
    import torchvision  # Check the torch/torchvision binary pairing too.

    started = time.monotonic()
    config = Stage1Config(**raw_config)
    if not torch.cuda.is_available():
        raise RuntimeError("The GPU check requires CUDA; verify the Modal GPU request")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This tutorial expects BF16 support")

    def seed_probe():
        return {
            "python": random.random(),
            "numpy": float(np.random.random()),
            "torch_cuda": torch.rand(4, device="cuda").cpu().tolist(),
        }

    seed_everything(config.seed)
    probe = seed_probe()
    seed_everything(config.seed)
    if seed_probe() != probe:
        raise RuntimeError("Repeated draws after reseeding did not match")

    # A real CUDA operation checks more than importing torch or detecting a GPU.
    values = torch.arange(16, device="cuda", dtype=torch.bfloat16).reshape(4, 4)
    product = (values @ values.T).float().cpu()
    # BF16 rounds the output; comparing to FP32 requires an explicit tolerance.
    torch.testing.assert_close(
        product, values.float().cpu() @ values.float().cpu().T, rtol=0.01, atol=1.0,
    )
    model_metadata = cached_model_metadata(config, offline=False)
    if model_metadata["revision"] != config.model_revision:
        raise RuntimeError("Loaded model config revision differs from the requested pin")

    packages = {
        dist.metadata["Name"]: str(dist.version)
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    }
    gpu = torch.cuda.get_device_properties(0)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "producer_task_id": os.environ["MODAL_TASK_ID"],
        "modal_image_id": os.environ["MODAL_IMAGE_ID"],
        "torch": str(torch.__version__),  # TorchVersion must not cross the RPC boundary.
        "torchvision": str(torchvision.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "gpu": str(gpu.name),
        "gpu_memory_bytes": int(gpu.total_memory),
        "bf16_supported": True,
        "hf_home": os.environ["HF_HOME"],
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "packages": dict(sorted(packages.items())),
        "model": model_metadata,
    }
    directory = run_path(RUNS_PATH, run_id)
    directory.mkdir(parents=True, exist_ok=False)  # Never overwrite an existing run.
    write_json(directory / "config.json", asdict(config))
    write_json(directory / "environment.json", environment)
    (directory / "packages.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in sorted(packages.items()))
    )
    torch.save({"seed": config.seed, "product": product}, directory / "probe.pt")
    manifest = write_manifest(directory, {
        "source_sha256": source_sha256,
        "gpu_requirements_sha256": sha256_file(Path("/root/requirements-gpu.txt")),
        "seed_probe": probe,
        "probe_sum": float(product.sum()),
        "elapsed_seconds": time.monotonic() - started,
    })
    return {"manifest": manifest, "config": asdict(config), "environment": environment}


def read_run(run_id: str) -> dict:
    import torch

    directory = run_path(RUNS_PATH, run_id)
    manifest = verify_artifacts(directory)
    raw_config = json.loads((directory / "config.json").read_text())
    config = Stage1Config(**raw_config)
    environment = json.loads((directory / "environment.json").read_text())
    state = torch.load(directory / "probe.pt", map_location="cpu", weights_only=True)
    if state["seed"] != config.seed or float(state["product"].sum()) != manifest["probe_sum"]:
        raise ValueError("Saved tensor probe does not match the run manifest")
    offline_metadata = cached_model_metadata(config, offline=True)
    if offline_metadata != environment["model"]:
        raise ValueError("Offline cached model metadata differs from the writer's metadata")
    reader_task_id = os.environ["MODAL_TASK_ID"]
    different_container = reader_task_id != environment["producer_task_id"]
    if not different_container:
        raise RuntimeError("The persistence check must run in a different container")
    return {
        "manifest": manifest,
        "config": raw_config,
        "environment": environment,
        "verification": {
            "artifact_checksums_match": True,
            "tensor_loaded_on_cpu": state["product"].device.type == "cpu",
            "model_metadata_loaded_offline": True,
            "different_container": different_container,
            "reader_task_id": reader_task_id,
        },
    }

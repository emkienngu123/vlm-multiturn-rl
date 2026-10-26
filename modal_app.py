"""Modal entry point. Heavy libraries are imported only inside remote functions."""

import base64
from dataclasses import asdict, fields
import json
from pathlib import Path

import modal

from vlm_multiturn_rl.workspace import (
    CACHE_PATH,
    CACHE_VOLUME_NAME,
    RUNS_PATH,
    RUNS_VOLUME_NAME,
    config_dict,
    new_run_id,
    run_path,
    source_hashes,
    verify_artifacts,
    write_json,
)
from vlm_multiturn_rl.vision import Stage2Config
from vlm_multiturn_rl.stage3 import Stage3Config
from vlm_multiturn_rl.stage4 import Stage4Config
from vlm_multiturn_rl.stage5 import Stage5Config
from vlm_multiturn_rl.stage6 import Stage6Config
from vlm_multiturn_rl.stage7 import Stage7Config


PROJECT_ROOT = Path(__file__).resolve().parent
app = modal.App("vlm-multiturn-rl")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(PROJECT_ROOT / "requirements-gpu.txt"))
    .env({
        "HF_HOME": str(CACHE_PATH / "huggingface"),
        "PYTHONHASHSEED": "0",  # Set before Python starts; RNG seeds are set per run.
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_file(PROJECT_ROOT / "requirements-gpu.txt", "/root/requirements-gpu.txt")
    .add_local_python_source("vlm_multiturn_rl")
)

# Keep earlier lessons' tested image. TRL has its own resolved Linux lock.
trl_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(PROJECT_ROOT / "requirements-gpu.txt"))
    .pip_install_from_requirements(str(PROJECT_ROOT / "requirements-trl.txt"))
    .env({"HF_HOME": str(CACHE_PATH / "huggingface"), "PYTHONHASHSEED": "0",
          "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(PROJECT_ROOT / "requirements-trl.txt", "/root/requirements-trl.txt")
    .add_local_python_source("vlm_multiturn_rl")
)

browser_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(PROJECT_ROOT / "requirements-gpu.txt"))
    .pip_install_from_requirements(str(PROJECT_ROOT / "requirements-browser.txt"))
    .run_commands("python -m playwright install --with-deps chromium")
    .add_local_file(PROJECT_ROOT/"scripts/install_miniwob.py", "/root/install_miniwob.py", copy=True)
    .run_commands("python /root/install_miniwob.py")
    .env({"HF_HOME": str(CACHE_PATH/"huggingface"), "PYTHONHASHSEED": "0", "TOKENIZERS_PARALLELISM": "false",
          "MINIWOB_URL": "file:///opt/miniwob/source/miniwob/html/miniwob/"})
    .add_local_file(PROJECT_ROOT/"requirements-browser.txt", "/root/requirements-browser.txt")
    .add_local_python_source("vlm_multiturn_rl")
)

model_cache = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
run_store = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)
volumes = {CACHE_PATH: model_cache, RUNS_PATH: run_store}


@app.function(image=image, gpu="L40S", timeout=600, scaledown_window=2)
def check_gpu():
    import torch

    result = {
        "torch": str(torch.__version__),
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(result)
    return result


@app.function(
    image=image, gpu="L40S", volumes=volumes, timeout=600,
    max_containers=1, single_use_containers=True,
)
def write_stage1_run(run_id: str, config: dict, source_sha256: dict):
    from vlm_multiturn_rl.stage1 import create_run

    # Reload before opening files, including when a container has been reused.
    model_cache.reload()
    run_store.reload()
    result = create_run(run_id, config, source_sha256)
    model_cache.commit()
    run_store.commit()  # Return only after both persistent writes are committed.
    return result


@app.function(
    image=image, volumes=volumes, timeout=300, single_use_containers=True,
    env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
)
def read_stage1_run(run_id: str):
    from vlm_multiturn_rl.stage1 import read_run

    model_cache.reload()
    run_store.reload()
    return read_run(run_id)


@app.function(
    image=image, volumes={CACHE_PATH: model_cache}, timeout=900,
    cpu=2, memory=4096, single_use_containers=True,
)
def prepare_stage2_model(config: dict):
    from vlm_multiturn_rl.stage2 import prepare_model

    model_cache.reload()
    receipt = prepare_model(config)
    model_cache.commit()
    return receipt


@app.function(
    image=image, gpu="L40S", volumes=volumes, timeout=900,
    cpu=2, memory=16384, max_containers=1, single_use_containers=True,
    env={"HF_HUB_OFFLINE": "1"},
)
def inspect_stage2_vlm(run_id: str, config: dict, source_sha256: dict, cache_receipt: dict):
    from vlm_multiturn_rl.stage2 import inspect_vlm

    model_cache.reload()
    run_store.reload()
    result = inspect_vlm(run_id, config, source_sha256, cache_receipt)
    run_store.commit()
    return result


@app.function(
    image=image, gpu="L40S", volumes=volumes, timeout=900,
    cpu=2, memory=16384, max_containers=1, single_use_containers=True,
    env={"HF_HUB_OFFLINE": "1"},
)
def evaluate_stage3_grid(run_id: str, config: dict, source_sha256: dict, cache_receipt: dict):
    from vlm_multiturn_rl.stage3 import evaluate_remote

    model_cache.reload()
    run_store.reload()
    result = evaluate_remote(run_id, config, source_sha256, cache_receipt)
    run_store.commit()
    return result


@app.function(
    image=image, gpu="L40S", volumes=volumes, timeout=1800,
    cpu=2, memory=24576, max_containers=1, single_use_containers=True,
    env={"HF_HUB_OFFLINE": "1"},
)
def train_stage4_sft(run_id: str, config: dict, source_sha256: dict, cache_receipt: dict):
    from vlm_multiturn_rl.stage4 import train_remote

    model_cache.reload()
    run_store.reload()
    result = train_remote(run_id, config, source_sha256, cache_receipt)
    run_store.commit()
    return result


@app.function(image=image, volumes={RUNS_PATH: run_store}, timeout=300, single_use_containers=True)
def prepare_stage5_source(config: dict):
    from vlm_multiturn_rl.stage5 import Stage5Config, source_receipt

    run_store.reload()
    return source_receipt(Stage5Config(**config))


@app.function(
    image=image, gpu="L40S", volumes=volumes, timeout=1800,
    cpu=2, memory=24576, max_containers=1, single_use_containers=True,
    env={"HF_HUB_OFFLINE": "1"},
)
def train_stage5_grpo(run_id: str, config: dict, source_sha256: dict, cache_receipt: dict, source: dict):
    from vlm_multiturn_rl.stage5 import train_remote

    model_cache.reload()
    run_store.reload()
    result = train_remote(run_id, config, source_sha256, cache_receipt, source)
    run_store.commit()
    return result


@app.function(image=image, volumes={RUNS_PATH: run_store}, timeout=300, single_use_containers=True)
def prepare_stage6_experiment(config: dict, source_sha256: dict):
    from vlm_multiturn_rl.stage6 import prepare_experiment

    run_store.reload()
    return prepare_experiment(config, source_sha256)


@app.function(
    image=image, gpu="L40S", volumes=volumes, timeout=1800,
    cpu=2, memory=24576, max_containers=1, single_use_containers=True,
    env={"HF_HUB_OFFLINE": "1"},
)
def train_stage6_experiment(run_id: str, prepared: dict, source_sha256: dict, cache_receipt: dict):
    from vlm_multiturn_rl.stage6 import train_remote

    model_cache.reload()
    run_store.reload()
    result = train_remote(run_id, prepared, source_sha256, cache_receipt, run_store.commit)
    run_store.commit()
    return result


@app.function(image=image, volumes={RUNS_PATH: run_store}, timeout=300, single_use_containers=True)
def prepare_stage7_source(config: dict):
    from vlm_multiturn_rl.stage5 import source_receipt
    from vlm_multiturn_rl.stage7 import Stage7Config

    run_store.reload()
    return source_receipt(Stage7Config(**config))


@app.function(image=trl_image, volumes={CACHE_PATH: model_cache}, timeout=300, cpu=2, memory=4096, single_use_containers=True,
              env={"HF_HUB_OFFLINE": "1"})
def check_stage7_stack(config: dict):
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor
    from vlm_multiturn_rl.stage7 import dataset_row, install_grid_protocol, objective_comparison, protocol_checks
    from vlm_multiturn_rl.gridworld import make_layouts
    from vlm_multiturn_rl.stage2 import SNAPSHOT_PATTERNS

    model_cache.reload()
    snapshot = snapshot_download(config["model_id"], revision=config["model_revision"],
                                 allow_patterns=SNAPSHOT_PATTERNS, local_files_only=True)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    processor.image_processor.size = {"shortest_edge": config["min_pixels"], "longest_edge": config["max_pixels"]}
    install_grid_protocol(processor)
    Dataset.from_list([dataset_row(make_layouts("train", 2)[1])])
    return {"chat_template_and_prefix_checks": True, "dataset_arrow_check": True,
            "objective_comparison": objective_comparison(), **protocol_checks(processor)}


@app.function(
    image=trl_image, gpu="L40S", volumes=volumes, timeout=1800,
    cpu=2, memory=24576, max_containers=1, single_use_containers=True,
    env={"HF_HUB_OFFLINE": "1"},
)
def train_stage7_trl(run_id: str, config: dict, source_sha256: dict, cache_receipt: dict, source: dict):
    from vlm_multiturn_rl.stage7 import train_remote

    model_cache.reload()
    run_store.reload()
    result = train_remote(run_id, config, source_sha256, cache_receipt, source)
    run_store.commit()
    return result


@app.function(image=browser_image, volumes={RUNS_PATH: run_store}, timeout=900,
              cpu=2, memory=8192, single_use_containers=True)
def check_stage8_browser(run_id: str, config: dict, source_sha256: dict):
    from vlm_multiturn_rl.stage8 import evaluate_cpu
    from vlm_multiturn_rl.browser import close_browser_runtime

    run_store.reload()
    try:
        result = evaluate_cpu(run_id, config, source_sha256)
    except Exception as exc:
        # Preserve a readable error without requiring browser packages locally.
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Browser preflight: {type(exc).__name__}: {exc}") from None
    finally:
        close_browser_runtime()
    run_store.commit()
    return result


@app.function(image=browser_image, gpu="L40S", volumes=volumes, timeout=1800,
              cpu=2, memory=24576, single_use_containers=True, max_containers=1,
              env={"HF_HUB_OFFLINE": "1"})
def evaluate_stage8_browser(run_id: str, config: dict, source_sha256: dict, cache_receipt: dict, cpu_run_id: str):
    from vlm_multiturn_rl.stage8 import evaluate_gpu
    from vlm_multiturn_rl.browser import close_browser_runtime

    model_cache.reload()
    run_store.reload()
    try:
        result = evaluate_gpu(run_id, config, source_sha256, cache_receipt, cpu_run_id)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Browser baseline: {type(exc).__name__}: {exc}") from None
    finally:
        close_browser_runtime()
    run_store.commit()
    return result


@app.function(image=image, volumes={RUNS_PATH: run_store}, timeout=300)
def prepare_stage9_source(config: dict, cpu_run_id: str):
    from vlm_multiturn_rl.stage9 import BrowserSFTConfig, validate_source

    run_store.reload()
    return validate_source(BrowserSFTConfig(**config), run_path(RUNS_PATH, cpu_run_id))


@app.function(image=browser_image, gpu="L40S", volumes=volumes, timeout=1800,
              cpu=2, memory=24576, single_use_containers=True, max_containers=1,
              env={"HF_HUB_OFFLINE": "1"})
def train_stage9_browser(run_id: str, config: dict, source: dict, cache_receipt: dict,
                         cpu_run_id: str, receipt: dict):
    from vlm_multiturn_rl.stage9 import train_remote
    from vlm_multiturn_rl.browser import close_browser_runtime

    model_cache.reload()
    run_store.reload()
    try:
        result = train_remote(run_id, config, source, cache_receipt, cpu_run_id, receipt)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Browser SFT failed: {type(exc).__name__}: {exc}") from None
    finally:
        close_browser_runtime()
    run_store.commit()
    return result


@app.function(image=browser_image, volumes={RUNS_PATH: run_store}, timeout=900,
              cpu=2, memory=8192, single_use_containers=True)
def check_stage10_browser(run_id: str, config: dict, source: dict):
    from vlm_multiturn_rl.stage10 import evaluate_cpu
    from vlm_multiturn_rl.browser import close_browser_runtime

    run_store.reload()
    try:
        result = evaluate_cpu(run_id, config, source)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Browser readiness preflight: {type(exc).__name__}: {exc}") from None
    finally:
        close_browser_runtime()
    run_store.commit()
    return result


@app.function(image=browser_image, gpu="L40S", volumes=volumes, timeout=1800,
              cpu=2, memory=24576, single_use_containers=True, max_containers=1,
              env={"HF_HUB_OFFLINE": "1"})
def evaluate_stage10_browser(run_id: str, config: dict, source: dict, cache_receipt: dict, cpu_run_id: str):
    from vlm_multiturn_rl.stage10 import evaluate_gpu
    from vlm_multiturn_rl.browser import close_browser_runtime

    run_store.reload()
    model_cache.reload()
    try:
        result = evaluate_gpu(run_id, config, source, cache_receipt, cpu_run_id)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Browser readiness: {type(exc).__name__}: {exc}") from None
    finally:
        close_browser_runtime()
        run_store.commit()  # Preserve incomplete diagnostic files on failure too.
    return result


@app.function(image=image, volumes={RUNS_PATH: run_store}, timeout=300)
def validate_stage10_preflight(config: dict, cpu_run_id: str):
    from vlm_multiturn_rl.stage10 import BrowserReadinessConfig, validate_cpu_preflight

    run_store.reload()
    return validate_cpu_preflight(BrowserReadinessConfig(**config), cpu_run_id)


def save_readiness_result(result, output_dir):
    from vlm_multiturn_rl.stage10 import CPU_FILES, GPU_FILES, replay_saved_run

    directory = run_path(Path(output_dir), result["manifest"]["run_id"])
    directory.mkdir(parents=True, exist_ok=False)
    expected = GPU_FILES if result["environment"]["mode"] == "gpu" else CPU_FILES
    files = result.pop("files")
    if set(files) != {*expected, "manifest.json"}: raise ValueError("Unexpected readiness artifacts")
    for name, value in files.items(): (directory/name).write_bytes(base64.b64decode(value, validate=True))
    replay_saved_run(directory)
    write_json(directory/"browser-readiness-result.json", result)
    print(json.dumps({"run_id": directory.name, "report": str(directory/"report.html"),
                      "metrics": result["metrics"], "readiness": result["readiness"]}, indent=2), flush=True)


def save_browser_result(result, output_dir):
    from vlm_multiturn_rl.stage8 import ARTIFACT_FILES, GPU_ARTIFACT_FILES

    run_id = result["manifest"]["run_id"]
    directory = run_path(Path(output_dir), run_id)
    directory.mkdir(parents=True, exist_ok=False)
    allowed = GPU_ARTIFACT_FILES if result["environment"]["mode"] == "gpu" else ARTIFACT_FILES
    files = result.pop("files")
    if set(files) != {*allowed, "manifest.json"}: raise ValueError("Unexpected browser artifacts")
    for name, content in files.items():
        (directory/name).write_bytes(base64.b64decode(content, validate=True))
    verify_artifacts(directory, stage=8, artifact_files=allowed)
    write_json(directory/"browser-result.json", result)
    print(json.dumps({"run_id": run_id, "report": str(directory/"report.html"),
                      "metrics": result["metrics"], "verification": result["verification"]}, indent=2), flush=True)


@app.function(image=browser_image, volumes={RUNS_PATH: run_store}, timeout=900,
              cpu=2, memory=8192, single_use_containers=True)
def check_stage11_browser(run_id: str, config: dict, source: dict):
    from vlm_multiturn_rl.stage11 import evaluate_cpu
    from vlm_multiturn_rl.browser import close_browser_runtime

    run_store.reload()
    try:
        return evaluate_cpu(run_id, config, source)
    except Exception as exc:
        raise RuntimeError(f"Hard-task preflight: {type(exc).__name__}: {exc}") from None
    finally:
        close_browser_runtime()
        run_store.commit()


@app.function(image=image, volumes={RUNS_PATH: run_store}, timeout=300)
def validate_stage11_preflight(config: dict, cpu_run_id: str):
    from vlm_multiturn_rl.stage11 import BrowserDifficultyConfig, validate_cpu_preflight

    run_store.reload()
    return validate_cpu_preflight(BrowserDifficultyConfig(**config), cpu_run_id)


@app.function(image=browser_image, gpu="L40S", volumes=volumes, timeout=1800,
              cpu=2, memory=24576, single_use_containers=True, max_containers=1,
              env={"HF_HUB_OFFLINE": "1"})
def evaluate_stage11_browser(run_id: str, config: dict, source: dict, cache_receipt: dict,
                             cpu_run_id: str, pilot_task: str, case_index: int):
    from vlm_multiturn_rl.stage11 import evaluate_gpu
    from vlm_multiturn_rl.browser import close_browser_runtime

    run_store.reload()
    model_cache.reload()
    try:
        return evaluate_gpu(run_id, config, source, cache_receipt, cpu_run_id, pilot_task, case_index)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Hard-task pilot: {type(exc).__name__}: {exc}") from None
    finally:
        close_browser_runtime()
        run_store.commit()


def save_difficulty_result(result, output_dir):
    from vlm_multiturn_rl.stage11 import CPU_FILES, GPU_FILES, replay_saved_run

    directory = run_path(Path(output_dir), result["manifest"]["run_id"])
    directory.mkdir(parents=True, exist_ok=False)
    expected = GPU_FILES if result["environment"]["mode"] == "gpu" else CPU_FILES
    files = result.pop("files")
    if set(files) != {*expected, "manifest.json"}: raise ValueError("Unexpected difficulty artifacts")
    for name,value in files.items(): (directory/name).write_bytes(base64.b64decode(value, validate=True))
    replay_saved_run(directory)
    write_json(directory/"browser-difficulty-result.json", result)
    print(json.dumps({"run_id":directory.name,"report":str(directory/"report.html"),
                      "difficulty":result["difficulty"]},indent=2),flush=True)


@app.function(image=browser_image, gpu="L40S", volumes=volumes, timeout=1800,
              cpu=2, memory=65536, single_use_containers=True, max_containers=1,
              env={"HF_HUB_OFFLINE": "1"})
def train_stage12_browser(run_id: str, config: dict, source: dict, cache_receipt: dict, cpu_run_id: str):
    from vlm_multiturn_rl.stage12 import train_remote
    from vlm_multiturn_rl.browser import close_browser_runtime

    run_store.reload()
    model_cache.reload()
    def persist():
        close_browser_runtime()
        run_store.commit()
    try:
        return train_remote(run_id,config,source,cache_receipt,cpu_run_id,persist)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Browser GRPO: {type(exc).__name__}: {exc}") from None
    finally:
        persist()


@app.function(image=browser_image, volumes={RUNS_PATH: run_store}, timeout=1800,
              cpu=2, memory=8192, single_use_containers=True)
def prepare_stage13_forms(run_id: str, config: dict, source: dict, audit_run_id: str):
    from vlm_multiturn_rl.stage13 import create_remote
    from vlm_multiturn_rl.browser import close_browser_runtime

    run_store.reload()
    def persist():
        close_browser_runtime()
        run_store.commit()
    try:
        return create_remote(run_id,config,source,audit_run_id,persist)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Form dataset/audit: {type(exc).__name__}: {exc}") from None
    finally:
        persist()


@app.function(image=browser_image, gpu="L40S", volumes=volumes, timeout=1800,
              cpu=2, memory=24576, single_use_containers=True, max_containers=1,
              env={"HF_HUB_OFFLINE":"1"})
def train_stage14_forms(run_id: str, config: dict, source: dict, cache: dict,
                        previous_id: str, chunk_updates: int):
    from vlm_multiturn_rl.stage14 import train_chunk

    model_cache.reload();run_store.reload()
    try:
        return train_chunk(run_id,config,source,cache,previous_id,chunk_updates,run_store.commit)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Form SFT: {type(exc).__name__}: {exc}") from None
    finally: run_store.commit()


@app.function(image=browser_image, gpu="L40S", volumes=volumes, timeout=1800,
              cpu=2, memory=24576, single_use_containers=True, max_containers=1,
              env={"HF_HUB_OFFLINE":"1"})
def evaluate_stage14_forms(run_id: str, config: dict, source: dict, cache: dict,
                           checkpoint_id: str, offset: int):
    from vlm_multiturn_rl.stage14 import evaluate_remote
    from vlm_multiturn_rl.browser import close_browser_runtime

    model_cache.reload();run_store.reload()
    try:
        return evaluate_remote(run_id,config,source,cache,checkpoint_id,offset)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Form SFT evaluation: {type(exc).__name__}: {exc}") from None
    finally:
        close_browser_runtime();run_store.commit()


def save_form_sft_result(result, output_dir, *, evaluation=False):
    from vlm_multiturn_rl.stage14 import TRAIN_FILES, EVAL_FILES, replay_training, replay_evaluation

    directory=run_path(Path(output_dir),result['manifest']['run_id'])
    directory.mkdir(parents=True,exist_ok=False);files=result.pop('files')
    expected=EVAL_FILES if evaluation else TRAIN_FILES
    if set(files)!={*expected,'manifest.json'}: raise ValueError('Unexpected form SFT files')
    for name,value in files.items(): (directory/name).write_bytes(base64.b64decode(value,validate=True))
    (replay_evaluation if evaluation else replay_training)(directory)
    print(json.dumps({'run_id':directory.name,'environment':result['environment'],
                     **({'metrics':result['metrics']} if evaluation else
                        {'completed_updates':result['training']['completed_updates'],
                         'dataset_passes':result['training']['dataset_passes']})},indent=2),flush=True)
    return directory


@app.function(image=browser_image, volumes={RUNS_PATH: run_store}, timeout=1800,
              cpu=2, memory=8192, single_use_containers=True)
def prepare_stage15_exploration(run_id: str, config: dict, source: dict):
    from vlm_multiturn_rl.stage15 import prepare_remote
    from vlm_multiturn_rl.browser import close_browser_runtime

    run_store.reload()
    try:
        return prepare_remote(run_id, config, source)
    except Exception as exc:
        raise RuntimeError(f'Exploration preflight: {type(exc).__name__}: {exc}') from None
    finally:
        close_browser_runtime(); run_store.commit()


@app.function(image=browser_image, gpu='L40S', volumes=volumes, timeout=1800,
              cpu=2, memory=24576, single_use_containers=True, max_containers=1,
              env={'HF_HUB_OFFLINE': '1'})
def probe_stage15_exploration(run_id: str, config: dict, source: dict, cache: dict,
                             cpu_run_id: str, shard_index: int):
    from vlm_multiturn_rl.stage15 import probe_remote
    from vlm_multiturn_rl.browser import close_browser_runtime

    run_store.reload(); model_cache.reload()
    try:
        return probe_remote(run_id, config, source, cache, cpu_run_id, shard_index)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f'Exploration probe: {type(exc).__name__}: {exc}') from None
    finally:
        close_browser_runtime(); run_store.commit()


def save_exploration_result(result, output_dir):
    from vlm_multiturn_rl.stage15 import CPU_FILES, GPU_FILES, replay_saved_run

    directory = run_path(Path(output_dir), result['manifest']['run_id'])
    directory.mkdir(parents=True, exist_ok=False)
    files = result.pop('files')
    expected = GPU_FILES if result['environment']['mode'] == 'gpu' else CPU_FILES
    if set(files) != {*expected, 'manifest.json'}: raise ValueError('Unexpected exploration artifacts')
    for name, value in files.items(): (directory/name).write_bytes(base64.b64decode(value, validate=True))
    replay_saved_run(directory)
    print(json.dumps({'run_id': directory.name, 'readiness': result['readiness'],
                      'seconds': result['environment']['seconds']}, indent=2), flush=True)
    return directory


@app.function(image=browser_image, volumes={RUNS_PATH: run_store}, timeout=1800,
              cpu=2, memory=8192, single_use_containers=True)
def prepare_stage16_forms(run_id: str, config: dict, source: dict):
    from vlm_multiturn_rl.stage16 import prepare_remote
    from vlm_multiturn_rl.browser import close_browser_runtime

    run_store.reload()
    try: return prepare_remote(run_id, config, source)
    except Exception as exc:
        raise RuntimeError(f'Form GRPO preflight: {type(exc).__name__}: {exc}') from None
    finally:
        close_browser_runtime(); run_store.commit()


@app.function(image=browser_image, gpu='L40S', volumes=volumes, timeout=1800,
              cpu=2, memory=65536, single_use_containers=True, max_containers=1,
              env={'HF_HUB_OFFLINE': '1'})
def train_stage16_forms(run_id: str, config: dict, source: dict, cache: dict,
                        cpu_run_id: str, previous_id: str, execution: dict | None = None):
    from vlm_multiturn_rl.stage16 import train_remote
    from vlm_multiturn_rl.browser import close_browser_runtime

    run_store.reload(); model_cache.reload()
    try: return json.loads(json.dumps(train_remote(run_id, config, source, cache, cpu_run_id, previous_id,
                                                  execution=execution)))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f'Form GRPO training: {type(exc).__name__}: {exc}') from None
    finally:
        close_browser_runtime(); run_store.commit()


@app.function(image=browser_image, gpu='L40S', volumes=volumes, timeout=1800,
              cpu=2, memory=24576, single_use_containers=True, max_containers=1,
              env={'HF_HUB_OFFLINE': '1'})
def evaluate_stage16_forms(run_id: str, config: dict, source: dict, cache: dict,
                           checkpoint_id: str, offset: int):
    from vlm_multiturn_rl.stage16 import evaluate_remote
    from vlm_multiturn_rl.browser import close_browser_runtime

    run_store.reload(); model_cache.reload()
    try: return evaluate_remote(run_id, config, source, cache, checkpoint_id, offset)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f'Form GRPO evaluation: {type(exc).__name__}: {exc}') from None
    finally:
        close_browser_runtime(); run_store.commit()


@app.function(image=browser_image, gpu='L40S', volumes=volumes, timeout=1800,
              cpu=2, memory=65536, single_use_containers=True, max_containers=1,
              scaledown_window=2, env={'HF_HUB_OFFLINE': '1'})
def benchmark_browser_gpu(run_id: str, source_id: str, requested_gpu: str, budget_gib: int, source: dict):
    from vlm_multiturn_rl.gpu_benchmark import benchmark_remote
    from vlm_multiturn_rl.browser import close_browser_runtime

    run_store.reload(); model_cache.reload()
    # A JSON round trip strips framework scalar/string subclasses before Modal
    # pickles the response, keeping the local controller free of PyTorch.
    try: return json.loads(json.dumps(benchmark_remote(run_id, source_id, requested_gpu, budget_gib, source)))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f'GPU benchmark: {type(exc).__name__}: {exc}') from None
    finally:
        close_browser_runtime(); run_store.commit()


def save_form_grpo_result(result, output_dir):
    from vlm_multiturn_rl.stage16 import CPU_FILES, TRAIN_FILES, EVAL_FILES, replay_saved_run

    directory = run_path(Path(output_dir), result['manifest']['run_id']); directory.mkdir(parents=True, exist_ok=False)
    files = result.pop('files'); kind = result['manifest']['kind']
    expected = {'form-grpo-preflight': CPU_FILES, 'form-grpo-checkpoint': TRAIN_FILES, 'form-grpo-dev': EVAL_FILES}[kind]
    if set(files) != {*expected, 'manifest.json'}: raise ValueError('Unexpected form GRPO files')
    for name, value in files.items():
        path = directory/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(value, validate=True))
    replay_saved_run(directory)
    print(json.dumps({'run_id': directory.name, 'kind': kind, 'seconds': result['environment']['seconds'],
                      'metrics': result['metrics']}, indent=2), flush=True)
    return directory


@app.local_entrypoint()
def main(
    task: str = "check-gpu", seed: int = 42, run_id: str = "", output_dir: str = "artifacts",
    max_pixels: int = 448 * 448, max_new_tokens: int = -1, episodes: int = 4,
    updates: int = 2, train_layouts: int = 8, gradient_accumulation: int = 2,
    sft_run_id: str = "",
    resume_run_id: str = "", resume_round: int = 0, verify_resume: bool = False,
    checkpoint_every: int = 1, eval_every: int = 5,
    fresh_cases: int = 8, probe_cases: int = 1,
    pilot_task: str = "all",
    pilot_case: int = -1,
    audit_run_id: str = "",
    dataset_run_id: str = "", epochs: int = 1, warmup_run_id: str = "", comparison_run_id: str = "",
):
    """Workspace checks, VLM inspection, grid baselines, LoRA SFT, or trajectory GRPO."""
    if task == "check-gpu":
        print(check_gpu.remote())
        return
    if task == 'browser-h100-grpo':
        from vlm_multiturn_rl.stage18 import run_workflow

        source = source_hashes(PROJECT_ROOT)
        run_workflow(output_dir, source, prepare_stage16_forms.remote, prepare_stage2_model.remote,
            train_stage16_forms.with_options(gpu='H100!', scaledown_window=2).remote,
            evaluate_stage16_forms.with_options(gpu='H100!', scaledown_window=2).remote,
            save_form_grpo_result)
        return
    if task == 'browser-gpu-benchmark':
        from vlm_multiturn_rl.gpu_benchmark import verify_source, save_result, comparison_report

        if not run_id: raise ValueError('Supply --run-id with the first mixed Stage 16 group')
        verify_source(run_path(Path(output_dir), run_id))
        source = source_hashes(PROJECT_ROOT); paths = []
        workflow_id = new_run_id(stage=17)
        path = Path(output_dir)/(workflow_id+'-workflow.json')
        workflow = {'source_run_id': run_id, 'status': 'running', 'runs': [], 'source_sha256': source}
        write_json(path, workflow)
        if resume_run_id:
            from vlm_multiturn_rl.gpu_benchmark import FILES
            recovered = run_path(Path(output_dir), resume_run_id)
            verify_artifacts(recovered, stage=17, artifact_files=FILES)
            old = json.loads((recovered/'benchmark.json').read_text())
            if old['status'] != 'complete' or old['requested_gpu'] != 'L40S' or old['source']['run_id'] != run_id:
                raise ValueError('Resume requires the complete L40S benchmark for the selected group')
            paths.append(recovered)
            workflow['runs'].append({'run_id': resume_run_id, 'gpu': 'L40S', 'status': 'complete', 'reused': True})
            write_json(path, workflow)
        for gpu, budget in (('L40S', 16), ('H100', 40)):
            if gpu == 'L40S' and resume_run_id: continue
            new_id = new_run_id(stage=17)
            function = benchmark_browser_gpu if gpu == 'L40S' else benchmark_browser_gpu.with_options(gpu='H100!')
            bundle = function.remote(new_id, run_id, gpu, budget, source)
            paths.append(save_result(bundle, output_dir))
            workflow['runs'].append({'run_id': new_id, 'gpu': gpu, 'status': bundle['result']['status']})
            write_json(path, workflow)
        report = Path(output_dir)/(workflow_id+'-comparison.html')
        comparison_report(paths, report)
        workflow.update(status='complete', report=str(report)); write_json(path, workflow)
        print(json.dumps(workflow, indent=2), flush=True)
        return
    if task == 'browser-form-grpo':
        from vlm_multiturn_rl.stage16 import FormGRPOConfig, source_receipts, plan, replay_saved_run

        if updates not in (2, 20): raise ValueError('Use --updates 2 for smoke or --updates 20 for the bounded study')
        obj = FormGRPOConfig(seed=seed, sft_run_id=sft_run_id, comparison_run_id=comparison_run_id,
            readiness_run_id=audit_run_id, max_pixels=max_pixels,
            max_new_tokens=64 if max_new_tokens == -1 else max_new_tokens)
        receipts = source_receipts(obj, Path(output_dir)); config = asdict(obj); source = source_hashes(PROJECT_ROOT)
        workflow = {'config': config, 'receipts': receipts, 'plan': plan(obj), 'rounds': [], 'eval_phases': {},
                    'status': 'preflight', 'requested_rounds': updates, 'final_test_unopened': True}
        path = Path(output_dir)/(new_run_id(stage=16)+'-workflow.json'); write_json(path, workflow)
        cpu_id = run_id or new_run_id(stage=16)
        if not run_id: save_form_grpo_result(prepare_stage16_forms.remote(cpu_id, config, source), output_dir)
        workflow['cpu_run_id'] = cpu_id; write_json(path, workflow)
        cache = prepare_stage2_model.remote({f.name: config[f.name] for f in fields(Stage2Config)})
        def evaluate_phase(label, checkpoint):
            workflow['status'] = 'evaluating-'+label; workflow['eval_phases'][label] = []; write_json(path, workflow)
            for offset in (0, 8):
                new_id = new_run_id(stage=16)
                result = evaluate_stage16_forms.remote(new_id, config, source, cache, checkpoint, offset)
                save_form_grpo_result(result, output_dir)
                workflow['eval_phases'][label].append(new_id); write_json(path, workflow)
        evaluate_phase('before', '')
        previous = resume_run_id; rows = []
        if previous:
            replay_saved_run(run_path(Path(output_dir), previous))
            rows = json.loads((run_path(Path(output_dir), previous)/'training.json').read_text())['history']
        workflow['resume_checkpoint'] = previous
        while len(rows) < updates:
            elapsed = sum(r['seconds'] for r in rows)
            predicted = max(r['seconds'] for r in rows)*1.25+30 if rows else 0.
            if elapsed+predicted > obj.training_seconds:
                workflow['stop_reason'] = 'training_time_budget'; break
            if len(rows) == 2 and rows[-1]['optimizer_steps'] == 0:
                workflow['stop_reason'] = 'no_nonconstant_smoke_group'; break
            workflow['status'] = 'training'; write_json(path, workflow)
            new_id = new_run_id(stage=16)
            result = train_stage16_forms.remote(new_id, config, source, cache, cpu_id, previous)
            save_form_grpo_result(result, output_dir)
            rows = result['training']['history']; previous = new_id
            workflow['rounds'].append({'run_id': new_id, 'round': len(rows), **rows[-1]})
            workflow['last_checkpoint'] = previous; write_json(path, workflow)
            if len(rows) == 2: evaluate_phase('smoke', previous)
        if not previous: raise RuntimeError('No completed training round')
        if len(rows) == 2 and 'smoke' in workflow['eval_phases']:
            workflow['eval_phases']['after'] = workflow['eval_phases']['smoke']
        else: evaluate_phase('after', previous)
        workflow.update(status='complete', completed_rounds=len(rows), optimizer_steps=rows[-1]['optimizer_steps'],
                        recorded_training_seconds=sum(r['seconds'] for r in rows),
                        stop_reason=workflow.get('stop_reason', 'requested_round_target'))
        write_json(path, workflow)
        print(json.dumps({'workflow': str(path), 'rounds': len(rows), 'optimizer_steps': rows[-1]['optimizer_steps'],
                          'stop_reason': workflow['stop_reason'], 'checkpoint': previous}, indent=2), flush=True)
        return
    if task == 'browser-exploration':
        from vlm_multiturn_rl.stage15 import ExplorationConfig, source_receipts, study_plan

        config_object = ExplorationConfig(seed=seed, sft_run_id=sft_run_id, warmup_run_id=warmup_run_id,
            max_pixels=max_pixels, max_new_tokens=64 if max_new_tokens == -1 else max_new_tokens)
        receipts = source_receipts(config_object, Path(output_dir))
        config = asdict(config_object); source = source_hashes(PROJECT_ROOT)
        plan = study_plan(config_object)
        workflow = {'config': config, 'sources': receipts, 'plan': plan, 'shards': [], 'status': 'preflight'}
        path = Path(output_dir)/(new_run_id(stage=15)+'-workflow.json')
        write_json(path, workflow)
        cpu_id = run_id or new_run_id(stage=15)
        if not run_id:
            save_exploration_result(prepare_stage15_exploration.remote(cpu_id, config, source), output_dir)
        workflow['cpu_run_id'] = cpu_id; workflow['status'] = 'probing'; write_json(path, workflow)
        cache = prepare_stage2_model.remote({f.name: config[f.name] for f in fields(Stage2Config)})
        selected = plan['shards'] if pilot_case == -1 else [s for s in plan['shards'] if s['index'] == pilot_case]
        if not selected: raise ValueError('Use --pilot-case -1 (all) or 0 through 7')
        for shard in selected:
            next_id = new_run_id(stage=15)
            result = probe_stage15_exploration.remote(next_id, config, source, cache, cpu_id, shard['index'])
            save_exploration_result(result, output_dir)
            workflow['shards'].append({'run_id': next_id, **shard, 'readiness': result['readiness']})
            write_json(path, workflow)
        workflow['status'] = 'complete' if len(selected) == 8 else 'partial-study'
        write_json(path, workflow)
        print(json.dumps({'workflow': str(path), 'status': workflow['status'], 'shards': workflow['shards']}, indent=2))
        return
    if task == 'browser-form-sft':
        from vlm_multiturn_rl.stage14 import (FormSFTConfig, source_receipts, replay_training,
                                             paired_comparison, schedule)

        config_object=FormSFTConfig(seed=seed,dataset_run_id=dataset_run_id,sft_run_id=sft_run_id,
            epochs=epochs,max_pixels=max_pixels,max_new_tokens=64 if max_new_tokens==-1 else max_new_tokens)
        receipts=source_receipts(config_object,Path(output_dir))
        config=asdict(config_object);source=source_hashes(PROJECT_ROOT)
        model_config={f.name:config[f.name] for f in fields(Stage2Config)}
        cache=prepare_stage2_model.remote(model_config)
        cursor=replay_training(run_path(Path(output_dir),resume_run_id)) if resume_run_id else 0
        previous=resume_run_id;total=len(schedule(config_object,receipts['dataset']['targets']))
        workflow={'config':config,'receipts':receipts,'chunks':[],'eval_shards':[],
                  'resume_checkpoint':resume_run_id,'status':'training','final_test_unopened':True}
        workflow_path=Path(output_dir)/(new_run_id(stage=14)+'-workflow.json')
        write_json(workflow_path,workflow)
        chunk_size=2 if cursor==0 else 64
        while cursor<total:
            next_id=new_run_id(stage=14)
            print(f'Form SFT chunk {cursor}->{min(cursor+chunk_size,total)}/{total}: {next_id}',flush=True)
            result=train_stage14_forms.remote(next_id,config,source,cache,previous,chunk_size)
            save_form_sft_result(result,output_dir)
            training=result['training'];new_cursor=training['completed_updates']
            if new_cursor<=cursor: raise RuntimeError('SFT continuation did not advance')
            workflow['chunks'].append({'run_id':next_id,'from_update':cursor,'to_update':new_cursor,
                                       'seconds':result['environment']['seconds']})
            previous=next_id;cursor=new_cursor
            workflow['last_checkpoint']=previous;write_json(workflow_path,workflow)
            # Keep each next chunk comfortably below the 30-minute timeout.
            max_update=max(r['seconds'] for r in training['updates'])
            backward=[p['seconds']*4 for p in training['causal_probes'] if p.get('longest_backward_checked')]
            conservative=max([max_update,*backward])
            chunk_size=max(1,min(64,int(900/max(conservative,1.))))
        workflow['status']='development';write_json(workflow_path,workflow)
        episodes=[];decoding=None;fingerprints={}
        for checkpoint in ('',previous):
            for offset in (0,8):
                eval_id=new_run_id(stage=14)
                result=evaluate_stage14_forms.remote(eval_id,config,source,cache,checkpoint,offset)
                directory=save_form_sft_result(result,output_dir,evaluation=True)
                evaluation=json.loads((directory/'evaluation.json').read_text())
                if decoding is not None and decoding!=evaluation['decoding']: raise ValueError('Paired decoding settings differ')
                decoding=evaluation['decoding'];policy=evaluation['policy']
                if policy in fingerprints and fingerprints[policy]!=evaluation['adapter_fingerprint']:
                    raise ValueError('Evaluation shards used different adapters')
                fingerprints[policy]=evaluation['adapter_fingerprint']
                episodes.extend(json.loads((directory/'episodes.json').read_text()))
                workflow['eval_shards'].append(eval_id);write_json(workflow_path,workflow)
        workflow['comparison']=paired_comparison(episodes,config_object);workflow['status']='complete'
        write_json(workflow_path,workflow)
        print(json.dumps({'workflow':str(workflow_path),'checkpoint':previous,
                          'comparison':workflow['comparison']},indent=2),flush=True)
        return
    if task == "browser-form-data":
        from vlm_multiturn_rl.stage13 import FILES, FormDatasetConfig, replay_saved_run

        if not audit_run_id: raise ValueError("Supply the completed Stage 12 diagnostic with --audit-run-id")
        if resume_run_id or run_id: raise ValueError("Dataset preparation creates a fresh run; continuation is not supported")
        config=FormDatasetConfig(seed=seed,max_pixels=max_pixels,
                                 max_new_tokens=64 if max_new_tokens==-1 else max_new_tokens)
        new_id=new_run_id(stage=13)
        print(f"CPU form audit/dataset: {new_id}; diagnostic source: {audit_run_id}",flush=True)
        result=prepare_stage13_forms.remote(new_id,asdict(config),source_hashes(PROJECT_ROOT),audit_run_id)
        directory=run_path(Path(output_dir),new_id);directory.mkdir(parents=True,exist_ok=False)
        files=result.pop('files')
        if set(files)!={*FILES,'manifest.json'}: raise ValueError("Unexpected form dataset artifacts")
        for name,value in files.items(): (directory/name).write_bytes(base64.b64decode(value,validate=True))
        replay_saved_run(directory)
        write_json(directory/'browser-form-data-result.json',result)
        print(json.dumps({'run_id':new_id,'report':str(directory/'report.html'),**result['summary']},indent=2),flush=True)
        return
    if task == "browser-grpo":
        from vlm_multiturn_rl.stage10 import latest_local_sft
        from vlm_multiturn_rl.stage11 import validate_cpu_preflight
        from vlm_multiturn_rl.stage12 import BrowserGRPOConfig, artifact_files, pilot_config, replay_saved_run

        if resume_run_id:raise ValueError("This bounded check saves/reloads checkpoints; CLI continuation is a later lesson")
        sft_run_id=sft_run_id or latest_local_sft(Path(output_dir))
        config_object=BrowserGRPOConfig(seed=seed,episodes=episodes,sft_run_id=sft_run_id,updates=updates,
            max_pixels=max_pixels,max_new_tokens=64 if max_new_tokens==-1 else max_new_tokens)
        parent=pilot_config(config_object)
        if not run_id:
            for candidate in sorted(Path(output_dir).glob('stage11-*/manifest.json'),reverse=True):
                if json.loads(candidate.read_text()).get('mode')!='cpu':continue
                try:validate_cpu_preflight(parent,candidate.parent.name,Path(output_dir))
                except (OSError,ValueError,KeyError):continue
                run_id=candidate.parent.name;break
        if not run_id:raise ValueError("Supply the completed Stage 11 CPU preflight with --run-id")
        validate_stage11_preflight.remote(asdict(parent),run_id)
        config=asdict(config_object);source=source_hashes(PROJECT_ROOT)
        model_config={f.name:config[f.name] for f in fields(Stage2Config)}
        cache_receipt=prepare_stage2_model.remote(model_config)
        new_id=new_run_id(stage=12)
        print(f"Browser GRPO correctness run: {new_id}, preflight: {run_id}",flush=True)
        result=train_stage12_browser.remote(new_id,config,source,cache_receipt,run_id)
        directory=run_path(Path(output_dir),new_id);directory.mkdir(parents=True,exist_ok=False)
        files=result.pop('files')
        if set(files)!={*artifact_files(config_object),'manifest.json'}:raise ValueError("Unexpected browser GRPO artifacts")
        for name,value in files.items():
            path=directory/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(base64.b64decode(value,validate=True))
        replay_saved_run(directory)
        write_json(directory/'browser-grpo-result.json',result)
        print(json.dumps({'run_id':new_id,'report':str(directory/'report.html'),
                          'metrics':result['metrics'],'training':result['training']},indent=2),flush=True)
        return
    if task in ("browser-hard-check", "browser-hard-pilot"):
        from vlm_multiturn_rl.stage10 import latest_local_sft
        from vlm_multiturn_rl.stage11 import BrowserDifficultyConfig, source_receipt

        if resume_run_id: raise ValueError("Use --run-id to reuse CPU preflight; this pilot does not optimize")
        sft_run_id = sft_run_id or latest_local_sft(Path(output_dir))
        config_object = BrowserDifficultyConfig(seed=seed, episodes=episodes, sft_run_id=sft_run_id,
            max_pixels=max_pixels, max_new_tokens=64 if max_new_tokens == -1 else max_new_tokens)
        if pilot_task != "all" and pilot_task not in config_object.tasks:
            raise ValueError("--pilot-task must be all or one of "+", ".join(config_object.tasks))
        if pilot_case not in range(-1,config_object.episodes): raise ValueError("Unknown --pilot-case index")
        if (Path(output_dir)/sft_run_id/"manifest.json").is_file():
            source_receipt(config_object, Path(output_dir))
        config, source = asdict(config_object), source_hashes(PROJECT_ROOT)
        cpu_id = run_id
        if cpu_id:
            validate_stage11_preflight.remote(config, cpu_id)
        else:
            cpu_id = new_run_id(stage=11)
            print(f"Hard-task CPU preflight: {cpu_id}", flush=True)
            save_difficulty_result(check_stage11_browser.remote(cpu_id, config, source), output_dir)
        if task == "browser-hard-check": return
        model_config = {f.name: config[f.name] for f in fields(Stage2Config)}
        cache_receipt = prepare_stage2_model.remote(model_config)
        # One four-trajectory group per bounded job, one L40S at a time.
        results = []
        for selected in config_object.tasks if pilot_task == "all" else (pilot_task,):
            for case_index in range(config_object.episodes) if pilot_case == -1 else (pilot_case,):
                gpu_id = new_run_id(stage=11)
                print(f"Hard-task GPU pilot {selected} case {case_index}: {gpu_id}, preflight: {cpu_id}", flush=True)
                result = evaluate_stage11_browser.remote(gpu_id, config, source, cache_receipt, cpu_id, selected, case_index)
                save_difficulty_result(result, output_dir)
                results.append({"run_id":gpu_id,**result["difficulty"]})
                write_json(Path(output_dir)/(cpu_id+"-pilot-summary.json"),results)
        return
    if task == "browser-readiness":
        from vlm_multiturn_rl.stage10 import BrowserReadinessConfig, latest_local_sft, source_receipt

        if resume_run_id:
            raise ValueError("Readiness creates a fresh GPU run; --run-id optionally reuses a CPU preflight")
        sft_run_id = sft_run_id or latest_local_sft(Path(output_dir))
        config_object = BrowserReadinessConfig(seed=seed, episodes=episodes, sft_run_id=sft_run_id,
            fresh_cases=fresh_cases, probe_cases=probe_cases, max_pixels=max_pixels,
            max_new_tokens=64 if max_new_tokens == -1 else max_new_tokens)
        # Check a local source when available; remote preflight remains authoritative.
        if (Path(output_dir)/sft_run_id/"manifest.json").is_file():
            source_receipt(config_object, Path(output_dir))
        config = asdict(config_object)
        source = source_hashes(PROJECT_ROOT)
        cpu_id = run_id
        if cpu_id:
            print(f"Reusing verified browser CPU preflight: {cpu_id}", flush=True)
            validate_stage10_preflight.remote(config, cpu_id)
        else:
            cpu_id = new_run_id(stage=10)
            print(f"Browser readiness CPU preflight: {cpu_id}", flush=True)
            save_readiness_result(check_stage10_browser.remote(cpu_id, config, source), output_dir)
        model_config = {f.name: config[f.name] for f in fields(Stage2Config)}
        cache_receipt = prepare_stage2_model.remote(model_config)
        gpu_id = new_run_id(stage=10)
        print(f"Browser readiness GPU: {gpu_id}, preflight: {cpu_id}", flush=True)
        save_readiness_result(evaluate_stage10_browser.remote(gpu_id, config, source, cache_receipt, cpu_id), output_dir)
        return
    if task == "browser-sft":
        from vlm_multiturn_rl.stage9 import (ARTIFACT_FILES, BrowserSFTConfig,
                                            latest_local_source, replay_saved_run)

        if resume_run_id:
            raise ValueError("Browser SFT starts fresh; --sft-run-id optionally selects a duration-comparison reference")
        config_object = BrowserSFTConfig(seed=seed, episodes=episodes, updates=updates,
            gradient_accumulation=gradient_accumulation, max_pixels=max_pixels,
            max_new_tokens=64 if max_new_tokens == -1 else max_new_tokens,
            diagnostic_sft_run_id=sft_run_id)
        config = asdict(config_object)
        cpu_run_id = run_id or latest_local_source(Path(output_dir), config_object)
        source = source_hashes(PROJECT_ROOT)
        receipt = prepare_stage9_source.remote(config, cpu_run_id)
        if receipt["requirements_sha256"] != source["requirements-browser.txt"]:
            raise ValueError("Browser lock changed since preflight; rerun browser-check")
        model_config = {f.name: config[f.name] for f in fields(Stage2Config)}
        cache_receipt = prepare_stage2_model.remote(model_config)
        new_id = new_run_id(stage=9)
        print(f"Browser SFT: {new_id}, CPU source: {cpu_run_id}", flush=True)
        result = train_stage9_browser.remote(new_id, config, source, cache_receipt, cpu_run_id, receipt)
        directory = run_path(Path(output_dir), new_id)
        directory.mkdir(parents=True, exist_ok=False)
        files = result.pop("files")
        if set(files) != {*ARTIFACT_FILES, "manifest.json"}: raise ValueError("Unexpected browser SFT artifacts")
        for name, encoded in files.items():
            (directory/name).write_bytes(base64.b64decode(encoded, validate=True))
        replay_saved_run(directory)
        write_json(directory/"browser-sft-result.json", result)
        print(json.dumps({"run_id": new_id, "report": str(directory/"report.html"),
                          "metrics": result["metrics"], "training": result["training"]}, indent=2), flush=True)
        return
    if task in {"browser-check", "browser-baseline"}:
        from vlm_multiturn_rl.browser import BrowserConfig

        if resume_run_id or sft_run_id:
            raise ValueError("Browser lesson starts from the base VLM and does not use grid checkpoints")
        config = asdict(BrowserConfig(seed=seed, episodes=episodes, max_pixels=max_pixels,
                                     max_new_tokens=64 if max_new_tokens == -1 else max_new_tokens))
        source = source_hashes(PROJECT_ROOT)
        if task == "browser-check" and run_id: raise ValueError("New preflights generate a unique run ID")
        cpu_run_id = run_id
        if not cpu_run_id:
            cpu_run_id = new_run_id(stage=8)
            print(f"Browser CPU preflight: {cpu_run_id}", flush=True)
            save_browser_result(check_stage8_browser.remote(cpu_run_id, config, source), output_dir)
        if task == "browser-baseline":
            model_config = {field.name: config[field.name] for field in fields(Stage2Config)}
            receipt = prepare_stage2_model.remote(model_config)
            gpu_run_id = new_run_id(stage=8)
            print(f"Browser GPU baseline: {gpu_run_id}, CPU source: {cpu_run_id}", flush=True)
            save_browser_result(evaluate_stage8_browser.remote(gpu_run_id, config, source, receipt, cpu_run_id), output_dir)
        return
    if task == "trl-grid":
        from vlm_multiturn_rl.stage5 import latest_local_sft
        from vlm_multiturn_rl.stage7 import ARTIFACT_FILES

        if run_id or resume_run_id:
            raise ValueError("Stage 7 creates a fresh one-update integration run from Stage 4 SFT")
        sft_run_id = sft_run_id or latest_local_sft(Path(output_dir))
        config = asdict(Stage7Config(seed=seed, episodes=episodes, train_layouts=train_layouts,
                                    sft_run_id=sft_run_id, max_pixels=max_pixels,
                                    max_new_tokens=64 if max_new_tokens == -1 else max_new_tokens))
        run_id = new_run_id(stage=7)
        print(f"Starting {run_id}: one TRL update", flush=True)
        source = prepare_stage7_source.remote(config)
        model_config = {field.name: config[field.name] for field in fields(Stage2Config)}
        receipt = prepare_stage2_model.remote(model_config)
        print("CPU TRL stack checks:", json.dumps(check_stage7_stack.remote(config)), flush=True)
        result = train_stage7_trl.remote(run_id, config, source_hashes(PROJECT_ROOT), receipt, source)
        directory = run_path(Path(output_dir), run_id)
        directory.mkdir(parents=True, exist_ok=False)
        files = result.pop("files")
        if set(files) != {*ARTIFACT_FILES, "manifest.json"}:
            raise ValueError("Unexpected Stage 7 report artifacts")
        for name, encoded in files.items():
            (directory/name).write_bytes(base64.b64decode(encoded, validate=True))
        verify_artifacts(directory, stage=7, artifact_files=ARTIFACT_FILES)
        write_json(directory/"trl-grid.json", result)
        print(json.dumps({"run_id": run_id, "report": str(directory/"report.html"),
                          "training": result["training"], "metrics": result["metrics"]}, indent=2))
        return
    if task == "train-grid":
        from vlm_multiturn_rl.stage5 import latest_local_sft
        from vlm_multiturn_rl.stage6 import ARTIFACT_FILES

        if run_id:
            raise ValueError("Stage 6 generates a new run ID; use --resume-run-id to resume a checkpoint")
        if not sft_run_id and not resume_run_id:
            sft_run_id = latest_local_sft(Path(output_dir))
        config = asdict(Stage6Config(seed=seed, episodes=episodes, updates=updates, train_layouts=train_layouts,
                                    max_pixels=max_pixels, max_new_tokens=64 if max_new_tokens == -1 else max_new_tokens,
                                    sft_run_id=sft_run_id, resume_run_id=resume_run_id, resume_round=resume_round,
                                    verify_resume=verify_resume, checkpoint_every=checkpoint_every, eval_every=eval_every))
        source_sha256 = source_hashes(PROJECT_ROOT)
        run_id = new_run_id(stage=6)
        print(f"Starting {run_id}", flush=True)
        prepared = prepare_stage6_experiment.remote(config, source_sha256)
        model_config = {field.name: prepared["config"][field.name] for field in fields(Stage2Config)}
        receipt = prepare_stage2_model.remote(model_config)
        result = train_stage6_experiment.remote(run_id, prepared, source_sha256, receipt)
        directory = run_path(Path(output_dir), run_id)
        directory.mkdir(parents=True, exist_ok=False)
        files = result.pop("files")
        if set(files) != {*ARTIFACT_FILES, "manifest.json"}:
            raise ValueError("Unexpected Stage 6 report artifacts")
        for name, encoded in files.items():
            (directory / name).write_bytes(base64.b64decode(encoded, validate=True))
        verify_artifacts(directory, stage=6, artifact_files=ARTIFACT_FILES)
        write_json(directory / "train-grid.json", result)
        print(json.dumps({"run_id": run_id, "report": str(directory / "report.html"),
                          "training": result["training"], "metrics": result["metrics"]}, indent=2))
        return
    if task not in {"write", "read", "inspect-vlm", "evaluate-grid", "sft-grid", "grpo-grid"}:
        raise ValueError("task must be check-gpu, write, read, inspect-vlm, evaluate-grid, sft-grid, grpo-grid, train-grid, or trl-grid")
    if task == "read" and not run_id:
        raise ValueError("read requires --run-id from a previous write job")

    if task in {"write", "inspect-vlm", "evaluate-grid", "sft-grid", "grpo-grid"}:
        if run_id:
            raise ValueError("New experiments generate a unique run ID; omit --run-id")
        # Validate all settings before allocating remote resources.
        if max_new_tokens == -1:
            max_new_tokens = 64 if task in {"sft-grid", "grpo-grid"} else 16
        if task == "write":
            config = config_dict(seed)
            stage = 1
        elif task == "inspect-vlm":
            config = asdict(Stage2Config(seed=seed, max_pixels=max_pixels, max_new_tokens=max_new_tokens))
            stage = 2
        elif task == "evaluate-grid":
            config = asdict(Stage3Config(seed=seed, max_pixels=max_pixels, max_new_tokens=max_new_tokens, episodes=episodes))
            stage = 3
        elif task == "sft-grid":
            config = asdict(Stage4Config(seed=seed, max_pixels=max_pixels, max_new_tokens=max_new_tokens,
                                        episodes=episodes, updates=updates, train_layouts=train_layouts,
                                        gradient_accumulation=gradient_accumulation))
            stage = 4
        else:
            from vlm_multiturn_rl.stage5 import latest_local_sft

            sft_run_id = sft_run_id or latest_local_sft(Path(output_dir))
            config = asdict(Stage5Config(seed=seed, max_pixels=max_pixels, max_new_tokens=max_new_tokens,
                                        episodes=episodes, updates=updates, train_layouts=train_layouts,
                                        sft_run_id=sft_run_id))
            stage = 5
            print(f"Starting from SFT checkpoint {sft_run_id}", flush=True)
        run_id = new_run_id(stage=stage)
        source_sha256 = source_hashes(PROJECT_ROOT)
        print(f"Starting {run_id}", flush=True)
        if task == "write":
            result = write_stage1_run.remote(run_id, config, source_sha256)
        else:
            if task == "grpo-grid":
                source = prepare_stage5_source.remote(config)
            print("Preparing the model cache on CPU before requesting an L40S...", flush=True)
            model_config = {field.name: config[field.name] for field in fields(Stage2Config)}
            receipt = prepare_stage2_model.remote(model_config)
            if task == "inspect-vlm":
                result = inspect_stage2_vlm.remote(run_id, config, source_sha256, receipt)
            elif task == "evaluate-grid":
                result = evaluate_stage3_grid.remote(run_id, config, source_sha256, receipt)
            elif task == "sft-grid":
                result = train_stage4_sft.remote(run_id, config, source_sha256, receipt)
            else:
                result = train_stage5_grpo.remote(run_id, config, source_sha256, receipt, source)
    else:
        run_path(Path(output_dir), run_id)  # Validate before starting the reader.
        result = read_stage1_run.remote(run_id)

    local_directory = run_path(Path(output_dir), run_id)
    local_directory.mkdir(parents=True, exist_ok=True)
    if task in {"inspect-vlm", "evaluate-grid", "sft-grid", "grpo-grid"}:
        if task == "inspect-vlm":
            from vlm_multiturn_rl.stage2 import ARTIFACT_FILES
        elif task == "evaluate-grid":
            from vlm_multiturn_rl.stage3 import GPU_ARTIFACT_FILES as ARTIFACT_FILES
        elif task == "sft-grid":
            from vlm_multiturn_rl.stage4 import GPU_ARTIFACT_FILES as ARTIFACT_FILES
        else:
            from vlm_multiturn_rl.stage5 import ARTIFACT_FILES

        files = result.pop("files")
        if set(files) != {*ARTIFACT_FILES, "manifest.json"}:
            raise ValueError("Unexpected report artifact set")
        for name, encoded in files.items():
            (local_directory / name).write_bytes(base64.b64decode(encoded, validate=True))
        verify_artifacts(local_directory, stage=stage, artifact_files=ARTIFACT_FILES)
    write_json(local_directory / f"{task}.json", result)
    print(json.dumps({
        "run_id": run_id,
        "task": task,
        "seed": result["config"]["seed"],
        "gpu": result["environment"]["gpu"],
        "model_revision": result["config"]["model_revision"],
        "saved_locally": str(local_directory / f"{task}.json"),
        "verification": result.get("verification"),
    }, indent=2))
    if task == "write":
        print(f"Next: uv run modal run modal_app.py --task read --run-id {run_id}")
    elif task == "inspect-vlm":
        print("Observed behavior:", json.dumps(result["inspection"]["behavior"]))
    elif task in {"evaluate-grid", "sft-grid", "grpo-grid"}:
        print("Evaluation metrics:", json.dumps(result["metrics"], indent=2))
        if task == "grpo-grid":
            print("Training:", json.dumps(result["training"], indent=2))

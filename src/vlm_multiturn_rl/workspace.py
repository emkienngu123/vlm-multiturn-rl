"""Stage 1 configuration and artifact helpers; deliberately standard-library only."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
CACHE_PATH = Path("/cache")
RUNS_PATH = Path("/runs")
CACHE_VOLUME_NAME = "vlm-multiturn-rl-cache"
RUNS_VOLUME_NAME = "vlm-multiturn-rl-runs"
ARTIFACT_FILES = ("config.json", "environment.json", "packages.txt", "probe.pt")


@dataclass(frozen=True)
class Stage1Config:
    seed: int = 42
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not re.fullmatch(r"[0-9a-f]{40}", self.model_revision):
            raise ValueError("model_revision must be an immutable 40-character commit SHA")


def new_run_id(stage: int = 1) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"stage{stage}-{timestamp}-{uuid4().hex[:8]}"


def run_path(root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
        raise ValueError("run_id must be a single directory name: letters, numbers, _ or -")
    return root / run_id


def write_json(path: Path, value: dict) -> None:
    # JSON rejects non-native objects (including tensors) instead of pickling them.
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(root: Path) -> dict:
    """Identify the source and dependency inputs used by a local launch."""
    files = [root / "modal_app.py", root / "pyproject.toml", root / "requirements-gpu.in",
             root / "requirements-gpu.txt", *sorted(root.glob("requirements-trl.*")),
             *sorted(root.glob("requirements-browser.*")), *sorted((root/"scripts").glob("*.py")),
             *sorted((root / "src").rglob("*.py"))]
    return {str(path.relative_to(root)): sha256_file(path) for path in files}


def write_manifest(
    directory: Path, metadata: dict, *, stage: int = 1,
    artifact_files: tuple[str, ...] = ARTIFACT_FILES,
) -> dict:
    manifest = {
        **metadata,
        "schema_version": 1,
        "stage": stage,
        "run_id": directory.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": {name: sha256_file(directory / name) for name in artifact_files},
    }
    # Written last: a directory without a manifest is an incomplete run.
    write_json(directory / "manifest.json", manifest)
    return manifest


def verify_artifacts(
    directory: Path, *, stage: int = 1, artifact_files: tuple[str, ...] = ARTIFACT_FILES,
) -> dict:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No completed Stage {stage} run at {directory}")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("stage") != stage
        or manifest.get("run_id") != directory.name
        or set(manifest.get("artifact_sha256", {})) != set(artifact_files)
    ):
        raise ValueError(f"Unexpected or incomplete Stage {stage} manifest")
    for name, expected in manifest["artifact_sha256"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    return manifest


def config_dict(seed: int = 42) -> dict:
    return asdict(Stage1Config(seed=seed))

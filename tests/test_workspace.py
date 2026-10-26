"""CPU checks for the run artifact contract; no torch, numpy, or Modal server."""

import json
from pathlib import Path
import tempfile
import unittest

from vlm_multiturn_rl.workspace import (
    ARTIFACT_FILES,
    Stage1Config,
    config_dict,
    new_run_id,
    run_path,
    verify_artifacts,
    write_json,
    write_manifest,
)


class WorkspaceTests(unittest.TestCase):
    def test_configuration_rejects_unreproducible_inputs(self):
        for seed in [-1, 2**32, True, 1.5]:
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                Stage1Config(seed=seed)
        with self.assertRaises(ValueError):
            Stage1Config(model_revision="main")
        self.assertEqual(config_dict(7)["seed"], 7)
        json.dumps(config_dict())

    def test_run_names_are_unique_and_cannot_escape_root(self):
        self.assertNotEqual(new_run_id(), new_run_id())
        for name in ["", "../other-run", "/runs/other", "a/b", "."]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                run_path(Path("/runs"), name)

    def test_manifest_detects_incomplete_and_corrupt_runs(self):
        with tempfile.TemporaryDirectory() as root:
            directory = run_path(Path(root), new_run_id())
            directory.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "No completed"):
                verify_artifacts(directory)
            for name in ARTIFACT_FILES:
                (directory / name).write_text("test artifact\n")
            manifest = write_manifest(directory, {"source_sha256": {}})
            self.assertEqual(verify_artifacts(directory), manifest)
            (directory / "probe.pt").write_bytes(b"interrupted or corrupt checkpoint")
            with self.assertRaisesRegex(ValueError, "checksum mismatch: probe.pt"):
                verify_artifacts(directory)

    def test_manifest_cannot_silently_omit_an_artifact(self):
        with tempfile.TemporaryDirectory() as root:
            directory = run_path(Path(root), new_run_id())
            directory.mkdir()
            for name in ARTIFACT_FILES:
                (directory / name).write_text("test artifact\n")
            manifest = write_manifest(directory, {})
            del manifest["artifact_sha256"]["config.json"]
            write_json(directory / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "incomplete Stage 1"):
                verify_artifacts(directory)

    def test_metadata_rejects_non_finite_values(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                write_json(Path(root) / "bad.json", {"reward": float("nan")})


if __name__ == "__main__":
    unittest.main()

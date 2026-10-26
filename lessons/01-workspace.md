# Lesson 01 — A reproducible Modal workspace

The goal is to establish that an experiment can run on the GPU, save enough
information to inspect later, exit, and be read by another job. We have not started
VLM inference or RL yet. This lesson provides the foundation for interpreting
those experiments.

By the end, you should be able to explain where code executes, where data lives,
what a seed controls, and what must be recorded to compare training runs.

## 1. Follow one run across the boundaries

```text
Laptop: modal_app.main(task="write")
  | sends a run ID, configuration, and source hashes
  v
L40S container: write_stage1_run -> stage1.create_run
  | reads/writes /cache/huggingface on the cache volume
  | writes /runs/<run-id>/ on the runs volume
  | commits both volumes, then returns plain Python data
  v
Laptop: saves artifacts/<run-id>/write.json; first job ends

Laptop: modal_app.main(task="read", run_id=...)
  v
Different CPU container: read_stage1_run -> stage1.read_run
  | reloads both volumes, verifies artifacts and offline cache
  v
Laptop: saves artifacts/<run-id>/read.json; second job ends
```

`@app.local_entrypoint()` executes on your laptop. Functions decorated with
`@app.function(...)` execute remotely when called with `.remote()`. The Modal
image supplies their dependencies. Source code from our package is explicitly
included using `add_local_python_source`.

This is why the local project depends on Modal while PyTorch belongs in
`requirements-gpu.txt`. Importing the entry point must work without local torch.
The earlier deserialization bug illustrated a second dependency boundary:
returning `torch.__version__` required the laptop to reconstruct a TorchVersion
object. We now return `str(torch.__version__)` and other plain values.

Read the declarations in `modal_app.py` first; then follow the calls into
`src/vlm_multiturn_rl/stage1.py`. Infrastructure stays in the entry point, while
the reusable experiment code stays in the package. See
[Modal's image guide](https://modal.com/docs/guide/images) for the image/source APIs.

## 2. Understand the two volumes

| Data | Remote path | Lifetime and purpose |
| --- | --- | --- |
| Python libraries | Image's Python environment | Installed when the image is built |
| Model metadata, later model weights | `/cache/huggingface` | Reused between runs |
| Configurations, results, later checkpoints | `/runs/<run-id>` | A separate experiment record |
| Unmounted temporary files | Container filesystem | Do not rely on these surviving replacement |
| Returned JSON copies | Local `artifacts/<run-id>` | Convenient inspection on your laptop |

The cache is reusable input; run artifacts describe a particular experiment.
Keeping them separate makes it easier to change one without mixing it into the
other. Every write creates a unique run directory and refuses to overwrite it.

The writer explicitly calls `commit()` before returning. The reader calls
`reload()` before opening files, so it sees committed changes. Reloading with
open files can fail; keep file handles closed around that operation. These are
the relevant [Modal volume semantics](https://modal.com/docs/guide/volumes#volume-commits-and-reloads).

`HF_HOME=/cache/huggingface` directs the Hub cache onto the mounted volume, and is
set before importing Hugging Face libraries. The CPU reader uses offline mode
and `local_files_only=True`: a successful read demonstrates that the needed
metadata survived. [Hub environment variable documentation](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)

## 3. Reproducibility is more than setting a seed

For a learning experiment, think of the result as

\[
Y = F(\text{model revision},\text{data},\text{processor},
      \text{objective},\text{hyperparameters},\text{randomness},\text{numerics}).
\]

Changing any of these may change the outcome. In multi-turn RL, an early sampled
action also changes later observations, so two trajectories can diverge rapidly.
Without an experiment record, an apparent improvement may come from different
tasks, processing, or decoding instead of the learning algorithm.

Stage 1 records:

- An immutable model commit SHA rather than the moving `main` branch.
- A seed and the requested model ID/revision in `config.json`.
- Python, package versions, CUDA runtime, GPU, and processor classes in
  `environment.json`, plus a full installed-package inventory in `packages.txt`.
- Modal image and task IDs. The task ID identifies the container; a Linux hostname
  is not a reliable unique identity across these jobs. See the
  [runtime environment variables](https://modal.com/docs/guide/environment_variables).
- Hashes of the source files and resolved GPU requirements in `manifest.json`.
- Python/NumPy/CUDA random draws and checksums of the persisted artifacts.

Source hashes identify changes; they do not archive the source itself. Keep the
matching source revision in version control when you compare experiments.

`seed_everything` seeds Python, NumPy, and torch; it also requests deterministic
torch algorithms and disables cuDNN benchmarking. `PYTHONHASHSEED` is fixed when
the container starts; assigning it inside a running Python process would be too
late. The seed probe checks that reseeding reproduces draws in this environment.

This does not guarantee identical results across releases, devices, or changes
to execution order. Strict deterministic algorithms may also be slower or reject
an operation. We start strict so unexpected behavior is visible. See
[PyTorch's reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).

Setting a seed at startup is also **not checkpoint/resume**. A future training
checkpoint must include adapter weights, optimizer/scheduler state, RNG states,
and progress through the task stream. Stage 6 will implement that; the tiny
`probe.pt` here only checks the save/load route.

## 4. Run the checks

From the repository root:

```bash
uv sync --locked
uv run vlm-multiturn-rl --seed 42
uv run python -m unittest discover -s tests -v
```

These exercises use the local CPU. The configuration command prints a model ID,
full model SHA, and seed. The tests check invalid configurations, unique run IDs,
incomplete runs, missing manifest entries, corruption, and non-finite metadata.

Now run the existing GPU check:

```bash
uv run modal run modal_app.py
```

Expect `cuda: True` and `gpu: NVIDIA L40S`. Then write a complete Stage 1 run:

```bash
uv run modal run modal_app.py --task write --seed 42
```

This performs a BF16 CUDA matrix multiplication and compares it with an FP32 CPU
calculation using a rounding tolerance. Detecting CUDA alone would not establish
that a real kernel works. The job also imports the Qwen model class, loads its
configuration and processor at the pinned revision, and commits the artifacts.
It does **not** download or instantiate the model weights.

Copy the printed run ID. Wait for that command to finish, then start a separate job:

```bash
uv run modal run modal_app.py --task read --run-id RUN_ID
```

The reader requests no GPU. Its verification should contain:

```json
{
  "artifact_checksums_match": true,
  "tensor_loaded_on_cpu": true,
  "model_metadata_loaded_offline": true,
  "different_container": true
}
```

Open `artifacts/RUN_ID/write.json` and `read.json`. The latter includes the
reader's verification. The persistent remote directory contains:

```text
/runs/RUN_ID/
  config.json
  environment.json
  packages.txt
  probe.pt
  manifest.json
```

The manifest is written last. Checksums detect accidental changes or truncated
artifacts; they do not make several files a single filesystem transaction.
Likewise, committing the cache and run volumes is sequential, not a transaction
across both volumes. The reader independently checks both.

## 5. Why these dependencies are pinned

The initial versions were inspected in the working Modal image. Direct versions
live in `requirements-gpu.in`; the resolved Linux dependencies live in
`requirements-gpu.txt`. `uv.lock` separately fixes the local CLI environment.

To intentionally refresh the GPU resolution after editing the input pins:

```bash
uv pip compile requirements-gpu.in \
  --python-platform x86_64-manylinux_2_28 \
  --python-version 3.12 \
  --only-binary :all: \
  --output-file requirements-gpu.txt
```

This resolves dependencies for the remote Linux machine without installing them
on your Mac. After a change, repeat the GPU writer and independent reader checks.
The per-run package inventory records what actually ran, including runtime
packages. Pins do not freeze the host driver or every aspect of Modal's base image.

This stage checks imports, configuration/processor loading, CUDA computation,
and persistence. It does not yet establish compatibility for full VLM generation,
LoRA backpropagation, or TRL; each later stage adds its own check.

## 6. Common issues and what to inspect

| Symptom | Likely cause or next check |
| --- | --- |
| Local `No module named torch` | A heavy import escaped to module scope, or a returned object needs torch to unpickle. Keep imports remote and metadata plain. |
| Remote `No module named vlm_multiturn_rl` | Run `uv sync --locked`; check `add_local_python_source` and run from the project environment. |
| Downloads repeat | Inspect `environment.hf_home`, the mounted cache volume, and whether the previous writer committed. |
| Offline reader cannot load metadata | Run the writer first; use its run ID and the same Modal workspace/environment and cache volume. |
| Artifact is absent | Confirm `/runs` mount, successful writer completion, correct run ID, and reader reload. Unmounted paths are not experiment storage. |
| `volume busy` during reload | Close files before reloading; do not reload while a model or writer is holding cache files open. |
| Torch/torchvision import failure | Recheck their binary compatibility and the resolved requirements instead of upgrading just one package. |
| Checksum mismatch or missing manifest | Treat the run as incomplete/corrupt; create a new run instead of trusting a partial artifact. |
| First run takes longer | Image construction and metadata downloads precede the small GPU operation. The recorded probe timer excludes provisioning/build time. |
| Same seed, different training result | Compare revisions, data, processor, sampling settings, execution order, numerical settings, and saved RNG states. |

GPU time is bounded by the function timeout and the writer uses a single-use
container. This is a smoke check, not a training benchmark or a dollar-cost cap.

## Exercise — Predict before running

1. Print the local configuration with seeds 42 and 7. Which fields change?
2. Write another remote run with seed 7, then read it in a subsequent job.
3. Compare `manifest.seed_probe` across the two seeds. Predict whether the
   fixed matrix's `probe_sum` should change, then check your prediction.
4. Explain why successfully printing GPU information would not prove that a
   training checkpoint could survive container replacement.

Expected reasoning: the random draws change with the seed, while the matrix uses
`arange`, so its result is independent of the seed. A saved seed selects an initial
RNG state; resuming midway through training requires the current RNG state.

You are ready for Stage 2 when you can locate both persistent volumes, interpret
the manifest, and pass the independent reader check. Next we will load the VLM
weights and inspect how an image becomes model inputs and a generated action.

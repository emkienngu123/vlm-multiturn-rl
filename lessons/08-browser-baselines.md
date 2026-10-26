# Stage 8 — Start visual browser agents with MiniWoB++

We are moving from the grid to **MiniWoB++ through BrowserGym**, following the
browser-agent direction chosen for this tutorial. The previously planned grid
memory/shaping ablation is deferred. This lesson establishes an executable
browser environment, creates demonstrations, and measures the base VLM.
Browser SFT and online GRPO follow in later lessons.

The first subset contains three public tasks:

| Task | Capability | Privileged demonstration length |
| --- | --- | --- |
| `click-test` | Map screenshot coordinates to a visible button | 1 action |
| `enter-text` | Focus a field, type the requested value, submit | 3 actions |
| `login-user` | Fill username and password fields, then submit | 5 actions |

Click-test is a calibration control. The other two tasks require sequential
interaction and changing screenshots. These three tasks are an integration
subset, not a full MiniWoB benchmark and not a realistic-web performance claim.
VisualWebArena remains the later target after this browser learning loop works.

## Run the lesson

```bash
# Local configuration exercise and tests: no Torch, Pillow, Chromium, or BrowserGym.
uv run vlm-multiturn-rl --stage 8 --episodes 2
uv run python -m unittest discover -s tests -v

# CPU-only: validate real browsers, generate demonstrations and random baselines.
uv run modal run modal_app.py --task browser-check --episodes 2

# Reuse that successful CPU report and evaluate the pinned base VLM on one L40S.
uv run modal run modal_app.py --task browser-baseline --episodes 2 --run-id CPU_RUN_ID

# Or run both phases in sequence, preserving the CPU report before starting GPU work.
uv run modal run modal_app.py --task browser-baseline --episodes 2

# Validate the downloaded report and histories locally, without launching a browser.
uv run vlm-multiturn-rl --stage 8 --replay artifacts/RUN_ID
```

`--episodes` is the number of instances **per task, per split**. With 2, there
are six training instances and six evaluation instances. CPU preflight checks
all twelve with the oracle, evaluates a random click policy on the six evaluation
instances, and adds one invalid-action control. Its 19 episodes and demonstrations
are saved before any GPU request. The GPU report includes those episodes and
six base-model evaluation episodes.

The browser CPU function has a 15-minute timeout; the L40S baseline has a
30-minute timeout. Each episode has at most eight actions. The environment's
wall-clock timeout is deliberately large so slow model inference does not
silently become the task's failure criterion. These jobs perform **no browser
training** and save no new adapter.

Open `report.html` for metrics and screenshot/action replays. JSON files retain
all observations, exact generated token IDs, prompt IDs, image grids, parsing
results, rewards, and terminal status. `demonstrations.json` contains only
successful **training-instance oracle actions**, prepared for the SFT bridge.

## 1. Pin the environment, not just the model

`requirements-browser.in` and `requirements-browser.txt` define a separate
remote image. The tested interface targets BrowserGym core/MiniWoB 0.14.3 and
Playwright 1.44.0, with its matching Chromium installed during the image build.
Earlier grid/TRL images retain their own dependency locks.

BrowserGym's setup documentation recommends MiniWoB commit
`7fd85d71a4b60325c6585396ec4f48377d049838`. The build installer downloads exactly
that archive, verifies SHA-256, and extracts it into the image. Tasks use local
`file://` pages; this experiment does not control live websites or personal
browser sessions. Record the task revision, browser version, package versions,
model revision, decoding configuration, source hashes, and task seeds together.
Changing a browser can change fonts, element positions, and coordinate behavior.

The remote wrapper explicitly stops BrowserGym's global synchronous Playwright
runtime before committing Modal volumes. Closing an individual browser context
does not stop that runtime; leaving its event loop alive can interfere with
Modal's own asynchronous I/O and shutdown.

The model remains `Qwen/Qwen2.5-VL-3B-Instruct` at our immutable model revision,
loaded in BF16 from the existing model cache. The grid LoRA adapter is not loaded:
its `move(direction)` policy is a different action interface.

## 2. Define exactly what the policy sees

Read `browser.public_observation`, `initial_history`, and `feedback_message`.
BrowserGym supplies rich observations, including DOM and accessibility data.
We create a new observation from an explicit whitelist:

\[
o_t=(\text{instruction},\ \text{screenshot}_t,\ \text{image dimensions}).
\]

Later turns include the raw previous assistant outputs, public action feedback,
and the next screenshot. The actor receives only these messages and the ordered
image objects. It receives no case IDs, seeds, DOM trees, element IDs, bounding
boxes, task validators, or oracle objects.

This boundary matters. A policy that selects DOM element IDs may solve a task
without learning pixel grounding. That is a valid different experiment, but
would not answer our screenshot-agent question. Hidden DOM information is
available only to the separately labeled privileged demonstration generator.

The policy is still

\[
\pi_\theta(a_t\mid h_t),\qquad
h_t=(\text{instruction},o_0,a_0,o_1,\ldots,o_t).
\]

Typing affects a focused field; clicking a submit button changes terminal state.
An action therefore changes future observations rather than merely extending an
answer string. A password field also hides typed characters, making action
history useful. This lesson does not establish a formal memory benefit; that
requires an explicit observation-history ablation.

## 3. Coordinates and a constrained action protocol

Actions use the same outer JSON tool-call style as the grid, with browser tools:

```text
<tool_call>{"name":"click","arguments":{"x":120,"y":130}}</tool_call>
<tool_call>{"name":"type","arguments":{"text":"example"}}</tool_call>
<tool_call>{"name":"press","arguments":{"key":"Tab"}}</tool_call>
<tool_call>{"name":"scroll","arguments":{"dy":100}}</tool_call>
```

Coordinates refer to the original screenshot's pixels, not normalized 0–1000
coordinates and not the processor's internal resized image. BrowserGym's pinned
MiniWoB wrapper captures a screenshot at scale **1.5** and maps mouse coordinates
back to CSS page pixels by dividing by that scale. A 332×214 viewport normally
produces a 498×321 screenshot. The privileged oracle multiplies CSS bounding-box
centers by 1.5, then executes the same coordinate action as the policy. Successful
oracle clicks validate the mapping rather than assuming the scale is harmless.

The model gets the original width and height as text. Qwen's processor may
internally resize the image to form visual tokens; the public coordinate frame
does not change. We preserve aspect ratio and the full screenshot.

`parse_action` accepts exactly one complete call, validates keys and arguments,
rejects duplicate JSON keys and nonfinite/out-of-bounds coordinates, and limits
text length and keyboard keys. The wrapper constructs an allowed BrowserGym
call from parsed values using quoted string arguments. **Raw model output is
never executed as Python.**

Invalid actions become a zero-wait no-op and consume an action. Execution errors
also consume actions and are reported as a boolean. Detailed internal exception
messages are kept out of policy feedback because they can reveal selectors.
`scroll(dy)` currently acts at the screenshot center; the selected oracle tasks
do not need scrolling. More complex scrollable tasks will need an explicit
pointer location and their own validation before adding them to the task set.

## 4. Reward and termination need validation

The selected tasks use raw rewards −1 for wrong submission, 0 while unfinished,
and +1 for successful completion. BrowserGym converts positive raw reward to 1
and other values to 0. We keep that native sparse binary reward:

\[
r_t=\mathbb{1}(\text{successful task completion at }t),\qquad
R=\sum_t r_t\in\{0,1\}.
\]

There is no step penalty or distance shaping in this browser lesson. The eight
action cap is a truncation; an actual successful or failed submission is
termination. We retain both raw reward and trainer-facing reward in artifacts.

An important selection detail: BrowserGym's wrapper maps **any** positive raw
reward to success, and some MiniWoB tasks provide partial credit. For example,
a checkbox task can award a positive fraction for an imperfect selection. We
avoid those tasks initially and assert that selected tasks have binary raw
rewards. Adding a new task requires reading its validator and defining what
success means before using its reward for RL.

The CPU job explicitly verifies that empty login submission fails, an ended
environment refuses another action, and eight malformed outputs consume the
full cap without success.

## 5. Demonstrations, baselines, and splits

Training instances use seeds 0 and 1 by default; evaluation uses 10000 and 10001.
Every policy uses the same evaluation task/seed pairs. The policy randomness
seed is recorded separately. These are held-out **instances of the same task
families**, not held-out task types. Goal text can recur in a task's finite
vocabulary; new seeds do not establish language or task-family generalization.

The oracle reads public instruction values and privileged DOM geometry, then
executes ordinary click/type actions. It never fills fields directly or changes
the task's success flag. Its episodes are labeled `oracle`. This supplies a
trustworthy reward/coordinate check and a small SFT dataset; oracle performance
is not a VLM result. Evaluation oracle actions are excluded from SFT targets.

The random baseline samples pixel clicks. It is a click-only reference, not a
uniform baseline over all action types. Report results per task as well as in
aggregate, so easy click calibration does not hide failure on text workflows.

The base VLM uses full image history and greedy decoding with 64 tokens per
action. Its output is parsed without repairing a malformed response. A context
guard fails explicitly rather than silently removing older images.

## 6. Two different meanings of replay

The remote jobs **re-execute** oracle and base-model actions in fresh browsers
with the same task seeds. They require the same initial instruction and initial
image hash, then compare executed actions, rewards, and termination flags.
They replay saved actions, not model sampling. Later screenshot bytes may vary
with caret blinking, so later pixel equality is not the criterion.

The local `--replay` command checks artifact hashes, histories, action parsing,
reward arithmetic, final observations, metrics, and demonstration split rules.
It does not launch Chromium and cannot independently prove browser dynamics.
These checks complement each other.

## Common failures

| Symptom | What to inspect |
| --- | --- |
| Clicks land consistently away from targets | Screenshot pixels versus CSS pixels; the 1.5 scale and processor coordinate convention |
| VLM appears strong without reading the screenshot | DOM/accessibility leakage, element IDs, oracle data in prompts |
| Correct text ends up in the wrong field | Focus is environment state; verify the preceding click and screenshot |
| Correct password appears missing | Password fields mask characters; inspect prior actions and final task reward |
| High reward with an incorrect solution | Task-specific partial credit and BrowserGym's positive-reward conversion |
| Latency causes apparent task failure | Wall-clock task timer versus action-count cap |
| Event-loop errors when a Modal job returns | Stop the global synchronous Playwright runtime before volume commits and worker shutdown |
| Oracle fails | Treat as an environment/coordinate problem before attributing failure to the model |
| Base model emits code, explanations, or grid moves | New action protocol; measure parsing and grounding before RL, then use browser SFT |
| Training reward rises but broader tasks fail | Same-family seed generalization is weaker than unseen-task-type evaluation |
| Exact old grid resume fails after adding this code | Source hashes changed; use the checkpoint's original source snapshot for exact resume |

## Exercise and next checkpoint

Open one `enter-text` or `login-user` replay. Match each assistant target in
`demonstrations.json` to its preceding screenshot. Confirm that no target
contains an environment response, DOM selector, or evaluation-instance case.
Explain why typing is a separate transition from focusing the field, and why
the final screenshot must be saved even after task termination.

The next lesson will train a **browser-specific** LoRA adapter with assistant-only
SFT, compare it with this base baseline, and save/reload it. Only then will we
run grouped online GRPO against live browser states. Maintain separate train,
validation, and final test instances when extending these smoke experiments.

## Observed first runs

The completed CPU report is `stage8-20261019T034403Z-a72e8d1c`; the GPU baseline
is `stage8-20261019T034323Z-fd28372a`. Both use two instances per task and split,
seed 42 for policy randomness, and the same pinned task/browser stack. Actual
Chromium version was 125.0.6422.26. All 55 local tests pass.

| Policy | Click-test | Enter-text | Login-user | Total held-out success |
| --- | --- | --- | --- | --- |
| Privileged oracle | 2/2 | 2/2 | 2/2 | 6/6 |
| Random pixel clicks | 1/2 | 0/2 | 0/2 | 1/6 |
| Base Qwen2.5-VL-3B | 0/2 | 0/2 | 0/2 | 0/6 |

The oracle also solved all six training instances, yielding **18 assistant-action
training targets**. Those training instances do not enter the metrics table.
All oracle action sequences re-executed successfully in fresh browsers, and the
saved base sequences were also re-executed without model regeneration. Local
artifact checks validate 19 CPU episodes and 25 episodes in the GPU report.

The base model's invalid-action rate was 100%. For example, its raw response
was `click(x=100, y=80)` rather than the required JSON tool call. We preserved
that output and counted it as invalid. These results diagnose protocol
noncompliance; they do **not** measure the model's visual-grounding accuracy
under a working action interface. They also do not establish that random clicks
are generally better than a trained VLM agent.

The GPU function took about **83 seconds**, peaking at **7.29 GiB allocated GPU
memory**, excluding image build, CPU preflight, and model-cache preparation.
No browser adapter or RL update was trained. The next checkpoint is the
browser-specific SFT bridge, followed by online RL with the same evaluator.

## Primary sources

- [BrowserGym MiniWoB setup and pinned task revision](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/miniwob/README.md)
- [MiniWoB++ documentation](https://miniwob.farama.org/)
- [BrowserGym project and environment interface](https://github.com/ServiceNow/BrowserGym)
- [Pinned task source](https://github.com/Farama-Foundation/miniwob-plusplus/tree/7fd85d71a4b60325c6585396ec4f48377d049838/miniwob/html/miniwob)

The installed wheel and saved dependency lock define the tested runtime; current
repository `main` may describe newer behavior.

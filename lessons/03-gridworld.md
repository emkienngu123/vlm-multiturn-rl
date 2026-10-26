# Lesson 03 — Build an environment before optimizing its policy

Stage 2 checked a VLM inference call. Stage 3 connects that call to a world:
**observe → choose one action → change the state → observe again**.
We first establish correct dynamics, rewards, evaluation tasks, and baselines.
No SFT or RL update happens in this lesson.

The environment and CPU exercises use only Python's standard library. The VLM
baseline reuses the cached model, processor, BF16/SDPA settings, and L40S from
Stage 2.

## 1. The task and its exact rules

Every task is a 4×4 board with a blue agent circle, green goal square, three dark
wall cells, and light empty cells. Coordinates in diagnostic data are `(row,
column)`, measured from the upper left.

| Event | State change | Reward | Budget |
| --- | --- | --- | --- |
| Valid move to an empty cell | Move one cell | −0.01 | Consumes one action |
| Move into a wall or outside the board | Stay in place | −0.01 | Consumes one action |
| Malformed answer | Stay in place | −0.01 | Consumes one action |
| Move onto the goal | Move to goal; terminate | +0.99 | Consumes one action |
| Eight actions without reaching goal | Stop collecting; truncate | No additional reward | Episode ends |

The goal contributes +1, and **every action**, including the successful one,
costs 0.01. Thus a successful episode of length `T` returns `1 − 0.01T`; an
unsuccessful eight-action episode returns `−0.08`. There is no distance-based
shaping and no extra penalty for invalid output.

Reaching the goal on action eight counts as success: `terminated=True` and
`truncated=False`. Further calls to `step()` raise an error until `reset()`.

The strict Stage 2 parser still accepts only a direction word, with whitespace
and case normalization. An explanation or a list of moves is not silently
converted into an action. We count malformed outputs separately from collisions:
`invalid_actions` measures syntax; `blocked_moves` measures valid directions
that cannot move the agent.

## 2. Run the CPU version first

```bash
uv run python -m unittest discover -s tests -v
uv run vlm-multiturn-rl --stage 3 --episodes 4
```

The second command generates the task suite, evaluates random and shortest-path
policies, checks every trajectory by replaying it, and saves a report. It makes
no Modal call and needs neither local PyTorch nor Pillow.

The printed run directory contains `replay.html`. Open it in a browser, expand
an episode, and scroll its frames from left to right. Each frame includes the
action, reward, collision status, and termination/truncation flags.

Recheck a saved run without generating another experiment:

```bash
uv run vlm-multiturn-rl --stage 3 --replay artifacts/RUN_ID
```

Replace `RUN_ID` with the printed directory name. This verifies artifact checksums
and re-executes recorded actions against the recorded layouts. It checks
**environment dynamics**, not whether a VLM would generate the same tokens again.

Read `gridworld.py` in this order: `Layout`, `destination`, `shortest_path`,
`GridWorld`, `run_episode`, and `replay_episode`. Read `stage3.py` afterward for
task pairing, metrics, and the image-only VLM adapter.

## 3. State, observation, action, transition

For the underlying goal-reaching task, write

\[
s_t=(\text{agent position},\text{wall map},\text{goal position}),\qquad
o_t=\operatorname{render}(s_t).
\]

An action is a cardinal direction, and the transition function either moves the
agent or leaves it in place. The map is static within an episode, and the core
transition is deterministic.

The VLM receives the current rendered image and a fixed instruction describing
the colors and valid directions. The adapter's interface is `actor(image)`.
The collector does not pass the environment object, coordinates, task ID,
shortest path, expected next action, or reward labels to that adapter.

The evaluator saves privileged state in the trace so we can debug and replay
it. That does not make it a policy input. This separation matters: a model given
textual coordinates might solve the task without using vision.

This version shows the full board and uses **current-image-only inference**.
The full visual state is available; history is not inherently required to choose
a good move. Each action still changes the next observation, so this is a
multi-step interaction rather than generating an entire plan in one response.
Stage 4 will introduce explicit conversation histories; Stage 8 will make the
observation partially visible so memory becomes necessary.

For a fixed layout `L`, a trajectory factorizes as

\[
p_\theta(\tau\mid L)=\prod_t
\pi_\theta(a_t\mid o_t,\text{instruction})
P(s_{t+1}\mid s_t,a_t).
\]

Here `P` is deterministic. Once we record the layout and actions, we can replay
the states and rewards without the model. This is why we can test most of this
stage on CPU before paying for inference.

## 4. Why use a shortest-path solver?

The graph has at most 16 cells and four outgoing directions per cell. Breadth-first
search gives a shortest path because every move has the same cost. We use a fixed
neighbor order to resolve ties reproducibly.

The solver has three jobs:

- Reject unreachable generated tasks.
- Restrict initial shortest-path lengths to two through six, inside the
  eight-action collection budget.
- Supply a privileged oracle and a path-length reference for evaluation.

The oracle reads the true state. It is an upper-bound and infrastructure check,
not evidence that a visual policy learned the task. If it cannot solve every
generated evaluation task, investigate the generator, transitions, stopping rules,
or solver before training.

The random policy samples uniformly from all four directions. It does not get
an action mask. Neither does the VLM. Giving random only legal moves would change
the comparison by granting it extra information.

## 5. Make evaluation repeatable without leaking training mazes

The generator owns a local RNG, separate from policy randomness. `dataset_seed`
defaults to 2026 and defines the task suite; the ordinary `seed` controls policy
randomness. Changing the policy seed therefore does not change the tasks.

Maze identity hashes the board size and wall locations. That identity assigns
the whole maze to train or evaluation. Moving the agent or goal does not allow
the same walls to cross the split boundary. Full task identity additionally
includes the start and goal.

The saved suite contains eight training layouts, unused in this stage, plus the
requested evaluation layouts. Each split contains distinct wall maps. Increasing
the evaluation count extends the deterministic prefix, and all policies see
exactly the same evaluation layouts.

This is a procedural held-out split, not evidence of broad out-of-distribution
generalization. Later experiments must deliberately vary things like wall
density, board size, colors, or path length to support those claims.

## 6. Termination and truncation are different facts

We define reaching the goal as a terminal state of the underlying task. The
eight-action cap is an **external collection limit**, recorded as truncation.
The state at the cap might still have a path to the goal.

This choice is explicit. If instead we defined “solve within eight actions” as
an intrinsic finite-horizon MDP, the horizon would be part of that task and
remaining time would need to be represented in the state/observation.

The distinction affects value targets. For a continuing-task critic, a common
target is

\[
y_t=r_t+\gamma(1-\text{terminated}_t)V(s_{t+1}),
\]

so a collection cutoff alone need not zero the bootstrap term. This stage uses
no critic or bootstrapping. Our reported score is simply return and success
within the collection cap. Later critic-free GRPO will optimize an explicitly
bounded-trajectory score; it will not silently claim that score estimates an
unbounded-task value. See
[Gymnasium's time-limit guide](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

## 7. Add the untrained VLM baseline

```bash
uv run modal run modal_app.py --task evaluate-grid --episodes 4
```

The CPU preparation job reuses/populates the pinned snapshot cache. Then one
L40S evaluates random, oracle, and VLM policies on the same layouts. The GPU job
is bounded to 15 minutes; four tasks allow at most 32 VLM action generations.
Weights are loaded once and remain unchanged.

The VLM uses greedy decoding, a fresh current-image prompt on each action, and
the Stage 2 per-call image processing interface. Prompt IDs and grid information
must stay aligned with the actual image. Exact generated token IDs, output text,
token counts, stopping reason, and generation duration accompany each VLM action.

No successful-action correction, oracle fallback, or legal-move masking is
applied to the VLM. If it repeats a blocked move, the trace records that loop.
On an identical observation, a deterministic greedy policy can repeat the same
mistake until truncation. Later exploration and SFT can address this; neither is
secretly enabled in the baseline.

## 8. Read the comparison and replay

| Metric | Meaning |
| --- | --- |
| Success rate | Fraction reaching the goal before collection stops |
| Mean return | Sum of the documented rewards, averaged over episodes |
| Mean actions | Includes malformed, blocked, and successful actions |
| Mean path efficiency | Shortest length / actions on success; zero on failure |
| Invalid-action rate | Malformed outputs / attempted actions |
| Blocked-move rate | Wall or boundary collisions / attempted actions |

An oracle should score 100% success, efficiency 1, and no invalid/blocked moves.
Other policies may fail. Four layouts are a smoke test, not a reliable ranking:
one extra success changes the reported rate by 25 percentage points. Use the
paired episodes to understand errors before making statistical claims.

The validated default run (`dataset_seed=2026`, policy `seed=42`, four evaluation
layouts) produced:

| Baseline | Successes | Mean return | Blocked-move rate |
| --- | --- | --- | --- |
| Random | 1/4 | 0.1850 | 38.5% |
| Shortest-path oracle | 4/4 | 0.9775 | 0% |
| Untrained VLM | 0/4 | −0.0800 | 93.8% |

The VLM answered `left` on all 32 attempts. All outputs passed the direction-word
parser, but most collided with a wall or boundary. This identifies a valid-output,
poor-action baseline; it does not diagnose the underlying perception or decision
error by itself. Every failed VLM episode also had the same return, illustrating
why sparse-reward GRPO may need a better initialization before it gets a useful
group-relative signal. No policy was repaired or trained to improve these numbers.

The independent CPU run reproduced the same layouts and random/oracle metrics.
All 12 episodes in the GPU comparison passed dynamics replay; 21 CPU tests cover
the environment and earlier-stage contracts.

Saved artifacts include:

- `layouts.json`: the train/eval suite and stable identities.
- `episodes.json`: raw answers, parsed actions, before/after states, rewards,
  flags, and per-action model diagnostics where applicable.
- `metrics.json`: aggregated results and enforced checks.
- `replay.html`: self-contained SVG reconstructions of recorded states.
- `config.json`, `environment.json`, `manifest.json`: settings, runtime details,
  source hashes, and file checksums.
- For GPU runs, `model-cache.json` and `preview.png`: the cache receipt and an
  actual raster observation. HTML frames are reconstructed views; rich image
  and conversation datasets arrive in Stage 4.

`--replay` checks each state, parsed action, reward, flag, final summary, and stop
condition. It rejects changed transitions and incomplete episodes. All baseline
episodes are replay-checked automatically before a report is marked complete.

## 9. Problems worth catching now

| Symptom | Likely cause or diagnostic |
| --- | --- |
| Oracle fails | Check reachability, wall collisions, reward timing, and whether the cap is shorter than the optimal path. |
| An episode never ends after invalid answers | Invalid attempts must consume budget; otherwise a malformed-output loop can run forever. |
| Unexpectedly high visual performance | Inspect the actual policy prompt for coordinates, solver actions, or target labels. |
| Random baseline changes after editing a different policy | Keep dataset and policy RNGs isolated instead of sharing a global stream. |
| Same wall map appears in train and evaluation | Split by maze geometry, not merely row number, RNG seed, or start/goal pair. |
| Replay shows every state as the final state | Save copies of mutable state; do not append the same mutable object repeatedly. |
| Terminal goal receives −0.01 only | Add the goal reward and step cost on the same successful transition. |
| Repeated invalid or blocked VLM actions | Preserve the behavior; inspect model formatting, visual grounding, and greedy loops separately. |
| All failed GRPO trajectories have identical returns | With a fixed cap and these rewards, every full-length failure returns −0.08. This can create zero-variance groups; a usable initialization and exploration matter. |

## Exercise — Predict an episode before executing it

1. On a layout whose start is in the upper-left corner, predict what an `up`
   action does to position, reward, and action count. Compare a malformed answer.
2. For a six-action shortest path, compute the oracle return. Predict the
   termination/truncation flags if the goal is reached on the last allowed action.
3. Run the CPU suite with `--seed 7`. Compare task IDs and random-policy actions
   with the default-seed run. Which should remain fixed?
4. Open a failed VLM episode. Classify each failed attempt as malformed output,
   collision, or a valid but unhelpful move. Do not infer all three from reward alone.
5. Explain why training on all-failure groups may produce no GRPO signal even
   though the trajectories contain different actions.

You are ready for Stage 4 when the oracle solves every generated task, you can
replay the saved transitions, and you can explain which information each baseline
receives. Next we will collect structured multi-turn histories and generate solver
demonstrations for a small assistant-only SFT bridge in
[Lesson 04](04-trajectories-sft.md).

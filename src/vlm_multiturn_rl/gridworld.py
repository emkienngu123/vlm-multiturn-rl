"""Deterministic grid dynamics and privileged diagnostics; standard library only."""

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import random

from .vision import IMAGE_SIDE, parse_action


DIRECTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
GRID_PROMPT = (
    "This is a top-down 4 by 4 board. The blue circle is your agent, the green "
    "square is the goal, dark filled cells are walls, and light cells are empty. "
    "Choose the next one-cell move toward the goal while avoiding walls and "
    "the board edge. Directions refer to the image: up is toward the top, down "
    "toward the bottom, left toward the left edge, right toward the right edge. "
    "Reply with exactly one word: up, down, left, or right. Do not explain."
)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Layout:
    start: tuple[int, int]
    goal: tuple[int, int]
    walls: tuple[tuple[int, int], ...] = ()
    size: int = 4

    def __post_init__(self):
        if type(self.size) is not int or self.size < 2:
            raise ValueError("size must be an integer >= 2")
        if not isinstance(self.walls, tuple) or any(not isinstance(p, tuple) for p in self.walls):
            raise ValueError("walls must be an immutable tuple of coordinate tuples")
        for position in (self.start, self.goal, *self.walls):
            if not isinstance(position, tuple) or len(position) != 2 or any(
                type(x) is not int or not 0 <= x < self.size for x in position
            ):
                raise ValueError("positions must be in-bounds (row, column) tuples")
        if self.start == self.goal or self.start in self.walls or self.goal in self.walls:
            raise ValueError("start and goal must be distinct, unblocked cells")
        if len(set(self.walls)) != len(self.walls):
            raise ValueError("wall coordinates must be unique")
        object.__setattr__(self, "walls", tuple(sorted(self.walls)))

    @property
    def maze_id(self) -> str:
        return _digest({"size": self.size, "walls": self.walls})

    @property
    def layout_id(self) -> str:
        return _digest(asdict(self))

    @property
    def split(self) -> str:
        # Hash maze geometry, not start/goal: different tasks in the same maze
        # cannot cross the train/eval boundary, even with a different dataset seed.
        return "eval" if int(self.maze_id[:8], 16) % 5 == 0 else "train"

    def to_dict(self) -> dict:
        # JSON round-trip converts coordinates to lists consistently everywhere.
        return json.loads(json.dumps({
            **asdict(self), "layout_id": self.layout_id, "maze_id": self.maze_id, "split": self.split,
        }))

    @classmethod
    def from_dict(cls, value: dict):
        layout = cls(tuple(value["start"]), tuple(value["goal"]),
                     tuple(tuple(p) for p in value["walls"]), value["size"])
        if value.get("layout_id", layout.layout_id) != layout.layout_id:
            raise ValueError("layout_id does not match layout geometry")
        if value.get("maze_id", layout.maze_id) != layout.maze_id or value.get("split", layout.split) != layout.split:
            raise ValueError("Maze identity or split does not match geometry")
        return layout


def destination(layout: Layout, position: tuple[int, int], action: str) -> tuple[int, int]:
    dr, dc = DIRECTIONS[action]
    target = (position[0] + dr, position[1] + dc)
    if any(x < 0 or x >= layout.size for x in target) or target in layout.walls:
        return position
    return target


def shortest_path(layout: Layout, position: tuple[int, int] | None = None) -> list[str] | None:
    """Breadth-first search; None means unreachable, [] means already at the goal."""
    start = layout.start if position is None else position
    frontier = deque([start])
    parents = {start: None}
    while frontier:
        current = frontier.popleft()
        if current == layout.goal:
            actions = []
            while parents[current] is not None:
                previous, action = parents[current]
                actions.append(action)
                current = previous
            return actions[::-1]
        for action in DIRECTIONS:  # Fixed order makes shortest-path tie-breaking reproducible.
            target = destination(layout, current, action)
            if target not in parents:
                parents[target] = (current, action)
                frontier.append(target)
    return None


def make_layouts(split: str, count: int, dataset_seed: int = 2026) -> list[Layout]:
    if split not in {"train", "eval"} or type(count) is not int or not 1 <= count <= 32:
        raise ValueError("Request train/eval and between 1 and 32 layouts")
    rng = random.Random(f"grid-v1:{dataset_seed}:{split}")
    cells = [(r, c) for r in range(4) for c in range(4)]
    layouts, seen_mazes = [], set()
    for _ in range(100_000):
        chosen = rng.sample(cells, 5)
        layout = Layout(chosen[0], chosen[1], tuple(chosen[2:]))
        if layout.split != split or layout.maze_id in seen_mazes:
            continue
        path = shortest_path(layout)
        if path is None or not 2 <= len(path) <= 6:
            continue
        layouts.append(layout)
        seen_mazes.add(layout.maze_id)
        if len(layouts) == count:
            return layouts
    raise RuntimeError("Could not generate the requested distinct, reachable mazes")


@dataclass(frozen=True)
class StepResult:
    reward: float
    terminated: bool
    truncated: bool
    valid_action: bool
    blocked: bool


class GridWorld:
    """Core transitions are CPU-only. State access is privileged, not a policy input."""

    def __init__(self, layout: Layout, max_steps: int = 8):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if shortest_path(layout) is None:
            raise ValueError("The goal is unreachable")
        self.layout = layout
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> None:
        self.position = self.layout.start
        self.steps = 0
        self.terminated = self.truncated = False

    def step(self, action: str | None) -> StepResult:
        if self.terminated or self.truncated:
            raise RuntimeError("Episode ended; reset before stepping again")
        valid = isinstance(action, str) and action in DIRECTIONS
        next_position = destination(self.layout, self.position, action) if valid else self.position
        blocked = valid and next_position == self.position
        self.position = next_position
        self.steps += 1
        self.terminated = self.position == self.layout.goal
        # The cap is an external collection limit on a goal-reaching task.
        # Goal arrival on the last allowed action takes precedence over truncation.
        self.truncated = self.steps >= self.max_steps and not self.terminated
        return StepResult(
            reward=(1.0 if self.terminated else 0.0) - 0.01,
            terminated=self.terminated, truncated=self.truncated,
            valid_action=valid, blocked=blocked,
        )


def render_observation(layout: Layout, position: tuple[int, int]):
    from .vision import render_grid

    return render_grid(position, layout.goal, layout.size, layout.walls)


def episode_seed(seed: int, layout: Layout) -> int:
    return int(_digest({"seed": seed, "layout_id": layout.layout_id})[:8], 16)


def run_episode(
    layout: Layout, policy: str, *, seed: int = 42, max_steps: int = 8,
    actor=None, renderer=render_observation,
) -> dict:
    if policy not in {"random", "oracle", "vlm"} or (policy == "vlm" and actor is None):
        raise ValueError("Choose random, oracle, or vlm with an image-only actor")
    env = GridWorld(layout, max_steps)
    rng_seed = episode_seed(seed, layout)
    rng = random.Random(rng_seed)
    transitions = []
    while not (env.terminated or env.truncated):
        if policy == "random":
            decision = {"text": rng.choice(tuple(DIRECTIONS))}
        elif policy == "oracle":
            decision = {"text": shortest_path(layout, env.position)[0]}
        else:
            # Only the rendered image crosses this boundary, never the env,
            # coordinates, goal label, shortest path, or diagnostic transition.
            decision = actor(renderer(layout, env.position))
        text = decision["text"]
        action = parse_action(text)
        before = list(env.position)
        outcome = env.step(action)
        transitions.append({
            "step": env.steps, "before": before, "text": text, "action": action,
            "after": list(env.position), **asdict(outcome),
            "generation": {k: v for k, v in decision.items() if k != "text"},
        })
    best = len(shortest_path(layout))
    return {
        "layout_id": layout.layout_id, "policy": policy, "seed": rng_seed,
        "max_steps": max_steps, "initial_position": list(layout.start),
        "transitions": transitions, "steps": env.steps, "success": env.terminated,
        "end_reason": "goal" if env.terminated else "step_limit",
        "return": round(sum(t["reward"] for t in transitions), 12),
        "shortest_path_length": best,
        "path_efficiency": best / env.steps if env.terminated else 0.0,
        "invalid_actions": sum(not t["valid_action"] for t in transitions),
        "blocked_moves": sum(t["blocked"] for t in transitions),
    }


def replay_episode(layout: Layout, episode: dict, *, action_parser=parse_action) -> None:
    """Check deterministic dynamics from recorded actions, not model reproducibility."""
    if episode["layout_id"] != layout.layout_id or episode["initial_position"] != list(layout.start):
        raise ValueError("Episode does not match its initial layout")
    env = GridWorld(layout, episode["max_steps"])
    for record in episode["transitions"]:
        if record["before"] != list(env.position) or record["action"] != action_parser(record["text"]):
            raise ValueError("Replay state/action mismatch")
        outcome = env.step(record["action"])
        if (record["after"] != list(env.position) or record["step"] != env.steps
                or any(record[key] != value for key, value in asdict(outcome).items())):
            raise ValueError("Replay transition mismatch")
    expected = {
        "steps": env.steps, "success": env.terminated,
        "end_reason": "goal" if env.terminated else "step_limit",
        "return": round(sum(t["reward"] for t in episode["transitions"]), 12),
        "shortest_path_length": len(shortest_path(layout)),
        "path_efficiency": len(shortest_path(layout)) / env.steps if env.terminated else 0.0,
        "invalid_actions": sum(not t["valid_action"] for t in episode["transitions"]),
        "blocked_moves": sum(t["blocked"] for t in episode["transitions"]),
    }
    if not (env.terminated or env.truncated) or any(episode[key] != value for key, value in expected.items()):
        raise ValueError("Replay episode summary or stopping condition mismatch")


def summarize(episodes: list[dict]) -> dict:
    if not episodes:
        raise ValueError("Cannot summarize zero episodes")
    n, steps = len(episodes), sum(e["steps"] for e in episodes)
    return {
        "episodes": n, "successes": sum(e["success"] for e in episodes),
        "success_rate": sum(e["success"] for e in episodes) / n,
        "mean_return": sum(e["return"] for e in episodes) / n,
        "mean_steps": steps / n,
        "mean_path_efficiency": sum(e["path_efficiency"] for e in episodes) / n,
        "invalid_action_rate": sum(e["invalid_actions"] for e in episodes) / steps,
        "blocked_move_rate": sum(e["blocked_moves"] for e in episodes) / steps,
    }


def render_svg(layout: Layout, position: list[int] | tuple[int, int]) -> str:
    """CPU-only visual replay; geometry mirrors the observation renderer."""
    cell = IMAGE_SIDE / layout.size
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {IMAGE_SIDE} {IMAGE_SIDE}" '
             'role="img" aria-label="Grid board"><rect width="100%" height="100%" fill="#f1f5f9"/>']
    for i in range(layout.size + 1):
        p = min(round(i * cell), IMAGE_SIDE - 1)
        parts.extend([f'<path d="M {p} 0 V {IMAGE_SIDE}" stroke="#475569" stroke-width="3"/>',
                      f'<path d="M 0 {p} H {IMAGE_SIDE}" stroke="#475569" stroke-width="3"/>'])
    for row, col in layout.walls:
        parts.append(f'<rect x="{col*cell+3}" y="{row*cell+3}" width="{cell-5}" '
                     f'height="{cell-5}" fill="#334155"/>')
    row, col = layout.goal
    parts.append(f'<rect x="{(col+.22)*cell}" y="{(row+.22)*cell}" width="{.56*cell}" '
                 f'height="{.56*cell}" fill="#22c55e" stroke="#166534" stroke-width="3"/>')
    row, col = position
    parts.append(f'<circle cx="{(col+.5)*cell}" cy="{(row+.5)*cell}" r="{.28*cell}" '
                 'fill="#2563eb" stroke="#1e3a8a" stroke-width="3"/></svg>')
    return "".join(parts)

"""The grid/TRL boundary. Importing this module requires only the standard library."""

from copy import deepcopy
from dataclasses import asdict
import json

from .gridworld import GridWorld, Layout, episode_seed, shortest_path
from .trajectories import initial_messages, parse_move, replay_trajectory, tool_message


# Preserve the Stage 4 role/image syntax and raw sampled text. The tool_calls
# branch also supports TRL's synthetic template-capability/prefix checks.
CHAT_TEMPLATE = """{%- for message in messages %}
{{- '<|im_start|>' + message['role'] + '\n' }}
{%- set ns = namespace(written=false) %}
{%- if message['content'] is string %}{{- message['content'] }}{%- set ns.written = message['content'] | length > 0 %}
{%- else %}{%- for part in message['content'] %}
{%- if part['type'] == 'image' %}{{- '<|vision_start|><|image_pad|><|vision_end|>' }}
{%- elif part['type'] == 'text' %}{{- part['text'] }}{%- if part['text'] %}{%- set ns.written = true %}{%- endif %}{%- endif %}
{%- endfor %}{%- endif %}
{%- if message.get('tool_calls') and not ns.written %}
{%- for call in message['tool_calls'] %}{{- '<tool_call>' }}{{- call['function'] | tojson }}{{- '</tool_call>' }}{%- endfor %}
{%- endif %}{{- '<|im_end|>\n' }}
{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"""


def dispatch_response(text: str) -> dict:
    """Dispatch every sampled turn once; malformed text remains an invalid move.

    The sentinel is an internal lifecycle event, never a repaired policy action.
    The raw content and exact generated IDs remain the policy history/targets.
    """
    direction = parse_move(text)
    return {"role": "assistant", "content": text, "tool_calls": [{
        "type": "function", "function": {"name": "move", "arguments": {
            "direction": direction if direction is not None else "__invalid__"}}}]}


class GridToolEnvironment:
    """A distinct instance per concurrent rollout; TRL may reset and reuse it."""

    def __init__(self, *, renderer, assets, episodes, image_owners, next_key, seed=42):
        self._renderer, self._assets, self._episodes = renderer, assets, episodes
        self._owners, self._next_key, self._seed = image_owners, next_key, seed

    def reset(self, layout: str, **kwargs):
        """Receive private dataset metadata and return only the initial image."""
        if hasattr(self, "_images"):
            for image in self._images:
                self._owners.pop(id(image), None)
        self._layout = Layout.from_dict(json.loads(layout))
        self._env = GridWorld(self._layout, max_steps=8)
        self._key = self._next_key()
        self._pending = None
        self._saved = False
        self._images, self._observations, self._transitions = [], [], []
        path, image = self._observe()
        self._history = initial_messages(path)
        # The dataset prompt already contains the Stage 4 system/user text.
        return [{"type": "image", "image": image},
                {"type": "text", "text": "Choose your next move."}]

    def _observe(self):
        path = f"images/{self._key}/{self._env.steps:03d}.png"
        if path in self._assets:
            raise ValueError("Rollout keys must be unique, including after reset")
        image = self._renderer(self._layout, self._env.position)
        self._assets[path] = image
        self._images.append(image)
        self._owners[id(image)] = self
        self._observations.append({"image": path, "position": list(self._env.position)})
        return path, image

    def _set_decision(self, text, generation):
        if self._pending is not None or self._env.terminated or self._env.truncated:
            raise RuntimeError("Cannot overwrite a decision or act after the episode ends")
        self._pending = {"text": text, **generation}

    def move(self, direction: str) -> list:
        """Move the agent one cell and return public feedback plus a board image.

        Args:
            direction: One of up, down, left, or right.
        """
        if self._pending is None:
            raise RuntimeError("Missing the actual sampled decision")
        decision, self._pending = self._pending, None
        text = decision.pop("text")
        action = parse_move(text)
        if direction != (action if action is not None else "__invalid__"):
            raise RuntimeError("Dispatcher changed the sampled action")
        before = list(self._env.position)
        result = self._env.step(action)
        record = {"step": self._env.steps, "before": before, "after": list(self._env.position),
                  "text": text, "action": action, **asdict(result),
                  "messages": deepcopy(self._history), "generation": decision}
        self._transitions.append(record)
        path, image = self._observe()
        feedback = tool_message(record, path)
        self._history.extend([{"role": "assistant", "content": text}, feedback])
        return [{"type": "text", "text": feedback["content"][0]["text"]},
                {"type": "image", "image": image}]

    def _episode(self):
        if not (self._env.terminated or self._env.truncated):
            raise RuntimeError("TRL ended before the grid's terminal state; check context/tool limits")
        best = len(shortest_path(self._layout))
        episode = {"layout_id": self._layout.layout_id, "policy": "trl_rollout",
                   "seed": episode_seed(self._seed, self._layout), "max_steps": 8,
                   "initial_position": list(self._layout.start), "transitions": self._transitions,
                   "messages": self._history, "observations": self._observations,
                   "steps": self._env.steps, "success": self._env.terminated,
                   "end_reason": "goal" if self._env.terminated else "step_limit",
                   "return": round(sum(t["reward"] for t in self._transitions), 12),
                   "shortest_path_length": best,
                   "path_efficiency": best/self._env.steps if self._env.terminated else 0.0,
                   "invalid_actions": sum(not t["valid_action"] for t in self._transitions),
                   "blocked_moves": sum(t["blocked"] for t in self._transitions)}
        replay_trajectory(self._layout, episode)
        return episode

    def get_reward(self) -> float:
        """The unchanged complete episode return, added once as a reward source."""
        episode = self._episode()
        if not self._saved:
            self._episodes.append(deepcopy(episode))
            self._saved = True
        return episode["return"]

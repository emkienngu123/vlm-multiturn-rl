"""Stage 4 tool protocol, complete histories, and replay; standard library only."""

from copy import deepcopy
from dataclasses import asdict
import json

from .gridworld import DIRECTIONS, GridWorld, Layout, episode_seed, replay_episode, shortest_path


TOOL_PROMPT = (
    "Navigate this top-down 4 by 4 board to the goal. The blue circle is your agent, "
    "the green square is the goal, dark filled cells are walls, and light cells are empty. "
    "Move one cell per turn, avoiding walls and the board edge. Directions refer to the image: "
    "up is toward the top, down toward the bottom, left toward the left edge, right toward the right edge. "
    "You have at most eight moves. Call the move tool exactly once per turn, with direction "
    "up, down, left, or right. Use this exact format, with no explanation: "
    '<tool_call>{"name":"move","arguments":{"direction":"up"}}</tool_call> '
    "The tool returns feedback and a new board image. Use the newest image to choose your next move."
)


def move_call(direction: str) -> str:
    if direction not in DIRECTIONS:
        raise ValueError("Unknown move direction")
    return '<tool_call>' + json.dumps(
        {"name": "move", "arguments": {"direction": direction}}, separators=(",", ":")
    ) + '</tool_call>'


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def parse_move(text: str) -> str | None:
    """One complete tool call only. No regex extraction or fallback action."""
    candidate = text.strip()
    if not (candidate.startswith('<tool_call>') and candidate.endswith('</tool_call>')):
        return None
    try:
        call = json.loads(candidate[len('<tool_call>'):-len('</tool_call>')], object_pairs_hook=_unique_keys)
        if not isinstance(call, dict) or set(call) != {"name", "arguments"} or call["name"] != "move":
            return None
        args = call["arguments"]
        if not isinstance(args, dict) or set(args) != {"direction"}:
            return None
        direction = args["direction"]
        return direction if isinstance(direction, str) and direction in DIRECTIONS else None
    except (ValueError, TypeError):
        return None


def initial_messages(image_path: str) -> list[dict]:
    return [{"role": "system", "content": TOOL_PROMPT},
            {"role": "user", "content": [{"type": "image", "image": image_path},
                                         {"type": "text", "text": "Choose your next move."}]}]


def tool_message(record: dict, image_path: str) -> dict:
    # Feedback is observable; coordinates, the solver path, and layout IDs are not.
    feedback = {key: record[key] for key in ("reward", "valid_action", "blocked", "terminated", "truncated")}
    return {"role": "tool", "content": [
        {"type": "text", "text": json.dumps(feedback, sort_keys=True)},
        {"type": "image", "image": image_path},
    ]}


def image_paths(messages: list[dict]) -> list[str]:
    """Flatten image references in exactly chat-template traversal order."""
    return [part["image"] for message in messages if isinstance(message["content"], list)
            for part in message["content"] if part["type"] == "image"]


def collect_trajectory(layout: Layout, policy: str, *, key: str, assets: dict, renderer,
                       extension: str, actor=None, seed: int = 42, max_steps: int = 8) -> dict:
    """The learned actor receives only public messages and ordered image objects."""
    if policy != "oracle" and actor is None:
        raise ValueError("A non-oracle policy requires a history actor")
    env = GridWorld(layout, max_steps)
    observations = []

    def observe():
        name = f"images/{key}/{env.steps:03d}.{extension}"
        if name in assets:
            raise ValueError("Episode image keys must be unique")
        assets[name] = renderer(layout, env.position)
        observations.append({"image": name, "position": list(env.position)})
        return name

    history = initial_messages(observe())
    transitions = []
    while not (env.terminated or env.truncated):
        if policy == "oracle":
            decision = {"text": move_call(shortest_path(layout, env.position)[0]),
                        "generated_ids": None, "token_source": "solver_text"}
        else:
            decision = actor(deepcopy(history), [assets[name] for name in image_paths(history)])
        text = decision["text"]
        action = parse_move(text)
        before = list(env.position)
        result = env.step(action)
        record = {"step": env.steps, "before": before, "after": list(env.position),
                  "text": text, "action": action, **asdict(result), "messages": deepcopy(history),
                  "generation": {k: v for k, v in decision.items() if k != "text"}}
        transitions.append(record)
        history.extend([{"role": "assistant", "content": text}, tool_message(record, observe())])
    best = len(shortest_path(layout))
    episode = {
        "layout_id": layout.layout_id, "policy": policy, "seed": episode_seed(seed, layout),
        "max_steps": max_steps, "initial_position": list(layout.start),
        "transitions": transitions, "messages": history, "observations": observations,
        "steps": env.steps, "success": env.terminated,
        "end_reason": "goal" if env.terminated else "step_limit",
        "return": round(sum(t["reward"] for t in transitions), 12),
        "shortest_path_length": best, "path_efficiency": best / env.steps if env.terminated else 0.0,
        "invalid_actions": sum(not t["valid_action"] for t in transitions),
        "blocked_moves": sum(t["blocked"] for t in transitions),
    }
    replay_trajectory(layout, episode)
    return episode


def replay_trajectory(layout: Layout, episode: dict) -> None:
    replay_episode(layout, episode, action_parser=parse_move)
    observations = episode["observations"]
    if len(observations) != episode["steps"] + 1 or observations[0]["position"] != list(layout.start):
        raise ValueError("Missing initial/final observation")
    if len({o["image"] for o in observations}) != len(observations):
        raise ValueError("Repeated observation reference")
    history = initial_messages(observations[0]["image"])
    for index, record in enumerate(episode["transitions"]):
        if record["messages"] != history or observations[index+1]["position"] != record["after"]:
            raise ValueError("History or observation ordering mismatch")
        history.extend([{"role": "assistant", "content": record["text"]},
                        tool_message(record, observations[index+1]["image"])])
    if episode["messages"] != history:
        raise ValueError("Final conversation mismatch")


def demonstration_examples(episodes: list[dict]) -> list[dict]:
    """Each action is a target once; previous assistant actions are context only."""
    examples = []
    for episode in episodes:
        if episode["policy"] != "oracle" or not episode["success"]:
            raise ValueError("SFT uses successful solver demonstrations only")
        for record in episode["transitions"]:
            examples.append({"layout_id": episode["layout_id"], "turn": record["step"],
                             "messages": deepcopy(record["messages"]), "target": record["text"]})
    return examples


def assistant_labels(prompt_ids: list[int], target_ids: list[int], image_token_id: int) -> tuple[list[int], list[int]]:
    """Construct unshifted causal-LM labels; the model shifts them exactly once."""
    if not prompt_ids or not target_ids or image_token_id in target_ids:
        raise ValueError("Need a nonempty prompt and a non-image assistant target")
    return prompt_ids + target_ids, [-100] * len(prompt_ids) + target_ids

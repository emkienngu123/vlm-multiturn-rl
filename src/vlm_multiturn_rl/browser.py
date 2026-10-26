"""Screenshot-only MiniWoB contract. Browser/GPU imports are deferred."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import random
import re
from typing import ClassVar

from .trajectories import _unique_keys
from .vision import Stage2Config


MINIWOB_REVISION = "7fd85d71a4b60325c6585396ec4f48377d049838"
MINIWOB_ARCHIVE_SHA256 = "89cc878fc4769d8815630f5367c5c53b0ad4ae0a98b6fa9663eba83cf99a3807"
TASKS = ("click-test", "enter-text", "login-user")
HARD_TASKS = ("click-tab-2-hard", "multi-orderings", "login-user-popup")
BROWSER_PROMPT = (
    "Complete the browser task using its instruction and screenshots. Return exactly one tool call per turn, "
    'with no explanation: <tool_call>{"name":"click","arguments":{"x":100,"y":80}}</tool_call>. '
    "Available tools: click(x,y), type(text), press(key), scroll(dy). "
    "Click coordinates are absolute pixels in the ORIGINAL screenshot: x increases right, y increases down. "
    "Use the supplied image width and height; do not use 0-1000 normalized coordinates. "
    "Click a text field before typing. type inserts text into the focused field. "
    "press accepts Enter, Tab, Backspace, or Control+a. scroll dy is pixels, positive down. "
    "Use the latest screenshot and previous actions. Submit only when the requested fields are correct. "
    "You have at most eight actions."
)


@dataclass(frozen=True)
class BrowserConfig(Stage2Config):
    task_suite: ClassVar[tuple] = TASKS
    action_budget: ClassVar[int] = 8
    context_budget: ClassVar[int] = 4096
    max_new_tokens: int = 64
    episodes: int = 2  # Per task, per split.
    max_steps: int = 8
    max_sequence_tokens: int = 4096
    tasks: tuple = TASKS

    def __post_init__(self):
        super().__post_init__()
        if type(self.episodes) is not int or not 1 <= self.episodes <= 4:
            raise ValueError("Browser episodes per task must be an integer in [1, 4]")
        if (tuple(self.tasks) != self.task_suite or type(self.max_steps) is not int
                or self.max_steps != self.action_budget or self.max_sequence_tokens != self.context_budget):
            raise ValueError("Browser task suite, action budget, or context guard differs from this lesson")
        if self.max_new_tokens < 32:
            raise ValueError("Browser JSON actions require 32 to 64 generated tokens")


def cases(config: BrowserConfig) -> list[dict]:
    return [{"case_id": f"{task}-{split}-{index:03d}", "task": task, "split": split,
             "seed": offset+index} for split, offset in (("train", 0), ("eval", 10000))
            for task in config.tasks for index in range(config.episodes)]


def action_text(name: str, **arguments) -> str:
    return '<tool_call>'+json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))+'</tool_call>'


def parse_action(text: str, width: int, height: int) -> dict | None:
    text = text.strip()
    if not (text.startswith('<tool_call>') and text.endswith('</tool_call>')):
        return None
    try:
        call = json.loads(text[11:-12], object_pairs_hook=_unique_keys)
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}: return None
        name, args = call["name"], call["arguments"]
        if not isinstance(args, dict): return None
        number = lambda v: type(v) in (int, float) and math.isfinite(v)
        if name == "click" and set(args) == {"x", "y"}:
            valid = all(number(args[k]) for k in args) and 0 <= args["x"] < width and 0 <= args["y"] < height
        elif name == "type" and set(args) == {"text"}:
            valid = isinstance(args["text"], str) and 0 < len(args["text"]) <= 128 and all(ord(c) >= 32 for c in args["text"])
        elif name == "press" and set(args) == {"key"}:
            valid = isinstance(args["key"], str) and args["key"] in {"Enter", "Tab", "Backspace", "Control+a"}
        elif name == "scroll" and set(args) == {"dy"}:
            valid = number(args["dy"]) and 0 < abs(args["dy"]) <= 600
        else: valid = False
        return call if valid else None
    except (TypeError, ValueError):
        return None


def executable_action(call: dict | None, width: int, height: int) -> str:
    """Construct allowed calls from validated data; model output is never Python."""
    if call is None: return "noop(0)"
    name, args = call["name"], call["arguments"]
    if name == "click": return f"mouse_click({args['x']!r}, {args['y']!r})"
    if name == "type": return f"keyboard_type({args['text']!r})"
    if name == "press": return f"keyboard_press({args['key']!r})"
    if name == "scroll": return f"scroll_at({width//2}, {height//2}, 0, {args['dy']!r})"
    raise ValueError("Unknown validated action")


def public_observation(obs: dict, image_path: str) -> dict:
    """Explicit whitelist: DOM, accessibility trees, element IDs and info are private."""
    height, width = obs["screenshot"].shape[:2]
    return {"goal": obs["goal"], "image": image_path, "width": int(width), "height": int(height)}


def browser_prompt(max_steps: int = 8) -> str:
    if max_steps == 8: return BROWSER_PROMPT
    if max_steps != 12: raise ValueError("Unsupported browser action budget")
    return BROWSER_PROMPT.replace("at most eight actions", "at most twelve actions")


def initial_history(public: dict, max_steps: int = 8) -> list[dict]:
    return [{"role": "system", "content": browser_prompt(max_steps)}, {"role": "user", "content": [
        {"type": "image", "image": public["image"]},
        {"type": "text", "text": json.dumps({k: public[k] for k in ("goal", "width", "height")}, sort_keys=True)}]}]


def feedback_message(record: dict, image_path: str) -> dict:
    feedback = {k: record[k] for k in ("reward", "valid_action", "execution_error", "terminated", "truncated")}
    return {"role": "tool", "content": [{"type": "text", "text": json.dumps(feedback, sort_keys=True)},
                                          {"type": "image", "image": image_path}]}


class BrowserWorld:
    def __init__(self, case: dict, max_steps: int = 8):
        import gymnasium as gym
        import browsergym.miniwob  # Register tasks, entirely inside the remote path.
        from browsergym.core.action.highlevel import HighLevelActionSet

        if case["task"] not in (*TASKS, *HARD_TASKS): raise ValueError("Unregistered browser task")
        self.case, self.max_steps, self.steps = case, max_steps, 0
        action_set = HighLevelActionSet(subsets=["coord"], multiaction=False, strict=True, demo_mode="off")
        self.env = gym.make("browsergym/miniwob."+case["task"], headless=True,
                            action_mapping=action_set.to_python_code, pre_observation_delay=.1,
                            task_kwargs={"episode_max_time": 3600000})
        try:
            self.obs, self.info = self.env.reset(seed=case["seed"])
        except Exception:
            self.env.close()
            raise
        self.terminated = self.truncated = False

    def step(self, text: str) -> dict:
        if self.terminated or self.truncated: raise RuntimeError("Cannot act after episode end")
        h, w = self.obs["screenshot"].shape[:2]
        action = parse_action(text, w, h)
        code = executable_action(action, w, h)
        self.obs, reward, terminated, truncated, self.info = self.env.step(code)
        self.steps += 1
        self.terminated = bool(terminated)
        self.truncated = bool(truncated or (self.steps >= self.max_steps and not terminated))
        raw = float(self.info["task_info"]["RAW_REWARD_GLOBAL"])
        if raw not in (-1., 0., 1.): raise RuntimeError("Expected a binary-reward task, not partial credit")
        if float(reward) != float(raw > 0): raise RuntimeError("BrowserGym reward contract changed")
        return {"action": action, "executed": code, "valid_action": action is not None,
                "execution_error": bool(self.obs.get("last_action_error")), "reward": float(reward),
                "raw_reward": raw, "terminated": self.terminated, "truncated": self.truncated}

    def close(self):
        self.env.close()


def close_browser_runtime():
    """Release Playwright's sync event loop before Modal performs async I/O."""
    import browsergym.core

    playwright = browsergym.core._PLAYWRIGHT
    if playwright is not None:
        playwright.stop()
        browsergym.core._set_global_playwright(None)


def oracle_actions(world: BrowserWorld) -> list[str]:
    """Privileged demonstration generator, never called by the learned policy."""
    page = world.env.unwrapped.page
    scale = page._bgym_scale_factor
    def click(selector):
        box = page.locator(selector).first.bounding_box()
        if box is None: raise RuntimeError("Oracle target is not visible")
        return action_text("click", x=round((box["x"]+box["width"]/2)*scale, 2),
                           y=round((box["y"]+box["height"]/2)*scale, 2))
    strings = re.findall(r'"([^\"]*)"', world.obs["goal"])
    if world.case["task"] == "click-test":
        return [click("#subbtn")]
    if world.case["task"] == "enter-text":
        return [click("#tt"), action_text("type", text=strings[0]), click("#subbtn")]
    if world.case["task"] == "login-user":
        return [click("#username"), action_text("type", text=strings[0]),
                click("#password"), action_text("type", text=strings[1]), click("#subbtn")]
    raise ValueError("This task requires the reactive oracle, not a fixed action list")


def oracle_next_action(world: BrowserWorld) -> str:
    """React to the current browser state; emit only the policy's allowed tools.

    DOM inspection is privileged teacher logic. It never enters public history.
    Tab search visits panels in order rather than consulting hidden target text.
    """
    task = world.case["task"]
    if task in TASKS: return oracle_actions(world)[world.steps]
    page = world.env.unwrapped.page
    scale = page._bgym_scale_factor
    def click(locator):
        box = locator.bounding_box()
        if box is None: raise RuntimeError("Oracle target is not visible")
        return action_text("click", x=round((box["x"]+box["width"]/2)*scale, 2),
                           y=round((box["y"]+box["height"]/2)*scale, 2))
    def fill(locator, value):
        if locator.input_value() == value: return None
        if locator.evaluate("el => el === document.activeElement"):
            if locator.input_value(): raise RuntimeError("Unexpected nonempty teacher field")
            return action_text("type", text=value)
        return click(locator)
    if task == "login-user-popup":
        popup = page.locator("#popup-cancel")
        if popup.is_visible(): return click(popup)
        strings = re.findall(r'"([^\"]*)"', world.obs["goal"])
        if len(strings) != 2: raise RuntimeError("Login goal format changed")
        for selector, value in zip(("#username", "#password"), strings):
            action = fill(page.locator(selector), value)
            if action: return action
        return click(page.locator("#subbtn"))
    if task == "multi-orderings":
        match = re.fullmatch(r"Search for (.+) movies directed by (.+) from year (\d+)\.", world.obs["goal"])
        if match is None: raise RuntimeError("Movie search goal format changed")
        for label, value in zip(("Genre", "Director", "Year"), match.groups()):
            row = page.locator("#area tr").filter(has=page.locator("th", has_text=re.compile("^"+label+"$")))
            action = fill(row.locator("input"), value)
            if action: return action
        return click(page.locator("#area .final"))
    if task == "click-tab-2-hard":
        match = re.search(r'link "([^"]+)"', world.obs["goal"])
        if match is None: raise RuntimeError("Tab search goal format changed")
        targets = page.locator("#area .alink").filter(has_text=re.compile("^"+re.escape(match[1])+"$"))
        for i in range(targets.count()):
            if targets.nth(i).is_visible(): return click(targets.nth(i))
        # The first tab is visible at reset. Each unsuccessful search opens the next.
        tabs = page.locator("#area .ui-tabs-nav a")
        index = world.steps+1
        if index >= tabs.count(): raise RuntimeError("Oracle exhausted tabs without finding target")
        return click(tabs.nth(index))
    raise ValueError("No oracle registered for this task")


def collect_episode(config: BrowserConfig, case: dict, policy: str, assets: dict, actor=None) -> dict:
    from PIL import Image

    world = BrowserWorld(case, config.max_steps)
    observations, transitions = [], []
    key = f"{policy}-{case['case_id']}"
    def observe():
        path = f"images/{key}/{world.steps:03d}.png"
        image = Image.fromarray(world.obs["screenshot"]).convert("RGB")
        assets[path] = image
        public = public_observation(world.obs, path)
        observations.append({**public, "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest()})
        return public
    try:
        history = initial_history(observe(), config.max_steps)
        rng = random.Random(config.seed+case["seed"])
        while not (world.terminated or world.truncated):
            if policy == "oracle":
                decision = {"text": oracle_next_action(world), "generated_ids": None, "token_source": "privileged_oracle"}
            elif policy == "random":
                public = observations[-1]
                text = action_text("click", x=rng.randrange(public["width"]), y=rng.randrange(public["height"]))
                decision = {"text": text, "generated_ids": None, "token_source": "random_policy"}
            else:
                paths = [o["image"] for o in observations]
                decision = actor(deepcopy(history), [assets[p] for p in paths])
            text = decision["text"]
            record = {"step": world.steps+1, "messages": deepcopy(history), "text": text,
                      "generation": {k:v for k,v in decision.items() if k != "text"}, **world.step(text)}
            transitions.append(record)
            public = observe()
            history.extend([{"role": "assistant", "content": text}, feedback_message(record, public["image"])])
        return {"case": case, "policy": policy, "max_steps": config.max_steps, "observations": observations,
                "transitions": transitions, "messages": history, "steps": world.steps,
                "success": bool(world.terminated and transitions[-1]["raw_reward"] == 1.),
                "return": sum(t["reward"] for t in transitions),
                "invalid_actions": sum(not t["valid_action"] for t in transitions),
                "execution_errors": sum(t["execution_error"] for t in transitions),
                "end_reason": "task_terminal" if world.terminated else "action_limit"}
    finally:
        world.close()


def validate_episode(episode: dict) -> None:
    obs, turns = episode["observations"], episode["transitions"]
    if not turns or len(turns) != episode["steps"] or len(obs) != len(turns)+1:
        raise ValueError("Missing browser action or observation")
    if not 0 < len(turns) <= episode["max_steps"]: raise ValueError("Episode exceeds its action budget")
    history = initial_history(obs[0], episode["max_steps"])
    for i, t in enumerate(turns):
        action = parse_action(t["text"], obs[i]["width"], obs[i]["height"])
        if (t["messages"] != history or t["step"] != i+1 or t["action"] != action
                or t["valid_action"] != (action is not None)
                or t["executed"] != executable_action(action, obs[i]["width"], obs[i]["height"])
                or t["reward"] != float(t["raw_reward"] > 0)):
            raise ValueError("Browser history, parser or reward mismatch")
        final = i == len(turns)-1
        if final != (t["terminated"] or t["truncated"]): raise ValueError("Invalid episode termination")
        history.extend([{"role":"assistant", "content":t["text"]}, feedback_message(t, obs[i+1]["image"])])
    expected = {"messages": history, "return": sum(t["reward"] for t in turns),
                "success": bool(turns[-1]["terminated"] and turns[-1]["raw_reward"] == 1.),
                "invalid_actions": sum(not t["valid_action"] for t in turns),
                "execution_errors": sum(t["execution_error"] for t in turns),
                "end_reason": "task_terminal" if turns[-1]["terminated"] else "action_limit"}
    if any(episode[k] != value for k,value in expected.items()): raise ValueError("Browser summary mismatch")


def summarize(episodes: list) -> dict:
    # Compare policies on evaluation instances only. Training demonstrations and
    # contract controls are available in episode replays, not evaluation scores.
    episodes = [e for e in episodes if e["case"]["split"] == "eval"]
    result = {}
    for policy in sorted({e["policy"] for e in episodes}):
        selected = [e for e in episodes if e["policy"] == policy]
        result[policy] = {}
        for task in ("all", *TASKS, *sorted({e["case"]["task"] for e in selected}-set(TASKS))):
            group = [e for e in selected if task == "all" or e["case"]["task"] == task]
            if not group: continue
            actions = sum(e["steps"] for e in group)
            result[policy][task] = {"episodes": len(group), "successes": sum(e["success"] for e in group),
                "success_rate": sum(e["success"] for e in group)/len(group),
                "mean_steps": actions/len(group), "mean_return": sum(e["return"] for e in group)/len(group),
                "invalid_action_rate": sum(e["invalid_actions"] for e in group)/actions,
                "execution_error_rate": sum(e["execution_errors"] for e in group)/actions}
    return result

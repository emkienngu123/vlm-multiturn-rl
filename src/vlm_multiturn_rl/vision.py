"""Stage 2 fixtures and token boundaries; imports need no third-party libraries."""

from dataclasses import dataclass

from .workspace import Stage1Config


PROMPT = (
    "This is a top-down board. The blue circle is your agent; the green square is "
    "the goal. All other cells are empty. Move exactly one cell to reach the goal. "
    "Directions refer to the image: up is toward the top, down toward the bottom, "
    "left toward the left edge, right toward the right edge. "
    "Reply with exactly one word: up, down, left, or right. Do not explain."
)
IMAGE_SIDE = 448
LOW_PIXELS = 224 * 224


@dataclass(frozen=True)
class Stage2Config(Stage1Config):
    min_pixels: int = LOW_PIXELS
    max_pixels: int = IMAGE_SIDE * IMAGE_SIDE
    max_new_tokens: int = 16

    def __post_init__(self):
        super().__post_init__()
        if self.min_pixels != LOW_PIXELS:
            raise ValueError("Stage 2 fixes min_pixels at 224*224 for its resolution comparison")
        if type(self.max_pixels) is not int or not self.min_pixels <= self.max_pixels <= IMAGE_SIDE**2:
            raise ValueError("max_pixels must be between 224*224 and 448*448")
        if type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= 64:
            raise ValueError("max_new_tokens must be an integer between 1 and 64")


@dataclass(frozen=True)
class Board:
    """Coordinates are (row, column), origin at the upper left; never sent as text."""

    agent: tuple[int, int]
    goal: tuple[int, int]
    size: int = 4

    def __post_init__(self):
        if type(self.size) is not int or self.size < 2:
            raise ValueError("Board size must be an integer >= 2")
        for position in (self.agent, self.goal):
            if len(position) != 2 or any(type(x) is not int or not 0 <= x < self.size for x in position):
                raise ValueError("Board coordinates must be in bounds")
        if self.agent == self.goal:
            raise ValueError("Agent and goal must occupy different cells")


def control_boards() -> dict[str, Board]:
    return {
        "goal-left": Board(agent=(1, 1), goal=(1, 0)),
        "goal-right": Board(agent=(1, 1), goal=(1, 2)),
    }


def expected_action(board: Board) -> str:
    delta = (board.goal[0] - board.agent[0], board.goal[1] - board.agent[1])
    directions = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}
    if delta not in directions:
        raise ValueError("Stage 2's one-action diagnostic requires an adjacent goal")
    return directions[delta]


def messages() -> list[dict]:
    # The processor receives the actual image separately, in placeholder order.
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]


def render_board(board: Board):
    return render_grid(board.agent, board.goal, board.size)


def render_grid(agent, goal, size: int = 4, walls: tuple = ()):
    """Draw a validated state, including terminal states where agent == goal."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (IMAGE_SIDE, IMAGE_SIDE), "#f1f5f9")
    draw = ImageDraw.Draw(image)
    cell = IMAGE_SIDE / size
    for i in range(size + 1):
        position = min(round(i * cell), IMAGE_SIDE - 1)
        draw.line((position, 0, position, IMAGE_SIDE - 1), fill="#475569", width=3)
        draw.line((0, position, IMAGE_SIDE - 1, position), fill="#475569", width=3)

    def box(position):
        row, column = position
        return ((column + 0.22) * cell, (row + 0.22) * cell,
                (column + 0.78) * cell, (row + 0.78) * cell)

    for row, column in walls:
        draw.rectangle((column*cell+3, row*cell+3, (column+1)*cell-3, (row+1)*cell-3), fill="#334155")
    draw.rectangle(box(goal), fill="#22c55e", outline="#166534", width=3)
    draw.ellipse(box(agent), fill="#2563eb", outline="#1e3a8a", width=3)
    return image


def parse_action(text: str) -> str | None:
    candidate = text.strip().lower()
    return candidate if candidate in {"up", "down", "left", "right"} else None


def generated_suffix(prompt_ids: list[int], output_ids: list[int]) -> list[int]:
    """Batch-size-one, decoder-only output: keep exact new tokens, not re-tokenized text."""
    if output_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Generated sequence does not preserve its input prefix")
    return output_ids[len(prompt_ids):]

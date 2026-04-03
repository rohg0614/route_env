"""Task presets for the route dispatch benchmark."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    name: str
    horizon_steps: int
    node_count: int
    max_shift_hours: float
    base_lambda: float
    lateness_budget: float
    distance_scale: float


TASKS: dict[str, TaskConfig] = {
    "easy": TaskConfig("easy", 60, 6, 8.0, 1.2, 0.45, 8.0),
    "medium": TaskConfig("medium", 84, 8, 9.0, 1.8, 0.32, 10.0),
    "hard": TaskConfig("hard", 108, 10, 10.0, 2.4, 0.24, 12.0),
}

TASK_ORDER = ["easy", "medium", "hard"]

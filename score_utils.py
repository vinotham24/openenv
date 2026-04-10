from __future__ import annotations


MIN_OPENENV_VALUE = 0.0001
MAX_OPENENV_VALUE = 0.9999
NEUTRAL_REWARD = 0.5
MIN_TASK_SCORE = MIN_OPENENV_VALUE
MAX_TASK_SCORE = MAX_OPENENV_VALUE


def clamp_openenv_value(
    value: float,
    min_value: float = MIN_OPENENV_VALUE,
    max_value: float = MAX_OPENENV_VALUE,
) -> float:
    return round(min(max(value, min_value), max_value), 4)


def bounded_unit_interval(value: float) -> float:
    return clamp_openenv_value(value, MIN_TASK_SCORE, MAX_TASK_SCORE)


def bounded_reward(value: float) -> float:
    return clamp_openenv_value(NEUTRAL_REWARD + value)

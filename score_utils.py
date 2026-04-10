from __future__ import annotations


MIN_TASK_SCORE = 0.1
MAX_TASK_SCORE = 0.9


def bounded_unit_interval(value: float) -> float:
    return round(min(max(value, MIN_TASK_SCORE), MAX_TASK_SCORE), 4)

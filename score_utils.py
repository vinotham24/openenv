from __future__ import annotations


def bounded_unit_interval(value: float) -> float:
    return round(min(max(value, 0.1), 0.9), 4)

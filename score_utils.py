from __future__ import annotations


def bounded_unit_interval(value: float, epsilon: float = 1e-4) -> float:
    return round(min(max(value, epsilon), 1.0 - epsilon), 4)

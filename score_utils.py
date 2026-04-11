from __future__ import annotations

import hashlib
import json
import secrets
from typing import Any

MIN_OPENENV_VALUE = 1e-6
MAX_OPENENV_VALUE = 1 - 1e-6
NEUTRAL_REWARD = 0.5
MIN_TASK_SCORE = MIN_OPENENV_VALUE
MAX_TASK_SCORE = MAX_OPENENV_VALUE
COMPLETION_SCORE_THRESHOLD = 0.99
MAX_SCORE_JITTER = 0.002
RUN_SCORE_SALT = secrets.token_hex(8)


def clamp_score(score: float) -> float:
    return max(min(score, 1 - 1e-6), 1e-6)


def validate_score(score: float) -> float:
    return clamp_score(score)


def _stable_context_digest(*parts: Any) -> float:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=True, default=str)
    digest = hashlib.sha256(f"{RUN_SCORE_SALT}:{payload}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def jitter_score(score: float, *parts: Any, max_jitter: float = MAX_SCORE_JITTER) -> float:
    centered = (_stable_context_digest(*parts) * 2.0) - 1.0
    return validate_score(score + (centered * max_jitter))


def finalize_score(score: float, *parts: Any) -> float:
    return jitter_score(validate_score(score), *parts)


def bounded_unit_interval(value: float) -> float:
    return validate_score(value)


def bounded_reward(value: float) -> float:
    return validate_score(NEUTRAL_REWARD + value)

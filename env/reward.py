from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set


@dataclass
class RewardTracker:
    seen_actions: Set[str] = field(default_factory=set)
    last_signature: str | None = None

    def score(
        self,
        signature: str,
        progress_delta: float,
        valid: bool,
        loop_detected: bool,
    ) -> tuple[float, Dict[str, float]]:
        reward = 0.0
        components = {
            "progress": 0.0,
            "invalid_action": 0.0,
            "repeat_penalty": 0.0,
            "loop_penalty": 0.0,
        }

        if progress_delta > 0:
            components["progress"] = round(min(progress_delta, 1.0), 4)
            reward += components["progress"]

        if not valid:
            components["invalid_action"] = -0.2
            reward -= 0.2

        if signature in self.seen_actions:
            components["repeat_penalty"] = -0.1
            reward -= 0.1
        else:
            self.seen_actions.add(signature)

        if loop_detected:
            components["loop_penalty"] = -0.15
            reward -= 0.15

        self.last_signature = signature
        reward = round(max(min(reward, 1.0), -1.0), 4)
        return reward, components

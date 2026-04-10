from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from env.schemas import Action, Observation
from tasks.code_review.grader import grade_code_review
from score_utils import bounded_unit_interval


class CodeReviewTask:
    task_id = "code_review"
    task_name = "code_review"
    difficulty = "hard"

    def __init__(self, seed: int = 42) -> None:
        del seed
        self.buggy_code = Path(__file__).with_name("buggy_code.py").read_text(encoding="utf-8-sig")
        self.bugs: List[str] = []
        self.fixed_code = ""
        self.history: List[Dict[str, Any]] = []
        self.last_progress = 0.0

    def reset(self) -> None:
        self.bugs = []
        self.fixed_code = ""
        self.history = []
        self.last_progress = 0.0

    def observation(self, max_steps: int) -> Observation:
        raw_progress = grade_code_review(self.bugs, self.fixed_code) if (self.bugs or self.fixed_code) else 0.0
        return Observation(
            task_id=self.task_id,
            task_name=self.task_name,
            difficulty=self.difficulty,
            instruction="Review the Python file, identify the defects, and provide corrected code.",
            content={
                "buggy_code": self.buggy_code,
                "expected_behavior": [
                    "Discount should subtract the actual amount.",
                    "Tax should be applied as a percentage.",
                    "Order summaries should sort by count descending and customer ascending.",
                ],
            },
            history=self.history,
            hints=["There are logic bugs in both functions.", "Submit both a bug list and fixed code."],
            progress=bounded_unit_interval(raw_progress),
            attempts_remaining=max(max_steps - len(self.history), 0),
        )

    def detect_loop(self, action: Action) -> bool:
        if len(self.history) < 2:
            return False
        last_two = self.history[-2:]
        return all(item["action_type"] == action.action_type for item in last_two)

    def apply_action(self, action: Action) -> Dict[str, Any]:
        valid = True
        completed = False
        error = None
        details: Dict[str, Any] = {}

        if action.action_type == "analyze":
            details["analysis"] = "Inspected arithmetic logic, sorting behavior, and expected test outcomes."
        elif action.action_type == "review_code":
            bugs = action.payload.get("bugs")
            fixed_code = action.payload.get("fixed_code")
            if not isinstance(bugs, list) or not all(isinstance(item, str) for item in bugs):
                valid = False
                error = "bugs must be a list of strings"
            elif not isinstance(fixed_code, str) or not fixed_code.strip():
                valid = False
                error = "fixed_code must be a non-empty string"
            else:
                self.bugs = bugs
                self.fixed_code = fixed_code.strip()
                details["bugs_reported"] = len(self.bugs)
        elif action.action_type == "submit":
            completed = True
            details["submitted"] = True
        else:
            valid = False
            error = "unsupported action for code review"

        self.history.append({"action_type": action.action_type, "payload": action.payload})
        raw_score = grade_code_review(self.bugs, self.fixed_code)
        if raw_score >= 0.9:
            completed = True
        progress_delta = max(raw_score - self.last_progress, 0.0)
        self.last_progress = max(self.last_progress, raw_score)
        return {
            "valid": valid,
            "completed": completed,
            "score": bounded_unit_interval(raw_score),
            "progress_delta": round(progress_delta if valid else 0.0, 4),
            "error": error,
            "details": details,
            "message": "Code review step processed.",
        }

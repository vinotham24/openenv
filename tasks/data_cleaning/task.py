from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from env.schemas import Action, Observation
from tasks.data_cleaning.grader import EXPECTED_DATA, grade_cleaned_csv
from score_utils import COMPLETION_SCORE_THRESHOLD, MAX_TASK_SCORE, MIN_TASK_SCORE, validate_score


class DataCleaningTask:
    task_id = "data_cleaning"
    task_name = "data_cleaning"
    difficulty = "medium"

    def __init__(self, seed: int = 42) -> None:
        del seed
        self.raw_csv = Path(__file__).with_name("dataset.csv").read_text(encoding="utf-8-sig")
        self.cleaned_csv = ""
        self.history: List[Dict[str, Any]] = []
        self.last_progress = MIN_TASK_SCORE

    def reset(self) -> None:
        self.cleaned_csv = ""
        self.history = []
        self.last_progress = MIN_TASK_SCORE

    def observation(self, max_steps: int) -> Observation:
        raw_progress = grade_cleaned_csv(self.cleaned_csv) if self.cleaned_csv else MIN_TASK_SCORE
        return Observation(
            task_id=self.task_id,
            task_name=self.task_name,
            difficulty=self.difficulty,
            instruction="Clean the CSV by filling missing values and normalizing dates, countries, amounts, and status values.",
            content={
                "raw_csv": self.raw_csv,
                "schema": list(EXPECTED_DATA[0].keys()),
                "requirements": [
                    "Dates must be YYYY-MM-DD.",
                    "Countries must be normalized.",
                    "purchase_total must have two decimal places.",
                    "status must be lowercase.",
                ],
            },
            history=self.history,
            hints=["Fill missing purchase_total for row 002.", "Fill missing signup_date for row 004."],
            progress=validate_score(raw_progress),
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
            details["analysis"] = "Inspected missing values, date formats, whitespace, and capitalization issues."
        elif action.action_type == "clean_data":
            cleaned_csv = action.payload.get("cleaned_csv")
            if not isinstance(cleaned_csv, str) or not cleaned_csv.strip():
                valid = False
                error = "cleaned_csv must be a non-empty string"
            else:
                self.cleaned_csv = cleaned_csv.strip()
                details["rows_submitted"] = len(self.cleaned_csv.splitlines()) - 1
        elif action.action_type == "submit":
            completed = True
            details["submitted"] = True
        else:
            valid = False
            error = "unsupported action for data cleaning"

        self.history.append({"action_type": action.action_type, "payload": action.payload})
        raw_score = grade_cleaned_csv(self.cleaned_csv) if self.cleaned_csv else MIN_TASK_SCORE
        if raw_score >= COMPLETION_SCORE_THRESHOLD:
            completed = True
        progress_delta = max(raw_score - self.last_progress, 0.0)
        self.last_progress = max(self.last_progress, raw_score)
        return {
            "valid": valid,
            "completed": completed,
            "score": validate_score(raw_score),
            "progress_delta": validate_score(progress_delta) if valid and progress_delta > 0 else 0.0,
            "error": error,
            "details": details,
            "message": "Data cleaning step processed.",
        }

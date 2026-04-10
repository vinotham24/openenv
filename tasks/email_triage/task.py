from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from env.schemas import Action, Observation
from env.utils import bounded_unit_interval, load_json
from tasks.email_triage.grader import grade_email_triage


class EmailTriageTask:
    task_id = "email_triage"
    task_name = "email_triage"
    difficulty = "easy"

    def __init__(self, seed: int = 42) -> None:
        del seed
        data = load_json(Path(__file__).with_name("data.json"))
        self.emails: List[Dict[str, Any]] = data["emails"]
        self.answers = {item["id"]: item["label"] for item in self.emails}
        self.predictions: List[Dict[str, str]] = []
        self.history: List[Dict[str, Any]] = []
        self.last_progress = 0.0

    def reset(self) -> None:
        self.predictions = []
        self.history = []
        self.last_progress = 0.0

    def observation(self, max_steps: int) -> Observation:
        raw_progress = grade_email_triage(self.predictions, self.answers)
        return Observation(
            task_id=self.task_id,
            task_name=self.task_name,
            difficulty=self.difficulty,
            instruction="Classify each incoming email as spam, important, or respond.",
            content={"emails": self.emails, "labels": ["spam", "important", "respond"]},
            history=self.history,
            hints=["Use exact labels.", "Urgent internal work is usually important."],
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
            details["analysis"] = "Reviewed sender trust, urgency, and follow-up requirements."
        elif action.action_type == "classify_email":
            email_id = action.payload.get("id")
            label = action.payload.get("label")
            if email_id not in self.answers or label not in {"spam", "important", "respond"}:
                valid = False
                error = "invalid email classification payload"
            else:
                self.predictions = [item for item in self.predictions if item["id"] != email_id]
                self.predictions.append({"id": email_id, "label": label})
                details["classified"] = {"id": email_id, "label": label}
        elif action.action_type == "submit":
            completed = True
            details["submitted"] = True
        else:
            valid = False
            error = "unsupported action for email triage"

        self.history.append({"action_type": action.action_type, "payload": action.payload})
        raw_score = grade_email_triage(self.predictions, self.answers)
        if len(self.predictions) == len(self.answers) and raw_score == 1.0:
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
            "message": "Email triage step processed.",
        }

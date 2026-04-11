from __future__ import annotations

from typing import Dict, List

from score_utils import finalize_score


def grade_email_triage(predictions: List[Dict[str, str]], answers: Dict[str, str]) -> float:
    if not answers:
        return finalize_score(0.0, "email_triage", predictions, answers)
    correct = 0
    for email_id, label in answers.items():
        predicted = next((item.get("label") for item in predictions if item.get("id") == email_id), None)
        if predicted == label:
            correct += 1
    return finalize_score(correct / len(answers), "email_triage", predictions, answers, correct)

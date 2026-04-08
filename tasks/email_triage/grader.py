from __future__ import annotations

from typing import Dict, List


def grade_email_triage(predictions: List[Dict[str, str]], answers: Dict[str, str]) -> float:
    if not answers:
        return 0.0
    correct = 0
    for email_id, label in answers.items():
        predicted = next((item.get("label") for item in predictions if item.get("id") == email_id), None)
        if predicted == label:
            correct += 1
    return round(correct / len(answers), 4)

from __future__ import annotations

from typing import Dict, List

from score_utils import finalize_score


LABEL_WEIGHTS = {
    "spam": 0.8,
    "important": 1.0,
    "respond": 0.9,
}


def grade_email_triage(predictions: List[Dict[str, str]], answers: Dict[str, str]) -> float:
    if not answers:
        return finalize_score(0.0, "email_triage", predictions, answers)

    prediction_map = {item.get("id"): item.get("label") for item in predictions}
    total_weight = sum(LABEL_WEIGHTS[label] for label in answers.values())
    matched_weight = 0.0
    completion_bonus = 0.0

    for email_id, expected_label in answers.items():
        predicted_label = prediction_map.get(email_id)
        if predicted_label == expected_label:
            matched_weight += LABEL_WEIGHTS[expected_label]

    coverage = len({email_id for email_id in prediction_map if email_id in answers}) / len(answers)
    if coverage > 0:
        completion_bonus = min(coverage * 0.08, 0.08)

    base_score = (matched_weight / total_weight) * 0.92 + completion_bonus
    return finalize_score(base_score, "email_triage", predictions, answers, matched_weight, coverage)

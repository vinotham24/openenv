from __future__ import annotations

from typing import List

from score_utils import bounded_unit_interval


BUG_KEYWORDS = ("discount", "tax", "sort")


def score_bug_report(report: List[str]) -> float:
    if not report:
        return bounded_unit_interval(0.0)
    text = " ".join(report).lower()
    hits = sum(1 for keyword in BUG_KEYWORDS if keyword in text)
    return bounded_unit_interval(hits / len(BUG_KEYWORDS))


def score_fixed_code(code: str) -> float:
    namespace = {}
    try:
        exec(compile(code, "candidate.py", "exec"), namespace)
    except Exception:
        return bounded_unit_interval(0.0)

    try:
        invoice = namespace["calculate_invoice_total"](
            [{"price": 100.0, "quantity": 2}, {"price": 50.0, "quantity": 1}],
            tax_rate=0.1,
            discount=20.0,
        )
        summary = namespace["summarize_orders"](
            [{"customer": "Beta"}, {"customer": "Acme"}, {"customer": "Beta"}]
        )
    except Exception:
        return bounded_unit_interval(0.0)

    score = 0.0
    if invoice == 253.0:
        score += 0.6
    if summary == [("Beta", 2), ("Acme", 1)]:
        score += 0.4
    return bounded_unit_interval(score)


def grade_code_review(report: List[str], fixed_code: str) -> float:
    return bounded_unit_interval((score_bug_report(report) * 0.4) + (score_fixed_code(fixed_code) * 0.6))

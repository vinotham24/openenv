from __future__ import annotations

from typing import Iterable, List

from score_utils import finalize_score, validate_score


BUG_CONCEPTS = [
    (("cutoff", "window"), ("recent", "old"), 0.25),
    (("vendor", "supplier"), ("risk", "score"), ("accumulate", "overwrite"), 0.25),
    (("sort", "order"), ("risk", "priority"), ("descending", "highest"), 0.25),
    (("route_family", "summary"), ("missing", "include"), 0.25),
]


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def score_bug_report(report: List[str]) -> float:
    if not report:
        return validate_score(0.0)

    text = " ".join(report).lower()
    coverage = 0.0
    for concept in BUG_CONCEPTS:
        term_groups = concept[:-1]
        weight = concept[-1]
        if all(_contains_any(text, group) for group in term_groups):
            coverage += weight
    return validate_score(coverage)


def score_fixed_code(code: str) -> float:
    namespace = {}
    try:
        exec(compile(code, "candidate.py", "exec"), namespace)
    except Exception:
        return validate_score(0.0)

    try:
        select_shipments = namespace["select_shipments_for_manual_review"]
        summarize = namespace["summarize_flagged_shipments"]
    except KeyError:
        return validate_score(0.0)

    now_iso = "2026-04-10T12:00:00"
    vendor_risk = {"SUP-1": 0.9, "SUP-2": 0.2, "SUP-3": 0.81}
    events = [
        {
            "shipment_id": "SH-310",
            "event_time": "2026-04-09T10:15:00",
            "declared_value": 17000,
            "supplier_id": "SUP-1",
            "seal_status": "ok",
            "invoice_status": "present",
            "route_family": "high_value",
            "temp_c": 4.0,
        },
        {
            "shipment_id": "SH-311",
            "event_time": "2026-04-08T04:30:00",
            "declared_value": 8000,
            "supplier_id": "SUP-3",
            "seal_status": "broken",
            "invoice_status": "missing",
            "route_family": "high_value",
            "temp_c": 3.0,
        },
        {
            "shipment_id": "SH-312",
            "event_time": "2026-03-20T06:00:00",
            "declared_value": 20000,
            "supplier_id": "SUP-1",
            "seal_status": "broken",
            "invoice_status": "missing",
            "route_family": "high_value",
            "temp_c": 2.0,
        },
        {
            "shipment_id": "SH-313",
            "event_time": "2026-04-09T23:00:00",
            "declared_value": 9000,
            "supplier_id": "SUP-2",
            "seal_status": "ok",
            "invoice_status": "present",
            "route_family": "cold",
            "temp_c": 6.2,
        },
    ]

    score = 0.0

    try:
        flagged = select_shipments(events, vendor_risk, now_iso)
        flagged_ids = [item["shipment_id"] for item in flagged]
        if flagged_ids[:3] == ["SH-311", "SH-310", "SH-313"]:
            score += 0.35
        if "SH-312" not in flagged_ids:
            score += 0.15
        if all("route_family" in item for item in flagged):
            score += 0.15
        summary = summarize(flagged)
        if summary == [("high_value", 7), ("cold", 1)]:
            score += 0.35
    except Exception:
        return validate_score(score)

    return validate_score(score)


def grade_code_review(report: List[str], fixed_code: str) -> float:
    report_score = score_bug_report(report)
    code_score = score_fixed_code(fixed_code)
    base_score = report_score * 0.25 + code_score * 0.75
    return finalize_score(base_score, "code_review", report, fixed_code, report_score, code_score)

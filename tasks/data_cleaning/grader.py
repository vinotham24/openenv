from __future__ import annotations

import csv
from io import StringIO
from typing import Dict, List

from score_utils import bounded_unit_interval


EXPECTED_DATA = [
    {"customer_id": "001", "signup_date": "2026-01-05", "country": "USA", "purchase_total": "1200.50", "status": "active"},
    {"customer_id": "002", "signup_date": "2026-02-05", "country": "India", "purchase_total": "810.42", "status": "active"},
    {"customer_id": "003", "signup_date": "2026-03-11", "country": "UK", "purchase_total": "980.00", "status": "inactive"},
    {"customer_id": "004", "signup_date": "2026-02-13", "country": "USA", "purchase_total": "450.00", "status": "active"},
    {"customer_id": "005", "signup_date": "2026-04-01", "country": "India", "purchase_total": "610.75", "status": "active"},
]


def parse_csv(text: str) -> List[Dict[str, str]]:
    reader = csv.DictReader(StringIO(text.strip()))
    return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def grade_cleaned_csv(text: str) -> float:
    rows = parse_csv(text)
    if len(rows) != len(EXPECTED_DATA):
        return bounded_unit_interval(0.0)
    total = len(EXPECTED_DATA) * len(EXPECTED_DATA[0])
    matched = 0
    for expected, actual in zip(EXPECTED_DATA, rows):
        for key, value in expected.items():
            if actual.get(key) == value:
                matched += 1
    return bounded_unit_interval(matched / total)

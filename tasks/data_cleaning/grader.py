from __future__ import annotations

import csv
from datetime import datetime
from io import StringIO
from typing import Dict, List

from score_utils import finalize_score


EXPECTED_DATA = [
    {
        "shipment_id": "SH-201",
        "carrier": "Polar Express",
        "route_family": "cold",
        "delay_hours": "2.5",
        "risk_trigger": "temperature",
        "priority": "critical",
        "owner_action": "cold_chain_ops",
        "resolution_tag": "replace_inventory",
    },
    {
        "shipment_id": "SH-202",
        "carrier": "Relay Freight",
        "route_family": "ambient",
        "delay_hours": "1.2",
        "risk_trigger": "none",
        "priority": "monitor",
        "owner_action": "transport_control",
        "resolution_tag": "hold_for_next_scan",
    },
    {
        "shipment_id": "SH-203",
        "carrier": "NorthStar Secure",
        "route_family": "high_value",
        "delay_hours": "6.5",
        "risk_trigger": "security",
        "priority": "critical",
        "owner_action": "fraud_desk",
        "resolution_tag": "escalate_security_hold",
    },
    {
        "shipment_id": "SH-204",
        "carrier": "Relay Freight",
        "route_family": "ambient",
        "delay_hours": "5.0",
        "risk_trigger": "delay",
        "priority": "high",
        "owner_action": "transport_control",
        "resolution_tag": "clear_customs_docs",
    },
]


def parse_csv(text: str) -> List[Dict[str, str]]:
    reader = csv.DictReader(StringIO(text.strip()))
    return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def _normalize_carrier(alias: str) -> str:
    alias = alias.strip().lower()
    carrier_map = {
        "px logistics": "Polar Express",
        "polar express": "Polar Express",
        "relay freight": "Relay Freight",
        "northstar secure": "NorthStar Secure",
    }
    return carrier_map.get(alias, alias.title())


def _parse_datetime(text: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported datetime format: {text}")


def _expected_from_raw_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned_rows: List[Dict[str, str]] = []
    for row in rows:
        planned = _parse_datetime(row["planned_eta_local"])
        latest = _parse_datetime(row["latest_eta_local"])
        delay_hours = round((latest - planned).total_seconds() / 3600, 1)
        route_family = row["route_family"].strip().lower()
        temp = float(row["temp_probe_c"])
        seal_status = row["seal_check"].strip().lower()
        note = row["exception_note"].strip().lower()

        risk_trigger = "none"
        if route_family == "high_value" and seal_status == "broken":
            risk_trigger = "security"
        elif route_family == "cold" and temp > 5:
            risk_trigger = "temperature"
        elif delay_hours >= 4:
            risk_trigger = "delay"

        if risk_trigger == "security" or (risk_trigger == "temperature" and delay_hours >= 2):
            priority = "critical"
        elif "customs" in note or delay_hours >= 4:
            priority = "high"
        else:
            priority = "monitor"

        if risk_trigger == "security":
            owner_action = "fraud_desk"
            resolution_tag = "escalate_security_hold"
        elif risk_trigger == "temperature":
            owner_action = "cold_chain_ops"
            resolution_tag = "replace_inventory"
        elif "customs" in note:
            owner_action = "transport_control"
            resolution_tag = "clear_customs_docs"
        else:
            owner_action = "transport_control"
            resolution_tag = "hold_for_next_scan"

        cleaned_rows.append(
            {
                "shipment_id": row["shipment_id"].strip(),
                "carrier": _normalize_carrier(row["carrier_alias"]),
                "route_family": route_family,
                "delay_hours": f"{delay_hours:.1f}",
                "risk_trigger": risk_trigger,
                "priority": priority,
                "owner_action": owner_action,
                "resolution_tag": resolution_tag,
            }
        )
    return cleaned_rows


def grade_cleaned_csv(text: str) -> float:
    rows = parse_csv(text)
    if not rows:
        return finalize_score(0.0, "data_cleaning", rows)

    total_fields = len(EXPECTED_DATA) * len(EXPECTED_DATA[0])
    expected_rows = EXPECTED_DATA

    matched = 0
    structure_bonus = 0.0
    if len(rows) == len(expected_rows) and set(rows[0].keys()) == set(expected_rows[0].keys()):
        structure_bonus = 0.08

    for expected, actual in zip(expected_rows, rows):
        for key, value in expected.items():
            if actual.get(key) == value:
                matched += 1

    base_score = (matched / total_fields) * 0.92 + structure_bonus
    return finalize_score(base_score, "data_cleaning", rows, matched, structure_bonus)

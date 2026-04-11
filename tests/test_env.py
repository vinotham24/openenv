from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from env.base_env import OpenEnvRealWorldSim
from score_utils import MAX_OPENENV_VALUE, MIN_OPENENV_VALUE, clamp_score
from server.app import app
from tasks.code_review.grader import grade_code_review
from tasks.data_cleaning.grader import EXPECTED_DATA, grade_cleaned_csv
from tasks.email_triage.grader import grade_email_triage


def test_reset_returns_observation() -> None:
    env = OpenEnvRealWorldSim(seed=42)
    observation = env.reset()
    assert observation.task_name == "email_triage"
    assert MIN_OPENENV_VALUE <= observation.progress <= MAX_OPENENV_VALUE
    assert observation.progress <= 0.02
    assert observation.attempts_remaining == 6


def test_step_returns_expected_tuple() -> None:
    env = OpenEnvRealWorldSim(seed=42)
    env.reset()
    observation, reward, done, info = env.step(
        {"action_type": "classify_email", "payload": {"id": "email-001", "label": "important"}}
    )
    assert observation.task_name == "email_triage"
    assert MIN_OPENENV_VALUE <= reward <= MAX_OPENENV_VALUE
    assert reward > 0.5
    assert done is False
    assert MIN_OPENENV_VALUE <= info["task_score"] <= MAX_OPENENV_VALUE
    assert info["task_score"] > 0.15
    assert info["task_score"] < 0.2


def test_reward_penalizes_invalid_action() -> None:
    env = OpenEnvRealWorldSim(seed=42)
    env.reset()
    _, reward, _, info = env.step({"action_type": "clean_data", "payload": {"cleaned_csv": "x"}})
    assert MIN_OPENENV_VALUE <= reward <= MAX_OPENENV_VALUE
    assert reward < 0.5
    assert info["reward"]["components"]["invalid_action"] == -0.2


def test_root_route_exists() -> None:
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["routes"]["docs"] == "/docs"


def test_reset_route_returns_full_observation() -> None:
    client = TestClient(app)
    response = client.post("/reset")

    assert response.status_code == 200
    body = response.json()
    assert body["observation"]["task_id"] == "email_triage"
    assert MIN_OPENENV_VALUE <= body["task_score"] <= MAX_OPENENV_VALUE
    assert body["info"]["details"]["reset"] is True


def test_step_route_returns_full_info() -> None:
    client = TestClient(app)
    client.post("/reset")
    response = client.post(
        "/step",
        json={"action_type": "classify_email", "payload": {"id": "email-001", "label": "important"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["observation"]["task_id"] == "email_triage"
    assert MIN_OPENENV_VALUE <= body["task_score"] <= MAX_OPENENV_VALUE
    assert "reward" in body["info"]


def test_clamp_score_excludes_zero_and_one() -> None:
    assert clamp_score(0.0) == pytest.approx(MIN_OPENENV_VALUE)
    assert clamp_score(1.0) == pytest.approx(MAX_OPENENV_VALUE)
    assert clamp_score(0.25) == pytest.approx(0.25)
    assert clamp_score(-1.0) == pytest.approx(MIN_OPENENV_VALUE)
    assert clamp_score(2.0) == pytest.approx(MAX_OPENENV_VALUE)
    assert clamp_score(None) == pytest.approx(MIN_OPENENV_VALUE)
    assert clamp_score(float("nan")) == pytest.approx(MIN_OPENENV_VALUE)


def test_environment_has_three_tasks() -> None:
    env = OpenEnvRealWorldSim(seed=42)
    assert len(env.tasks) >= 3


def test_graders_depend_on_submission_quality() -> None:
    email_answers = {"email-001": "important", "email-002": "spam"}
    weak_email_score = grade_email_triage([{"id": "email-001", "label": "spam"}], email_answers)
    strong_email_score = grade_email_triage(
        [{"id": "email-001", "label": "important"}, {"id": "email-002", "label": "spam"}],
        email_answers,
    )
    assert MIN_OPENENV_VALUE <= weak_email_score < strong_email_score <= MAX_OPENENV_VALUE

    weak_csv_score = grade_cleaned_csv("customer_id,signup_date,country,purchase_total,status\n001,bad,USA,0,wrong")
    strong_csv_score = grade_cleaned_csv(
        "\n".join(
            [
                "shipment_id,carrier,route_family,delay_hours,risk_trigger,priority,owner_action,resolution_tag",
                *[
                    ",".join(
                        [
                            row["shipment_id"],
                            row["carrier"],
                            row["route_family"],
                            row["delay_hours"],
                            row["risk_trigger"],
                            row["priority"],
                            row["owner_action"],
                            row["resolution_tag"],
                        ]
                    )
                    for row in EXPECTED_DATA
                ],
            ]
        )
    )
    assert MIN_OPENENV_VALUE <= weak_csv_score < strong_csv_score <= MAX_OPENENV_VALUE

    weak_code_score = grade_code_review([], "")
    strong_code_score = grade_code_review(
        [
            "The cutoff window is reversed, so recent shipments are skipped while stale events are reviewed.",
            "Supplier risk is overwritten instead of accumulated with the other signals.",
            "Manual review output should sort highest-risk shipments first.",
            "The summary path needs route_family preserved on each flagged shipment.",
        ],
        """from datetime import datetime, timedelta


def select_shipments_for_manual_review(events, vendor_risk, now_iso, lookback_days=7):
    now = datetime.fromisoformat(now_iso)
    cutoff = now - timedelta(days=lookback_days)
    flagged = []

    for event in events:
        event_time = datetime.fromisoformat(event["event_time"])
        if event_time < cutoff:
            continue

        risk = 0
        if event["declared_value"] >= 15000:
            risk += 1
        if vendor_risk.get(event["supplier_id"], 0) > 0.75:
            risk += 2
        if event["seal_status"] == "broken":
            risk += 2
        if event["invoice_status"] == "missing" and event["route_family"] == "high_value":
            risk += 1
        if event["temp_c"] > 5 and event["route_family"] == "cold":
            risk += 1

        if risk >= 2:
            flagged.append(
                {
                    "shipment_id": event["shipment_id"],
                    "risk": risk,
                    "event_time": event["event_time"],
                    "route_family": event["route_family"],
                }
            )

    return sorted(flagged, key=lambda item: (-item["risk"], item["event_time"]))[:5]


def summarize_flagged_shipments(flagged):
    summary = {}
    for item in flagged:
        family = item["route_family"]
        summary[family] = summary.get(family, 0) + item["risk"]
    return sorted(summary.items(), key=lambda pair: pair[0])
""",
    )
    assert MIN_OPENENV_VALUE <= weak_code_score < strong_code_score <= MAX_OPENENV_VALUE

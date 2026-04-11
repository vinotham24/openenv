from __future__ import annotations

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
    assert observation.progress < 0.01
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
    assert info["task_score"] > 0.3
    assert info["task_score"] < 0.35


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


def test_clamp_score_excludes_zero_and_one() -> None:
    assert clamp_score(0.0) == MIN_OPENENV_VALUE
    assert clamp_score(1.0) == MAX_OPENENV_VALUE
    assert clamp_score(0.25) == 0.25


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
                "customer_id,signup_date,country,purchase_total,status",
                *[
                    ",".join(
                        [
                            row["customer_id"],
                            row["signup_date"],
                            row["country"],
                            row["purchase_total"],
                            row["status"],
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
            "Discount handling is wrong because it subtracts discount multiplied by 100.",
            "Tax handling is wrong because it adds tax as a flat value instead of applying a percentage.",
            "Sort order is wrong because results should sort by count descending and customer ascending.",
        ],
        """def calculate_invoice_total(items, tax_rate=0.1, discount=0.0):
    subtotal = 0.0
    for item in items:
        subtotal += item["price"] * item["quantity"]
    subtotal -= discount
    taxed_total = subtotal * (1 + tax_rate)
    return round(taxed_total, 2)


def summarize_orders(orders):
    summary = {}
    for order in orders:
        customer = order["customer"]
        summary[customer] = summary.get(customer, 0) + 1
    return sorted(summary.items(), key=lambda pair: (-pair[1], pair[0]))
""",
    )
    assert MIN_OPENENV_VALUE <= weak_code_score < strong_code_score <= MAX_OPENENV_VALUE

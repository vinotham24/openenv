from __future__ import annotations

from fastapi.testclient import TestClient

from env.base_env import OpenEnvRealWorldSim
from score_utils import MAX_OPENENV_VALUE, MIN_OPENENV_VALUE
from server.app import app


def test_reset_returns_observation() -> None:
    env = OpenEnvRealWorldSim(seed=42)
    observation = env.reset()
    assert observation.task_name == "email_triage"
    assert observation.progress == MIN_OPENENV_VALUE
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
    assert info["task_score"] == 0.3333


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

from __future__ import annotations

from env.base_env import OpenEnvRealWorldSim


def test_reset_returns_observation() -> None:
    env = OpenEnvRealWorldSim(seed=42)
    observation = env.reset()
    assert observation.task_name == "email_triage"
    assert observation.progress == 0.0001
    assert observation.attempts_remaining == 6


def test_step_returns_expected_tuple() -> None:
    env = OpenEnvRealWorldSim(seed=42)
    env.reset()
    observation, reward, done, info = env.step(
        {"action_type": "classify_email", "payload": {"id": "email-001", "label": "important"}}
    )
    assert observation.task_name == "email_triage"
    assert reward > 0
    assert done is False
    assert info["task_score"] == 0.3333


def test_reward_penalizes_invalid_action() -> None:
    env = OpenEnvRealWorldSim(seed=42)
    env.reset()
    _, reward, _, info = env.step({"action_type": "clean_data", "payload": {"cleaned_csv": "x"}})
    assert reward < 0
    assert info["reward"]["components"]["invalid_action"] == -0.2

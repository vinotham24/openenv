from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from env.reward import RewardTracker
from env.schemas import Action, Observation, Reward
from env.utils import setup_logger, stable_signature
from score_utils import bounded_reward, bounded_unit_interval
from tasks.code_review.task import CodeReviewTask
from tasks.data_cleaning.task import DataCleaningTask
from tasks.email_triage.task import EmailTriageTask


@dataclass
class EnvironmentSnapshot:
    current_task_index: int
    total_tasks: int
    current_task_name: str | None
    cumulative_reward: float
    done: bool


class OpenEnvRealWorldSim:
    def __init__(self, seed: int = 42, max_steps_per_task: int = 6) -> None:
        self.logger = setup_logger()
        self.seed = seed
        self.max_steps_per_task = max_steps_per_task
        self.tasks = [
            EmailTriageTask(seed=seed),
            DataCleaningTask(seed=seed),
            CodeReviewTask(seed=seed),
        ]
        self.current_task_index = 0
        self.current_task = None
        self.current_step = 0
        self.done = False
        self.cumulative_reward = bounded_reward(0.0)
        self.reward_tracker = RewardTracker()

    def reset(self) -> Observation:
        self.current_task_index = 0
        self.current_task = self.tasks[self.current_task_index]
        self.current_task.reset()
        self.current_step = 0
        self.done = False
        self.cumulative_reward = bounded_reward(0.0)
        self.reward_tracker = RewardTracker()
        return self.current_task.observation(self.max_steps_per_task)

    def step(self, action: Dict[str, Any] | Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.current_task is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self.done:
            raise RuntimeError("Environment already completed.")

        parsed = action if isinstance(action, Action) else Action.model_validate(action)
        self.current_step += 1

        result = self.current_task.apply_action(parsed)
        signature = stable_signature(parsed.model_dump())
        loop_detected = self.current_task.detect_loop(parsed)
        reward_value, components = self.reward_tracker.score(
            signature=signature,
            progress_delta=result["progress_delta"],
            valid=result["valid"],
            loop_detected=loop_detected,
        )

        reward = Reward(
            value=reward_value,
            components=components,
            message=result["message"],
        )
        self.cumulative_reward = bounded_unit_interval(self.cumulative_reward + (reward.value - 0.5))

        info = {
            "task_id": self.current_task.task_id,
            "task_name": self.current_task.task_name,
            "difficulty": self.current_task.difficulty,
            "task_score": result["score"],
            "error": result["error"],
            "details": result["details"],
            "reward": reward.model_dump(),
        }

        task_complete = result["completed"] or self.current_step >= self.max_steps_per_task
        if task_complete:
            self._advance_task()

        observation = (
            self.current_task.observation(self.max_steps_per_task)
            if not self.done
            else Observation(
                task_id="terminal",
                task_name="completed",
                difficulty="terminal",
                instruction="All tasks completed.",
                content={"state": self.state()},
                history=[],
                hints=[],
                progress=bounded_unit_interval(1.0),
                attempts_remaining=0,
            )
        )
        return observation, reward.value, self.done, info

    def state(self) -> Dict[str, Any]:
        snapshot = EnvironmentSnapshot(
            current_task_index=self.current_task_index,
            total_tasks=len(self.tasks),
            current_task_name=self.current_task.task_name if self.current_task else None,
            cumulative_reward=self.cumulative_reward,
            done=self.done,
        )
        return snapshot.__dict__

    def _advance_task(self) -> None:
        self.current_task_index += 1
        self.current_step = 0
        self.reward_tracker = RewardTracker()
        if self.current_task_index >= len(self.tasks):
            self.current_task = None
            self.done = True
            return
        self.current_task = self.tasks[self.current_task_index]
        self.current_task.reset()

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
import uvicorn

from env.base_env import OpenEnvRealWorldSim
from env.schemas import Action
from score_utils import validate_score

app = FastAPI()
ENV = OpenEnvRealWorldSim(seed=42)


def _serialize_step(
    observation: Any,
    reward: float,
    done: bool,
    info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "observation": observation.model_dump(),
        "reward": validate_score(reward),
        "done": done,
        "info": info,
        "task_score": validate_score(info["task_score"]),
    }


@app.get("/")
def root():
    return {
        "name": "openenv-realworld-sim",
        "status": "ok",
        "message": "API is running.",
        "routes": {
            "health": "/health",
            "metadata": "/metadata",
            "reset": "/reset",
            "step": "/step",
            "state": "/state",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    return {
        "name": "openenv-realworld-sim",
        "description": "OpenEnv real-world benchmark: email triage, data cleaning, code review.",
        "tasks": [
            {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "difficulty": task.difficulty,
            }
            for task in ENV.tasks
        ],
        "actions": list(Action.model_json_schema()["properties"]["action_type"]["enum"]),
    }


@app.post("/reset")
def reset():
    observation = ENV.reset()
    return {
        "observation": observation.model_dump(),
        "reward": validate_score(0.5),
        "done": False,
        "info": {
            "task_id": observation.task_id,
            "task_name": observation.task_name,
            "difficulty": observation.difficulty,
            "task_score": validate_score(observation.progress),
            "error": None,
            "details": {"reset": True},
            "reward": {
                "value": validate_score(0.5),
                "components": {},
                "message": "Environment reset.",
            },
        },
        "task_score": validate_score(observation.progress),
    }


@app.post("/step")
def step(action: Action):
    observation, reward, done, info = ENV.step(action)
    return _serialize_step(observation, reward, done, info)


@app.get("/state")
def state():
    return ENV.state()


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from score_utils import bounded_unit_interval

app = FastAPI()

# Simple in-memory environment used by the OpenEnv validator.
emails = [
    {"id": "email-001", "text": "Project deadline tomorrow", "label": "important"},
    {"id": "email-002", "text": "Win a free iPhone", "label": "spam"},
    {"id": "email-003", "text": "Can we schedule a meeting?", "label": "respond"},
]

current_step = 0


class Action(BaseModel):
    action_type: str
    payload: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    return {
        "name": "openenv-email-triage",
        "description": "Minimal OpenEnv environment",
    }


@app.post("/reset")
def reset():
    global current_step
    current_step = 0
    return {
        "observation": emails[current_step],
        "reward": bounded_unit_interval(0.0),
        "done": False,
    }


@app.post("/step")
def step(action: Action):
    global current_step

    correct_label = emails[current_step]["label"]
    predicted = action.payload.get("label")

    reward = bounded_unit_interval(1.0 if predicted == correct_label else 0.0)

    current_step += 1
    done = current_step >= len(emails)
    observation = None if done else emails[current_step]

    return {
        "observation": observation,
        "reward": reward,
        "done": done,
    }


@app.get("/state")
def state():
    return {"step": current_step}


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

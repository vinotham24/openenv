from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from score_utils import MAX_TASK_SCORE, bounded_reward, bounded_unit_interval
from tasks.email_triage.grader import grade_email_triage
from tasks.data_cleaning.grader import grade_cleaned_csv
from tasks.code_review.grader import grade_code_review

app = FastAPI()

# ── Email triage state ────────────────────────────────────────────────────────
EMAIL_ANSWERS = {
    "email-001": "important",
    "email-002": "spam",
    "email-003": "respond",
}
email_predictions: list[dict] = []

# ── Data cleaning state ───────────────────────────────────────────────────────
cleaned_csv: str = ""

# ── Code review state ─────────────────────────────────────────────────────────
submitted_bugs: list[str] = []
submitted_fixed_code: str = ""

# ── Step counter ──────────────────────────────────────────────────────────────
current_step: int = 0


class Action(BaseModel):
    action_type: str
    payload: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    return {
        "name": "openenv-realworld-sim",
        "description": "OpenEnv real-world benchmark: email triage, data cleaning, code review.",
        "tasks": ["email_triage", "data_cleaning", "code_review"],
    }


@app.post("/reset")
def reset():
    global current_step, email_predictions, cleaned_csv, submitted_bugs, submitted_fixed_code
    current_step = 0
    email_predictions = []
    cleaned_csv = ""
    submitted_bugs = []
    submitted_fixed_code = ""
    return {
        "observation": {"task": "email_triage", "step": current_step},
        "task_score": bounded_unit_interval(0.0),
        "reward": bounded_reward(0.0),
        "done": False,
    }


@app.post("/step")
def step(action: Action):
    global current_step, email_predictions, cleaned_csv, submitted_bugs, submitted_fixed_code

    current_step += 1
    reward = bounded_reward(0.0)
    task_score = bounded_unit_interval(0.0)
    done = False
    observation: dict = {}

    action_type = action.action_type
    payload = action.payload

    if action_type == "classify_email":
        email_id = payload.get("id")
        label = payload.get("label")
        if email_id and label:
            email_predictions = [p for p in email_predictions if p.get("id") != email_id]
            email_predictions.append({"id": email_id, "label": label})
        raw = grade_email_triage(email_predictions, EMAIL_ANSWERS)
        task_score = bounded_unit_interval(raw)
        reward = bounded_reward(raw)
        done = len(email_predictions) >= len(EMAIL_ANSWERS)
        observation = {"classified": email_id, "label": label}

    elif action_type == "clean_data":
        csv_text = payload.get("cleaned_csv", "")
        if csv_text:
            cleaned_csv = csv_text.strip()
        raw = grade_cleaned_csv(cleaned_csv) if cleaned_csv else 0.0
        task_score = bounded_unit_interval(raw)
        reward = bounded_reward(raw)
        done = raw >= MAX_TASK_SCORE
        observation = {"rows_submitted": len(cleaned_csv.splitlines()) - 1 if cleaned_csv else 0}

    elif action_type == "review_code":
        bugs = payload.get("bugs", [])
        fixed_code = payload.get("fixed_code", "")
        if bugs:
            submitted_bugs = bugs
        if fixed_code:
            submitted_fixed_code = fixed_code.strip()
        raw = grade_code_review(submitted_bugs, submitted_fixed_code)
        task_score = bounded_unit_interval(raw)
        reward = bounded_reward(raw)
        done = raw >= MAX_TASK_SCORE
        observation = {"bugs_reported": len(submitted_bugs)}

    elif action_type == "submit":
        done = True
        email_raw = grade_email_triage(email_predictions, EMAIL_ANSWERS)
        data_raw = grade_cleaned_csv(cleaned_csv) if cleaned_csv else 0.0
        code_raw = grade_code_review(submitted_bugs, submitted_fixed_code)
        best = max(email_raw, data_raw, code_raw)
        task_score = bounded_unit_interval(best)
        reward = bounded_reward(best)
        observation = {"submitted": True}

    elif action_type == "analyze":
        task_score = bounded_unit_interval(0.5)
        reward = bounded_reward(0.1)
        observation = {"analysis": "Inspected task inputs."}

    else:
        task_score = bounded_unit_interval(0.0)
        reward = bounded_reward(0.0)
        observation = {"error": f"unknown action_type: {action_type}"}

    return {
        "observation": observation,
        "task_score": task_score,
        "reward": reward,
        "done": done,
    }


@app.get("/state")
def state():
    return {
        "step": current_step,
        "email_predictions": len(email_predictions),
        "has_cleaned_csv": bool(cleaned_csv),
        "has_code_review": bool(submitted_bugs or submitted_fixed_code),
    }


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

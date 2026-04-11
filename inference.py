from __future__ import annotations

import csv
import json
import os
from io import StringIO
from typing import Any, Dict, List

from openai import OpenAI

from env.base_env import OpenEnvRealWorldSim
from env.schemas import Action
from env.utils import lowercase_bool
from score_utils import COMPLETION_SCORE_THRESHOLD


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

SYSTEM_PROMPT = (
    "You are an evaluation agent for a deterministic OpenEnv benchmark. "
    "Return JSON only with keys action_type, payload, reasoning."
)


def heuristic_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    task_name = observation["task_name"]
    history = observation.get("history", [])

    if task_name == "email_triage":
        completed_ids = {
            item.get("payload", {}).get("id")
            for item in history
            if item.get("action_type") == "classify_email"
        }
        for email in observation["content"]["emails"]:
            if email["id"] in completed_ids:
                continue
            text = f'{email["subject"]} {email["body"]} {email.get("thread_hint", "")}'.lower()
            if any(term in text for term in ("password", "verify", "portal mirror", "wire batch", "reimbursement")):
                label = "spam"
            elif any(term in text for term in ("blocked", "executive", "deadline")):
                label = "important"
            else:
                label = "respond"
            return {
                "action_type": "classify_email",
                "payload": {"id": email["id"], "label": label},
                "reasoning": "Heuristic classification using only obvious urgency and phishing cues.",
            }
        return {"action_type": "submit", "payload": {}, "reasoning": "All visible emails handled."}

    if task_name == "data_cleaning":
        if any(item.get("action_type") == "clean_data" for item in history):
            return {"action_type": "submit", "payload": {}, "reasoning": "Baseline already submitted one cleaned extract."}

        reader = csv.DictReader(StringIO(observation["content"]["raw_csv"].strip()))
        rows = ["shipment_id,carrier,route_family,delay_hours,risk_trigger,priority,owner_action,resolution"]
        for row in reader:
            rows.append(
                ",".join(
                    [
                        row["shipment_id"].strip(),
                        row["carrier_alias"].strip().title(),
                        row["route_family"].strip().lower(),
                        "0.0",
                        "none",
                        "monitor",
                        "transport_control",
                        "hold_for_next_scan",
                    ]
                )
            )
        return {
            "action_type": "clean_data",
            "payload": {"cleaned_csv": "\n".join(rows)},
            "reasoning": "Baseline normalization covers obvious fields but skips deeper rule synthesis.",
        }

    if any(item.get("action_type") == "review_code" for item in history):
        return {"action_type": "submit", "payload": {}, "reasoning": "Baseline already submitted a review patch."}

    fixed_code = '''from datetime import datetime, timedelta


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
            risk =+ 2
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
'''
    return {
        "action_type": "review_code",
        "payload": {
            "bugs": [
                "The manual review queue skips recent events instead of excluding only stale ones.",
                "The review queue order is wrong for analyst priority.",
                "The summary function assumes route_family exists on every flagged record.",
            ],
            "fixed_code": fixed_code,
        },
        "reasoning": "Baseline patch addresses only the most obvious review-pipeline bugs.",
    }


def llm_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(observation, ensure_ascii=True)},
        ],
        temperature=0,
    )
    return json.loads(response.output_text)


def choose_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    try:
        action = llm_action(observation)
        Action.model_validate(action)
        return action
    except Exception:
        action = heuristic_action(observation)
        Action.model_validate(action)
        return action


def compact_action(action: Dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=True, separators=(",", ":"))


def print_start(task_name: str) -> None:
    print(f"[START] task={task_name} env=openenv model={MODEL_NAME}")


def print_step(step: int, action: Dict[str, Any], reward: float, done: bool, error: str | None) -> None:
    error_text = "null" if error is None else error
    print(
        f"[STEP] step={step} action={compact_action(action)} reward={reward:.2f} "
        f"done={lowercase_bool(done)} error={error_text}"
    )


def print_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={lowercase_bool(success)} steps={steps} rewards={rewards_text}")


def run_task(task_index: int) -> None:
    env = OpenEnvRealWorldSim(seed=42)
    env.tasks = [env.tasks[task_index]]
    env.reset()
    observation = env.current_task.observation(env.max_steps_per_task)
    print_start(observation.task_name)

    rewards: List[float] = []
    steps = 0
    success = False

    try:
        done = False
        while not done and steps < env.max_steps_per_task:
            action = choose_action(observation.model_dump())
            steps += 1
            next_observation, reward, done, info = env.step(action)
            rewards.append(reward)
            task_done = info["task_score"] >= COMPLETION_SCORE_THRESHOLD
            print_step(steps, action, reward, task_done, info["error"])
            observation = next_observation
            success = task_done
        print_end(success, steps, rewards)
    except Exception:
        print_end(False, steps, rewards)


def main() -> None:
    env = OpenEnvRealWorldSim(seed=42)
    env.reset()
    for index in range(len(env.tasks)):
        run_task(index)


if __name__ == "__main__":
    main()

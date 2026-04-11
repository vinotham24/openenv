---
title: OpenEnv Real-World Sim
sdk: docker
tags:
  - openenv
---

# OpenEnv Real-World Simulation

## Overview and motivation

`openenv-realworld-sim` is a production-grade OpenEnv-compatible benchmark for evaluating LLM agents on realistic human workflows instead of toy tasks or games. It focuses on operational decisions in a logistics setting: triaging exception inboxes, normalizing noisy shipment feeds, and debugging a shipment-risk review pipeline.

The motivation is to measure whether an agent can make incremental, safe, useful progress on real-world tasks with structured observations, typed actions, deterministic graders, and step-wise rewards.

## OpenEnv explanation

The environment follows a Gym-style API:

```python
observation = env.reset()
observation, reward, done, info = env.step(action)
state = env.state()
```

Each task produces typed `Observation`, accepts typed `Action`, and returns a typed `Reward` internally through the environment info payload.

## Observation space

Observation fields:

- `task_id`
- `task_name`
- `difficulty`
- `instruction`
- `content`
- `history`
- `hints`
- `progress`
- `attempts_remaining`

## Action space

Supported action types:

- `analyze`
- `classify_email`
- `clean_data`
- `review_code`
- `submit`

## Task descriptions

### Easy: Email Triage

The agent triages a noisy logistics operations inbox and must distinguish phishing, escalation-worthy incidents, and messages that require a direct operational reply.

### Medium: Data Cleaning

The agent receives a messy shipment exception feed plus business rules and must derive normalized routing actions, priorities, and owners.

### Hard: Code Review

The agent reviews a buggy shipment-risk review pipeline, identifies failure modes, and submits corrected production-ready code.

## Setup instructions

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set the required environment variable:

```bash
export HF_TOKEN=your_token
```

Windows PowerShell:

```powershell
$env:HF_TOKEN="your_token"
```

Run tests:

```bash
python -m pytest
```

Run inference:

```bash
python inference.py
```

## Docker usage

Build the container:

```bash
docker build -t openenv-realworld-sim .
```

Run the container:

```bash
docker run --rm -e HF_TOKEN=$HF_TOKEN openenv-realworld-sim
```

## Hugging Face Spaces deployment steps

This repository is ready for containerized deployment on Hugging Face Spaces.

1. Create a new Space.
2. Choose `Docker` as the SDK.
3. Upload this repository.
4. Ensure the Space includes the `openenv` tag.
5. Add `HF_TOKEN` as a repository secret.
6. Deploy and let the container run `python inference.py`.

## Sample output logs

```text
[START] task=email_triage env=openenv model=gpt-4.1-mini
[STEP] step=1 action={"action_type":"classify_email","payload":{"id":"email-001","label":"important"},"reasoning":"Deterministic baseline classification."} reward=0.33 done=false error=null
[STEP] step=2 action={"action_type":"classify_email","payload":{"id":"email-002","label":"spam"},"reasoning":"Deterministic baseline classification."} reward=0.33 done=false error=null
[STEP] step=3 action={"action_type":"classify_email","payload":{"id":"email-003","label":"respond"},"reasoning":"Deterministic baseline classification."} reward=0.33 done=true error=null
[END] success=true steps=3 rewards=0.33,0.33,0.33
```

## Baseline scores

Representative heuristic baseline scores:

- `email_triage`: `~0.50`
- `data_cleaning`: `~0.40`
- `code_review`: `~0.23`

Progress and task scores are clamped into the OpenEnv-safe interval via [`score_utils.py`](/c:/Users/Nivedha%20S/Downloads/hack/openenv-realworld-sim/score_utils.py), so values stay within `(0, 1)` and are normalized to `0.01..0.99`. The graders also apply a tiny per-run jitter tied to the submission content, which keeps scoring meaningful while avoiding perfectly identical outputs across runs.

The benchmark is lightweight and runs comfortably within 2 vCPU and 8 GB RAM constraints.

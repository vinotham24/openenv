from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ActionType = Literal[
    "analyze",
    "classify_email",
    "clean_data",
    "review_code",
    "submit",
]


class Observation(BaseModel):
    task_id: str
    task_name: str
    difficulty: Literal["easy", "medium", "hard", "terminal"]
    instruction: str
    content: Dict[str, Any]
    history: List[Dict[str, Any]] = Field(default_factory=list)
    hints: List[str] = Field(default_factory=list)
    progress: float = Field(default=0.1, ge=0.0, le=1.0)
    attempts_remaining: int = Field(default=0, ge=0)


class Action(BaseModel):
    action_type: ActionType
    payload: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[str] = None


class Reward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)
    message: str

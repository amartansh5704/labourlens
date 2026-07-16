# api/schemas/request.py
# Defines what data the API accepts in requests
# Pydantic validates all incoming data automatically

from pydantic import BaseModel, field_validator
from typing import Optional
from shared.constants import JURISDICTION_NAMES, TOPIC_KEYS


class AskRequest(BaseModel):
    """
    Request body for POST /api/ask
    User sends this when asking a question
    """

    question: str
    jurisdiction: Optional[str] = None
    topic: Optional[str] = None
    top_k: int = 5

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        if len(v.strip()) > 500:
            raise ValueError(
                "Question too long. Maximum 500 characters"
            )
        return v.strip()

    @field_validator("jurisdiction")
    @classmethod
    def validate_jurisdiction(cls, v):
        if v is None:
            return v
        if v == "All":
            return None
        if v not in JURISDICTION_NAMES:
            raise ValueError(
                f"Invalid jurisdiction: {v}. "
                f"Must be one of: {JURISDICTION_NAMES}"
            )
        return v

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v):
        if v is None:
            return v
        if v == "All":
            return None
        if v not in TOPIC_KEYS:
            raise ValueError(
                f"Invalid topic: {v}. "
                f"Must be one of: {TOPIC_KEYS}"
            )
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        if v < 1:
            return 1
        if v > 10:
            return 10
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the minimum wage for unskilled workers in Delhi?",
                "jurisdiction": "Delhi",
                "topic": "minimum_wage",
                "top_k": 5
            }
        }


class CompareRequest(BaseModel):
    """
    Request body for POST /api/compare
    User sends this when comparing two jurisdictions
    """

    question: str
    jurisdiction1: str
    jurisdiction2: str
    topic: Optional[str] = None

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

    @field_validator("jurisdiction1", "jurisdiction2")
    @classmethod
    def validate_jurisdictions(cls, v):
        if v not in JURISDICTION_NAMES:
            raise ValueError(
                f"Invalid jurisdiction: {v}. "
                f"Must be one of: {JURISDICTION_NAMES}"
            )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the minimum wage?",
                "jurisdiction1": "Delhi",
                "jurisdiction2": "Maharashtra",
                "topic": "minimum_wage"
            }
        }
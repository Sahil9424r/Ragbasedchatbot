"""
app/models.py

Pydantic request/response models.
FastAPI validates all incoming requests against these automatically —
invalid payloads return a 422 error before reaching any business logic.
"""

from pydantic import BaseModel
from typing import Optional


# ── Session ──────────────────────────────
class SessionCreateResponse(BaseModel):
    session_id: str


# ── Document QA ──────────────────────────
class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class QuestionResponse(BaseModel):
    answer: str


# ── Summarizer ───────────────────────────
class SummarizeRequest(BaseModel):
    text: str
    session_id: Optional[str] = None

class SummarizeResponse(BaseModel):
    summary: str


# ── Career ───────────────────────────────
class CareerRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class CareerResponse(BaseModel):
    advice: str


# ── Personal Support ─────────────────────
class PersonalRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class PersonalResponse(BaseModel):
    reply: str


# ── Entertainment ────────────────────────
class EntertainmentRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class EntertainmentResponse(BaseModel):
    recommendation: str


# ── MCQ & Notes ──────────────────────────
class MCQRequest(BaseModel):
    text: str
    session_id: Optional[str] = None

class MCQResponse(BaseModel):
    result: str

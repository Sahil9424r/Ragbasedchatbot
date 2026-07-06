from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn

from app.models import (
    QuestionRequest, QuestionResponse,
    SummarizeRequest, SummarizeResponse,
    CareerRequest, CareerResponse,
    PersonalRequest, PersonalResponse,
    EntertainmentRequest, EntertainmentResponse,
    MCQRequest, MCQResponse,
    SessionCreateResponse
)
from app.services import (
    process_uploaded_files,
    answer_from_documents,
    summarize_text,
    career_counseling_response,
    personal_support_response,
    entertainment_response,
    generate_mcq_and_notes
)
from app.session import session_manager
from app.state import app_state

# ─────────────────────────────────────────
# Lifespan — runs at startup & shutdown
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources ONCE at server startup — before any user hits the API."""
    print("🚀 Loading embedding model at startup...")
    app_state.load_embeddings()          # ~90MB HuggingFace model — cached globally
    print("✅ Embedding model ready. Server is live.")
    yield
    # Shutdown cleanup
    print("🛑 Shutting down. Clearing state.")
    app_state.clear()

# ─────────────────────────────────────────
# Rate Limiter — mirrors Gemini free tier
# ─────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ─────────────────────────────────────────
# App Init
# ─────────────────────────────────────────
app = FastAPI(
    title="RAG AI Assistant",
    description="Production-grade RAG backend with FastAPI, FAISS, and Gemini",
    version="2.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ─────────────────────────────────────────
# Root — serves frontend
# ─────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ─────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "embedding_model": "loaded" if app_state.embeddings else "not loaded",
        "faiss_index": "loaded" if app_state.faiss_index else "not loaded"
    }

# ─────────────────────────────────────────
# Session Management
# ─────────────────────────────────────────
@app.post("/session/create", response_model=SessionCreateResponse)
async def create_session():
    """Each user gets a unique session ID to isolate their chat history."""
    session_id = session_manager.create_session()
    return SessionCreateResponse(session_id=session_id)

@app.get("/session/{session_id}/history")
async def get_history(session_id: str):
    history = session_manager.get_history(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": history}

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    session_manager.delete_session(session_id)
    return {"message": "Session deleted"}

# ─────────────────────────────────────────
# Tab 1 — Document Upload & QA
# ─────────────────────────────────────────
@app.post("/documents/upload")
@limiter.limit("10/minute")
async def upload_documents(request: Request, files: list[UploadFile] = File(...)):
    """
    Process uploaded PDF/TXT/DOCX files.
    Builds FAISS index from extracted text — stored in app_state (shared, in-memory).
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    await process_uploaded_files(files)
    return {"message": f"✅ {len(files)} file(s) processed and indexed successfully"}


@app.post("/documents/ask", response_model=QuestionResponse)
@limiter.limit("60/minute")
async def ask_document(request: Request, body: QuestionRequest):
    """
    Stateless FAISS similarity search + Gemini LLM call.
    Read-only on shared index — safe for concurrent requests.
    """
    if not app_state.faiss_index:
        raise HTTPException(status_code=400, detail="No documents uploaded yet. Please upload files first.")

    answer = await answer_from_documents(body.question)

    # Save to session history
    session_manager.append_history(body.session_id, body.question, answer)

    return QuestionResponse(answer=answer)


# ─────────────────────────────────────────
# Tab 2 — Summarizer
# ─────────────────────────────────────────
@app.post("/summarize", response_model=SummarizeResponse)
@limiter.limit("60/minute")
async def summarize(request: Request, body: SummarizeRequest):
    """Summarize any given text using Gemini."""
    summary = await summarize_text(body.text)
    session_manager.append_history(body.session_id, body.text[:100] + "...", summary)
    return SummarizeResponse(summary=summary)


# ─────────────────────────────────────────
# Tab 3 — Career Counseling
# ─────────────────────────────────────────
@app.post("/career", response_model=CareerResponse)
@limiter.limit("60/minute")
async def career(request: Request, body: CareerRequest):
    """Career counseling powered by Gemini."""
    advice = await career_counseling_response(body.question)
    session_manager.append_history(body.session_id, body.question, advice)
    return CareerResponse(advice=advice)


# ─────────────────────────────────────────
# Tab 4 — Personal & Emotional Support
# ─────────────────────────────────────────
@app.post("/personal", response_model=PersonalResponse)
@limiter.limit("60/minute")
async def personal(request: Request, body: PersonalRequest):
    """Empathetic personal support responses."""
    reply = await personal_support_response(body.question)
    session_manager.append_history(body.session_id, body.question, reply)
    return PersonalResponse(reply=reply)


# ─────────────────────────────────────────
# Tab 5 — Entertainment
# ─────────────────────────────────────────
@app.post("/entertainment", response_model=EntertainmentResponse)
@limiter.limit("60/minute")
async def entertainment(request: Request, body: EntertainmentRequest):
    """Movie, show, music recommendations."""
    recommendation = await entertainment_response(body.question)
    session_manager.append_history(body.session_id, body.question, recommendation)
    return EntertainmentResponse(recommendation=recommendation)


# ─────────────────────────────────────────
# Tab 6 — MCQ & Notes Generator
# ─────────────────────────────────────────
@app.post("/mcq", response_model=MCQResponse)
@limiter.limit("60/minute")
async def mcq(request: Request, body: MCQRequest):
    """Generate MCQs and short notes from study material."""
    result = await generate_mcq_and_notes(body.text)
    session_manager.append_history(body.session_id, body.text[:100] + "...", result)
    return MCQResponse(result=result)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

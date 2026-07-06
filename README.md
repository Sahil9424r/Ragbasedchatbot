# RAG AI Assistant — FastAPI Version

Production-grade rewrite of the Streamlit RAG chatbot using FastAPI.

## Architecture Changes from Streamlit

| Feature | Streamlit (old) | FastAPI (new) |
|---|---|---|
| Concurrency | Thread per session | Async event loop |
| Model loading | `@st.cache_resource` (on first request) | Lifespan startup event (before first request) |
| Session state | `st.session_state` (auto) | UUID-keyed dict (explicit) |
| Rate limiting | None | `slowapi` middleware (60 req/min) |
| FAISS per request | Reloaded from disk | Loaded once into `app_state` |
| Frontend | Tied to Python | Decoupled HTML/JS |

---

## Project Structure

```
rag-fastapi/
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI app, routes, lifespan
│   ├── services.py    # All business logic (async)
│   ├── models.py      # Pydantic request/response models
│   ├── session.py     # Per-user session management
│   └── state.py       # Global shared state (embeddings, FAISS)
├── templates/
│   └── index.html     # Frontend UI
├── static/            # CSS/JS assets (if any)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

```bash
# 1. Clone and enter project
cd rag-fastapi

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 5. Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server health check |
| POST | `/session/create` | Create user session |
| GET | `/session/{id}/history` | Get chat history |
| POST | `/documents/upload` | Upload & index files |
| POST | `/documents/ask` | Ask from documents |
| POST | `/summarize` | Summarize text |
| POST | `/career` | Career counseling |
| POST | `/personal` | Personal support |
| POST | `/entertainment` | Entertainment recommendations |
| POST | `/mcq` | Generate MCQs & notes |

Interactive docs: `http://localhost:8000/docs`

---

## Production Deployment

```bash
# Multiple workers — bypasses Python GIL entirely
gunicorn app.main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000

# Docker
docker build -t rag-assistant .
docker run -p 8000:8000 --env-file .env rag-assistant
```

---

## Scaling Path

| Scale | Solution |
|---|---|
| >60 req/min | Upgrade Gemini API to paid tier |
| Multi-instance | Replace session dict with Redis |
| Multi-tenant FAISS | Per-user indexes or Pinecone/Weaviate |
| High traffic | Kubernetes + load balancer |

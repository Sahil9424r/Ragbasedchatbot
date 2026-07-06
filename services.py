"""
app/services.py

Core business logic — async versions of your original Streamlit functions.

Key changes from Streamlit version:
  1. All functions are async — FastAPI's event loop handles concurrency
  2. Embedding model loaded from app_state (not reloaded per request)
  3. FAISS index stored in app_state (not reloaded from disk per request)
  4. LangChain calls wrapped with asyncio.run_in_executor where needed
     (LangChain is sync — we offload to thread pool to avoid blocking event loop)
"""

import asyncio
from io import BytesIO
from typing import List

from fastapi import UploadFile
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from docx import Document

from app.state import app_state


# ─────────────────────────────────────────
# File Text Extraction (same as your original)
# ─────────────────────────────────────────
def _extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract raw text from PDF, TXT, or DOCX."""
    ext = filename.split('.')[-1].lower()
    text = ""

    if ext == 'pdf':
        reader = PdfReader(BytesIO(file_bytes))
        for page in reader.pages:
            text += page.extract_text() or ""

    elif ext == 'txt':
        text = file_bytes.decode("utf-8")

    elif ext == 'docx':
        doc = Document(BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"

    return text


def _chunk_text(text: str) -> list:
    """Split text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)


def _build_faiss_index(chunks: list) -> FAISS:
    """
    Build FAISS vector store from text chunks.
    Uses the globally cached embedding model from app_state —
    no model reload per request.
    """
    return FAISS.from_texts(chunks, embedding=app_state.embeddings)


# ─────────────────────────────────────────
# Document Processing
# ─────────────────────────────────────────
async def process_uploaded_files(files: List[UploadFile]):
    """
    Read uploaded files, extract text, build FAISS index.
    Stores result in app_state.faiss_index (shared, in-memory).

    Heavy CPU work (FAISS build) is offloaded to thread pool
    so it doesn't block the async event loop.
    """
    raw_text = ""
    for file in files:
        file_bytes = await file.read()
        raw_text += _extract_text_from_file(file_bytes, file.filename) + "\n"

    chunks = _chunk_text(raw_text)

    # Offload CPU-bound FAISS build to thread pool
    loop = asyncio.get_event_loop()
    index = await loop.run_in_executor(None, _build_faiss_index, chunks)

    app_state.set_faiss_index(index)


# ─────────────────────────────────────────
# Document QA
# ─────────────────────────────────────────
def _get_qa_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    Make sure to provide all the details.
    If the answer is not in the context, say: "Answer is not available in the context."
    Do not make up answers.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


async def answer_from_documents(question: str) -> str:
    """
    Stateless FAISS similarity search + Gemini LLM call.

    FAISS search is read-only on the shared index — safe for concurrent requests.
    LangChain chain is created fresh per request (stateless, no shared mutable state).
    Gemini API call is I/O bound — event loop handles concurrency during await.
    """
    loop = asyncio.get_event_loop()

    # Similarity search — offload to thread pool (FAISS is sync)
    docs = await loop.run_in_executor(
        None,
        lambda: app_state.faiss_index.similarity_search(question)
    )

    chain = _get_qa_chain()

    # LangChain chain call — offload to thread pool
    response = await loop.run_in_executor(
        None,
        lambda: chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    )

    return response["output_text"]


# ─────────────────────────────────────────
# Summarizer
# ─────────────────────────────────────────
def _get_summarize_chain():
    prompt_template = """
    Summarize the following text in a clear and concise way:

    Text:
    {text}

    Summary:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    return LLMChain(llm=model, prompt=prompt)


async def summarize_text(text: str) -> str:
    """Summarize text — LangChain call offloaded to thread pool."""
    chain = _get_summarize_chain()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: chain.run(text=text))


# ─────────────────────────────────────────
# Career Counseling
# ─────────────────────────────────────────
def _get_career_chain():
    prompt_template = """
    You are a helpful and professional career counselor.
    Answer the following query with actionable advice and clarity.

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    return LLMChain(llm=model, prompt=prompt)


async def career_counseling_response(question: str) -> str:
    chain = _get_career_chain()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: chain.run(question=question))


# ─────────────────────────────────────────
# Personal & Emotional Support
# ─────────────────────────────────────────
def _get_personal_chain():
    prompt_template = """
    You are a friendly, understanding, and supportive AI assistant.
    A user is asking a personal or emotional question.
    Respond with empathy, encouragement, and practical advice.

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    return LLMChain(llm=model, prompt=prompt)


async def personal_support_response(question: str) -> str:
    chain = _get_personal_chain()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: chain.run(question=question))


# ─────────────────────────────────────────
# Entertainment
# ─────────────────────────────────────────
def _get_entertainment_chain():
    prompt_template = """
    You are a friendly, understanding, and supportive AI assistant.
    A user is asking entertainment related questions about songs, films,
    web series, TV shows, web shows, short films, recommendations.
    Also give recommendations about films/shows related to a particular topic or genre.
    Respond with entertaining, encouraging, and practical advice.

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    return LLMChain(llm=model, prompt=prompt)


async def entertainment_response(question: str) -> str:
    chain = _get_entertainment_chain()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: chain.run(question=question))


# ─────────────────────────────────────────
# MCQ & Notes Generator
# ─────────────────────────────────────────
def _get_mcq_chain():
    prompt_template = """
    You're a helpful educational assistant. From the text below, do the following:

    1. Generate 3 multiple choice questions (MCQs) with 4 options each
       and indicate the correct answer.
    2. Provide short notes summarizing the key points.

    Text:
    {text}

    Output:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    return LLMChain(llm=model, prompt=prompt)


async def generate_mcq_and_notes(text: str) -> str:
    chain = _get_mcq_chain()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: chain.run(text=text))

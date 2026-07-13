"""
app/state.py

Global application state — holds resources loaded ONCE at startup.

API Keys:
  GOOGLE_API_KEY    → Gemini LLM
  GROK_API_KEY      → future Grok integration  
  PINE_CONE_API_KEY → Pinecone vector database
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from typing import Optional

# ── Load all keys from .env ONCE ───────────────────────────────────────────
load_dotenv()

GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY")
GROK_API_KEY     = os.getenv("GROK_API_KEY")
PINECONE_API_KEY = os.getenv("PINE_CONE_API_KEY")

# Configure Gemini globally — all LangChain Gemini calls use this
genai.configure(api_key=GOOGLE_API_KEY)

# Pinecone config
PINECONE_INDEX_NAME = "rag-assistant"
PINECONE_DIMENSION  = 384     # must match all-MiniLM-L6-v2 output
PINECONE_METRIC     = "cosine"


class AppState:
    def __init__(self):
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.pinecone_client: Optional[Pinecone] = None
        self.pinecone_index = None

    def load_embeddings(self):
        """
        Load HuggingFace model (~90MB) ONCE at startup.
        All requests share this object — safe because
        .encode() is read-only on model weights.
        """
        print("  → Loading HuggingFace embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("  ✅ Embedding model ready.")

    def load_pinecone(self):
        """
        Connect to Pinecone cloud and get index reference ONCE at startup.
        
        Why Pinecone instead of FAISS:
          - FAISS: local, in-memory, lost on restart, single server only
          - Pinecone: cloud, persistent, works across multiple server instances,
            native concurrent reads, no disk I/O per request
        """
        print("  → Connecting to Pinecone...")
        self.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

        # Create index if it doesn't exist
        existing = [i.name for i in self.pinecone_client.list_indexes()]
        if PINECONE_INDEX_NAME not in existing:
            print(f"  → Creating index '{PINECONE_INDEX_NAME}'...")
            self.pinecone_client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=PINECONE_DIMENSION,
                metric=PINECONE_METRIC,
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        self.pinecone_index = self.pinecone_client.Index(PINECONE_INDEX_NAME)
        print(f"  ✅ Pinecone index '{PINECONE_INDEX_NAME}' connected.")

    def clear(self):
        self.embeddings = None
        self.pinecone_client = None
        self.pinecone_index = None


# Single global instance — imported by services.py and main.py
app_state = AppState()

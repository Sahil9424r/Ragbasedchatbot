"""
app/state.py

Global application state — holds resources that are:
  - Loaded ONCE at startup (lifespan event)
  - Shared READ-ONLY across all concurrent requests
  - Never modified during request handling (thread/async safe)

This replaces Streamlit's @st.cache_resource.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Optional


class AppState:
    def __init__(self):
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.faiss_index: Optional[FAISS] = None

    def load_embeddings(self):
        """
        Load HuggingFace embedding model (~90MB) into memory.
        Called ONCE at server startup via lifespan event.
        All requests share this single object — safe because
        .encode() is a read-only operation on model weights.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def set_faiss_index(self, index: FAISS):
        """
        Called after document upload/processing.
        Replaces the shared index — in production this would
        use a per-user index keyed by session_id.
        """
        self.faiss_index = index

    def clear(self):
        self.embeddings = None
        self.faiss_index = None


# Single global instance — imported everywhere
app_state = AppState()

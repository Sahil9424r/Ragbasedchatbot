"""
app/session.py

Per-user session management.
Replaces Streamlit's st.session_state.

Each user gets a UUID session_id. Their chat history is stored
in a dictionary keyed by that ID — completely isolated from other users.

Production upgrade path:
  Replace the in-memory dict with Redis:
  - Survives server restarts
  - Works across multiple server instances behind a load balancer
"""

import uuid
from typing import Optional


class SessionManager:
    def __init__(self):
        # { session_id: { "history": [(role, message), ...] } }
        self._sessions: dict = {}

    def create_session(self) -> str:
        """Generate a unique session ID and initialize empty state."""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {"history": []}
        return session_id

    def get_history(self, session_id: str) -> Optional[list]:
        """Return chat history for a session, or None if not found."""
        session = self._sessions.get(session_id)
        return session["history"] if session else None

    def append_history(self, session_id: str, user_msg: str, bot_msg: str):
        """Append a user/bot exchange to session history."""
        if session_id and session_id in self._sessions:
            self._sessions[session_id]["history"].append({
                "role": "user",
                "message": user_msg
            })
            self._sessions[session_id]["history"].append({
                "role": "bot",
                "message": bot_msg
            })

    def delete_session(self, session_id: str):
        """Clean up a session when user is done."""
        self._sessions.pop(session_id, None)

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)


# Single global instance
session_manager = SessionManager()

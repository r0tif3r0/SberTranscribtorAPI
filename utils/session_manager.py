import time
from threading import Lock
from typing import Dict, List, Optional
from langchain_core.documents import Document

from core.schemas import TranscriptSegment
from utils.rag_engine import RAGChatEngine

class SessionManager:
    def __init__(self, ttl_seconds: int = 3600):
        self._sessions: Dict[str, RAGChatEngine] = {}
        self._last_access: Dict[str, float] = {}
        self._lock = Lock()
        self.ttl = ttl_seconds

    def get_or_create_session(self, session_id: str, data: List[Document]) -> RAGChatEngine:
        with self._lock:
            if session_id in self._sessions:
                self._last_access[session_id] = time.time()
                return self._sessions[session_id]

            engine = RAGChatEngine(data)
            self._sessions[session_id] = engine
            self._last_access[session_id] = time.time()
            return engine

    def get_session(self, session_id: str) -> Optional[RAGChatEngine]:
        with self._lock:
            engine = self._sessions.get(session_id)
            if engine:
                self._last_access[session_id] = time.time()
            return engine

    def unload_session(self, session_id: str):
        with self._lock:
            engine = self._sessions.pop(session_id, None)
            self._last_access.pop(session_id, None)
        if engine:
            engine.close()
            print(f"[SessionManager] Session {session_id} выгружена")

    def clear_expired_sessions(self):
        now = time.time()
        expired_ids = []
        with self._lock:
            for session_id, last_used in self._last_access.items():
                if now - last_used > self.ttl:
                    expired_ids.append(session_id)

        for sid in expired_ids:
            self.unload_session(sid)

    def list_active_sessions(self):
        with self._lock:
            return list(self._sessions.keys())
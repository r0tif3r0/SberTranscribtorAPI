from typing import List
from fastapi.concurrency import run_in_threadpool
from langchain_core.documents import Document

from core.schemas import TranscriptSegment, PDFPage
from utils.file_manager import json_to_readable_text
from utils.session_manager import SessionManager

_session_manager = SessionManager(ttl_seconds=3600)


async def load_chat_session_audio(session_id: str, data: List[TranscriptSegment]):
    if not data:
        raise ValueError("Пустой запрос")

    doc = Document(page_content=json_to_readable_text(data))
    _session_manager.get_or_create_session(session_id, [doc])
    print(f"[ai_chat] Контекст загружен для сессии: {session_id}")
    print(f"[ai_chat] Активные сессии: {_session_manager.list_active_sessions()}")

async def load_chat_session_documents(session_id: str, data: List[Document]):
    if not data:
        raise ValueError("Пустой запрос")

    _session_manager.get_or_create_session(session_id, data)
    print(f"[ai_chat] Контекст загружен для сессии: {session_id}")
    print(f"[ai_chat] Активные сессии: {_session_manager.list_active_sessions()}")


async def ask_question(session_id: str, question: str) -> str:
    engine = _session_manager.get_session(session_id)
    if not engine:
        raise ValueError(f"Контекст для сессии '{session_id}' не загружен")

    return await run_in_threadpool(engine.ask, question)


def unload_chat_session(session_id: str):
    _session_manager.unload_session(session_id)


def cleanup_expired_sessions():
    _session_manager.clear_expired_sessions()

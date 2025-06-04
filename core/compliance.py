from typing import List
from fastapi.concurrency import run_in_threadpool

from core.schemas import TranscriptSegment
from utils.rag_engine import RAG230FZEngine

# rag_engine = RAG230FZEngine()

async def check_230_fz(json_data: List[TranscriptSegment]) -> str:
    return await run_in_threadpool(rag_engine.check_compliance, json_data)
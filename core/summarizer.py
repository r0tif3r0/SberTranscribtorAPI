from typing import List

from fastapi.concurrency import run_in_threadpool
from langchain_core.messages import HumanMessage, SystemMessage

from core.schemas import TranscriptSegment
from utils.giga_chat import get_giga_chat

system_prompt = """
Ты получаешь на входе массив JSON-записей, каждая из которых содержит транскрибацию разговора по сегментам.
Твоя задача - создать краткое, связное и информативное резюме всего разговора на основе этих сегментов.
Убедись, что суммаризация передает основные идеи и ключевые детали исходного материала.
"""

async def summarize(segments: List[TranscriptSegment]) -> str:

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=str(segments))
    ]

    try:
        response = await run_in_threadpool(get_giga_chat(temp_value=0.1, top_p_value=0.4).invoke, messages)
        print(response.content)
        summarizing = str(response.content)
    except Exception as e:
        print(f"Ошибка при транскрибировании с помощью GigaChat: {str(e)}")
        summarizing = ''

    return summarizing
from typing import List, Dict
from GigaAM import gigaam
from fastapi.concurrency import run_in_threadpool
from langchain_core.messages import HumanMessage, SystemMessage
import ast
from utils.giga_chat import get_giga_chat

system_prompt = """
Ты получаешь на входе массив JSON-записей, каждая из которых содержит транскрибацию разговора по сегментам.
Каждая JSON-запись содержит следующие поля:
'start': время начала сегмента. Формат времени: 'MM:SS.mm'.
'end': время окончания сегмента. Формат времени: 'MM:SS.mm'.
'text': текст транскрибированного сегмента. Текст является строкой без заглавных букв и знаков препинания.
'speaker': это опциональное поле, которое может быть с каким-то значением или None, указывающее, кто говорит.

ТВОЯ ЕДИНСТВЕННАЯ И ГЛАВНАЯ ЗАДАЧА: сформировать НОВЫЙ массив JSON-записей.
В НОВЫХ JSON-записях, в полях 'text', ты должен **ТОЛЬКО** расставить знаки препинания и заглавные буквы, строго согласно правилам русского языка.
КРАЙНЕ ВАЖНО: Слова в полях 'text' в новых JSON-записях **ДОЛЖНЫ ПОЛНОНОСТЬЮ ИДЕНТИЧНО СООТВЕТСТВОВАТЬ** словам исходного текста сегмента. НИКАКИХ ИЗМЕНЕНИЙ СЛОВ, ИХ ОКОНЧАНИЙ ИЛИ ПОРЯДКА!
Это означает: если в исходном тексте "привет", то в твоём ответе должно быть "Привет", а НЕ "Здравствуйте" или "Здравствуй". Сохрани каждое слово.

Если в JSON-записи исходный текст сегмента является пустой строкой, то эта запись НЕ ДОЛЖНА быть добавлена в новый массив.
Поля 'start', 'end' и 'speaker' в новых JSON-записях должны остаться БЕЗ ЕДИНЫХ ИЗМЕНЕНИЙ!
Формат вывода должен быть строго массив JSON-объектов, как на входе.
Используй только ОДИНАРНЫЕ кавычки
"""

async def process_audio(audio_path: str, model, diarize: bool, grammar: bool) -> Dict[str, List[Dict]]:

    recognition_result = await run_in_threadpool(
        model.transcribe_longform,
        audio_path,
        use_speaker_diarization=diarize
    )

    segments: List[Dict] = []
    for utterance in recognition_result:
        segment = {
            "start": gigaam.format_time(utterance["boundaries"][0]),
            "end": gigaam.format_time(utterance["boundaries"][1]),
            "text": utterance["transcription"],
            "speaker": utterance.get("speaker") if diarize else None
        }
        segments.append(segment)

    if not grammar:
        return {"transcript": segments}

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=str(segments))
    ]

    try:
        response = await run_in_threadpool(get_giga_chat(temp_value=0.1, top_p_value=0.4).invoke, messages)
        ai_segments: List[Dict] = ast.literal_eval(response.content)
    except Exception as e:
        print(f"Ошибка при транскрибировании с помощью GigaChat: {str(e)}")
        ai_segments = []

    return {"transcript": ai_segments,}

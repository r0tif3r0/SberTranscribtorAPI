import os
import tempfile
import json
from typing import List, Union, Dict

import tiktoken

from core.schemas import TranscriptSegment

FILE_PATH = '../resultFiles'

def write_to_file(text):
    i = 1
    while True:
        filename = os.path.join(FILE_PATH, f'file_{i:03d}.txt')
        if not os.path.exists(filename):
            break
        i += 1
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)
        print(f'File {filename} was created')


def save_upload_file(upload_file, tmp_dir: str) -> str:
    """
    Сохраняет UploadFile во временный файл, возвращает путь к нему.
    """
    suffix = os.path.splitext(upload_file.filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tmp_dir)
    tmp.write(upload_file.file.read())
    tmp.close()
    return tmp.name

def remove_file(path: str):
    if os.path.exists(path):
        os.unlink(path)

def json_to_readable_text(json_input: List[TranscriptSegment]) -> str:
    """
    Преобразует список объектов TranscriptSegment в человекочитаемый формат.
    """
    lines = []

    for segment in json_input:
        start = segment.start or "???"
        end = segment.end or "???"
        text = (segment.text or "").strip()
        speaker = segment.speaker

        if speaker:
            line = f"[{start} – {end}] {speaker}: {text}"
        else:
            line = f"[{start} – {end}] {text}"

        lines.append(line)

    return "\n".join(lines)

def split_text_by_token_limit(text: str, max_tokens: int = 4096, model_name: str = "gpt-3.5-turbo") -> List[str]:
    """
    Разбивает текст на части, чтобы каждая часть укладывалась в лимит по токенам.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    lines = text.split("\n")

    chunks = []
    current_chunk = []
    current_token_count = 0

    for line in lines:
        line_tokens = len(encoding.encode(line))
        if current_token_count + line_tokens > max_tokens:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_token_count = line_tokens
        else:
            current_chunk.append(line)
            current_token_count += line_tokens

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks
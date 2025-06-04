import glob
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import json
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FILES_PATH = os.path.join(BASE_DIR, "..", "docs")

def load_pdfs(folder_path: str = PDF_FILES_PATH) -> List[Document]:
    docs = []
    if os.path.isdir(folder_path):
        for pdf_file in glob.glob(f"{folder_path}/*.pdf"):
            loader = PyPDFLoader(pdf_file)
            docs.extend(loader.load())
    else:
        loader = PyPDFLoader(folder_path)
        docs = loader.load()
    print(f"Загружено {len(docs)} страниц.")
    return docs


def split_json_by_blocks(json_str: str, blocks_per_chunk: int = 5) -> List[str]:
    """
    Разбивает JSON-массив на чанки по количеству блоков (объектов в списке).

    :param json_str: Исходный JSON в виде строки
    :param blocks_per_chunk: Сколько блоков должно быть в каждом чанке
    :return: Список строк (JSON чанков)
    """
    data = json.loads(json_str)

    chunks = [
        json.dumps(data[i:i + blocks_per_chunk], ensure_ascii=False)
        for i in range(0, len(data), blocks_per_chunk)
    ]

    return chunks
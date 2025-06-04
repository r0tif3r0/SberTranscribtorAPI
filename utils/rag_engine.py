import os
from typing import List
from dotenv import load_dotenv, find_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from starlette.concurrency import run_in_threadpool
import tempfile
import shutil

from core.schemas import TranscriptSegment
from utils.docs_loader import load_pdfs
from utils.giga_chat import get_giga_chat
from utils.file_manager import json_to_readable_text, split_text_by_token_limit

load_dotenv(find_dotenv())

class RAG230FZEngine:
    def __init__(self):
        self.embeddings = GigaChatEmbeddings(
            model="EmbeddingsGigaR",
            scope="GIGACHAT_API_CORP",
            credentials=os.environ.get("GIGACHAT_CREDENTIALS"),
            verify_ssl_certs=False
        )

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        law_docs = load_pdfs()
        splits = text_splitter.split_documents(law_docs)
        self.vectorstore = Chroma.from_documents(documents=splits, embedding=self.embeddings)
        retriever = self.vectorstore.as_retriever()

        self.system_prompt = (
            "Ты получаешь на вход записи разговора, они представляют собой текстовую транскрибацию разговора по сегментам.\n"
            "Ты должен решить, соблюдается ли закон 230-ФЗ в предоставленном тебе текста разговора.\n"
            "Отвечай коротко и по существу в 2-3 предложения.\n"
            "Вот текст закона, который ты должен использовать для ответа:\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
        ])

        giga = get_giga_chat()
        qa_chain = create_stuff_documents_chain(giga, prompt)
        self.rag_chain = create_retrieval_chain(retriever, qa_chain)

    def check_compliance(self, conversation_json: List[TranscriptSegment]) -> str:
        readable_text = json_to_readable_text(conversation_json)
        text_chunks = split_text_by_token_limit(readable_text, max_tokens=4096)

        intermediate_answers = []
        for chunk in text_chunks:
            result = self.rag_chain.invoke({"input": chunk})
            intermediate_answers.append(result["answer"])

        if len(intermediate_answers) == 1:
            return intermediate_answers[0]

        aggregation_prompt = SystemMessage(
            content="Ты получил несколько ответов по частям разговора. Объедини их в единое логическое суждение: соблюдается ли закон 230-ФЗ?"
        )
        user_message = HumanMessage(content="\n\n".join(intermediate_answers))

        giga = get_giga_chat()

        try:
            response = run_in_threadpool(giga.invoke, [aggregation_prompt, user_message])
            final_result = response.result().content
        except Exception as e:
            final_result = f"Ошибка при агрегации ответов: {str(e)}"

        return final_result


class RAGChatEngine:
    def __init__(self, base_docs: List[Document]):
        self.embeddings = GigaChatEmbeddings(
            model="EmbeddingsGigaR",
            scope="GIGACHAT_API_CORP",
            credentials=os.environ.get("GIGACHAT_CREDENTIALS"),
            verify_ssl_certs=False
        )

        self.temp_dir = tempfile.mkdtemp()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(base_docs)

        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.temp_dir
        )
        retriever = self.vectorstore.as_retriever()

        self.system_prompt = (
            "Ты ассистент, который отвечает на вопросы, используя только предоставленный ниже текст.\n"
            "Текст представляет из себя сегменты телефонного разговора в формате ([время сегмента] текст) или ([время сегмента]: спикер текст).\n"
            "Отвечай кратко, четко и по существу. Если ты не знаешь ответа, скажи честно, что информации недостаточно.\n\n"
            "В конце ответа указывай время сегмента, где ты нашел ответ.\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
        ])

        giga = get_giga_chat()
        qa_chain = create_stuff_documents_chain(giga, prompt)
        self.rag_chain = create_retrieval_chain(retriever, qa_chain)

    def ask(self, question: str) -> str:
        try:
            result = self.rag_chain.invoke({"input": question})
            return result["answer"]
        except Exception as e:
            return f"Ошибка при выполнении запроса: {str(e)}"

    def close(self):
        try:
            if hasattr(self.vectorstore, "persist"):
                self.vectorstore.persist()
            if hasattr(self.vectorstore, "close"):
                self.vectorstore.close()
        except Exception as e:
            print(f"[RAGChatEngine] Ошибка при закрытии vectorstore: {e}")

        try:
            shutil.rmtree(self.temp_dir)
            print(f"[RAGChatEngine] Временная директория {self.temp_dir} удалена")
        except Exception as e:
            print(f"[RAGChatEngine] Ошибка при удалении временной директории: {e}")

    def __del__(self):
        try:
            self.close()
        except:
            pass
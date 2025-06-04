from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
import os
import tempfile
from typing import List, Dict

from pydantic import BaseModel

from core.ai_chat import ask_question, load_chat_session_audio, load_chat_session_documents
from core.compliance import check_230_fz
from core.summarizer import summarize
from core.transcriber import process_audio
from core.schemas import TranscriptSegment, PDFPage
from utils.docs_loader import load_pdfs


class LoadAudioChatRequest(BaseModel):
    session_id: str
    data: List[TranscriptSegment]

class LoadDocumentChatRequest(BaseModel):
    session_id: str
    data: List[TranscriptSegment]

class AskChatRequest(BaseModel):
    session_id: str
    question: str

def create_router(model):
    router = APIRouter()

    @router.post(
        "/transcribe",
        response_model=Dict[str, List[TranscriptSegment]],
        response_model_exclude_none=True
    )
    async def transcribe_audio(
            file: UploadFile = File(...),
            diarize: bool = Query(True),
            grammar: bool = Query(True)
    ):
        # Validate file format
        if not file.filename.lower().endswith((".wav", ".m4a", ".mp3")):
            raise HTTPException(400, detail="Unsupported file format")

        # Save upload to a temporary file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = await process_audio(tmp_path, model, diarize, grammar)
            return result
        except Exception as e:
            raise HTTPException(500, detail=f"Processing error: {str(e)}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @router.post(
        "/summarize",
        response_model=str,
    )
    async def summarize_segments(
            segments: List[TranscriptSegment]
    ):
        if not segments:
            raise HTTPException(400, detail="No segments provided for summarization.")

        try:
            summarized_text = await summarize(segments)
            if not summarized_text:
                raise HTTPException(500, detail="Summarization returned empty result.")
            return summarized_text
        except Exception as e:
            print(f"Error during summarization: {e}")
            raise HTTPException(500, detail=f"Summarization error: {str(e)}")

    @router.post(
        "/check-230fz",
        response_model=str,
    )
    async def check_for_230_fz(data: List[TranscriptSegment]):
        if not data:
            raise HTTPException(status_code=400, detail="No data provided for checking.")

        try:
            result = await check_230_fz(data)
            if not result:
                raise HTTPException(500, detail="Checking returned empty result.")
            return result
        except Exception as e:
            print(f"Error during summarization: {e}")
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


    @router.post("/chat/load")
    async def load_chat_context(request: LoadAudioChatRequest):
        try:
            await load_chat_session_audio(request.session_id, request.data)
            return {"message": f"Контекст успешно загружен в сессию {request.session_id}"}
        except Exception as e:
            raise HTTPException(500, detail=f"Ошибка загрузки чата: {str(e)}")

    @router.post("/chat/ask")
    async def ask_chat(request: AskChatRequest):
        try:
            answer = await ask_question(request.session_id, request.question)
            return {"answer": answer}
        except ValueError as ve:
            raise HTTPException(400, detail=str(ve))
        except Exception as e:
            raise HTTPException(500, detail=f"Ошибка запроса: {str(e)}")

    @router.post(
        "/load_docs",
        response_model=List[PDFPage],
        response_model_exclude_none=True
    )
    async def load_pdf_documents(
            file: UploadFile = File(...),
            session_id: str = Form(...),
    ):
        # Validate file format
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, detail="Unsupported file format")

        # Save upload to a temporary file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            docs = load_pdfs(tmp_path)
            await load_chat_session_documents(session_id, docs)
            result = []
            for doc in docs:
                page_num : int = doc.metadata.get('page')
                page_content = doc.page_content
                result.append(PDFPage(page=page_num+1, content=page_content))
            return result
        except Exception as e:
            raise HTTPException(500, detail=f"Processing error: {str(e)}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return router

import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn

from app.routes import create_router
from GigaAM import gigaam
from core.ai_chat import cleanup_expired_sessions

# Загрузка переменных окружения и инициализация модели
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

MODEL_NAME = "v2_rnnt"
model = gigaam.load_model(MODEL_NAME)

# Lifespan-контекст
@asynccontextmanager
async def app_lifespan(app_: FastAPI):
    task = asyncio.create_task(session_cleaner_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("[session_cleaner] Остановлена фоновая задача")

# Создаём FastAPI-приложение
app = FastAPI(title="Legal Ally API", lifespan=app_lifespan)

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware ограничения размера файла
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    max_size = 100 * 1024 * 1024
    content_length = int(request.headers.get("content-length", 0))
    if content_length > max_size:
        return JSONResponse(status_code=413, content={"detail": "File too large. Max size 100MB"})
    return await call_next(request)

# Роуты
app.include_router(create_router(model))

# Задача очистки устаревших сессий
async def session_cleaner_loop():
    while True:
        await asyncio.sleep(600)
        print("[session_cleaner] Очистка устаревших сессий...")
        cleanup_expired_sessions()

# Точка входа
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

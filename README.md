## Установка

### Требования
- Python ≥ 3.8
- установленный и добавленный в PATH [ffmpeg](https://ffmpeg.org/)

### Запуск
1. Установить необходимые библиотеки
```bash
pip install -r requirements.txt
```
2. Запустить сервер
```bash
uvicorn app.main:app --reload
```

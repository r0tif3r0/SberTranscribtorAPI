from pydantic import BaseModel
from typing import Optional

class TranscriptSegment(BaseModel):
    speaker: Optional[str]
    start: str
    end: str
    text: str

    class Config:
        exclude_none = True

class PDFPage(BaseModel):
    page: int
    content: str
from pydantic import BaseModel
from typing import List, Optional

class HealthResponse(BaseModel):
    status: str
    version: str
    components: dict

class CaptionResponse(BaseModel):
    caption: str

class VQARequest(BaseModel):
    question: str

class VQAResponse(BaseModel):
    question: str
    answer: str

class RetrievalMatch(BaseModel):
    image_id: str
    caption: str
    score: float

class RetrievalResponse(BaseModel):
    matches: List[RetrievalMatch]

class AnalyzeResponse(BaseModel):
    caption: str
    vqa_answer: Optional[str] = None
    retrieval_matches: List[RetrievalMatch]
    heatmap_base64: str

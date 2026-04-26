from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
import io
import base64
import os
from ..schemas import AnalyzeResponse, RetrievalMatch
from .caption import get_captioner
from .vqa import get_vqa
from .retrieve import get_retriever

def get_gradcam():
    from ...models.gradcam import BasicGradCAM
    return BasicGradCAM()

router = APIRouter()

@router.post("/analyze", response_model=AnalyzeResponse)
async def full_analysis(
    file: UploadFile = File(...),
    question: str = Form(None),
    top_k: int = Form(3)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # 1. Caption
        captioner = get_captioner()
        caption = captioner.generate_caption(image)
        
        # 2. VQA (Optional)
        vqa_answer = None
        if question:
            vqa_model = get_vqa()
            vqa_answer = vqa_model.answer_question(image, question)
            
        # 3. Retrieve
        retrieval_matches = []
        retriever = get_retriever()
        if retriever.index is not None:
            matches = retriever.retrieve_similar(image, k=top_k)
            retrieval_matches = [RetrievalMatch(**m) for m in matches]
            
        # 4. Attention Heatmap
        cam = get_gradcam()
        heatmap_img = cam.generate_heatmap(image)
        
        # Convert heatmap to base64 for API transmission
        buffered = io.BytesIO()
        heatmap_img.save(buffered, format="PNG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return AnalyzeResponse(
            caption=caption,
            vqa_answer=vqa_answer,
            retrieval_matches=retrieval_matches,
            heatmap_base64=heatmap_base64
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
import io
import os
from ..schemas import VQAResponse

model_instance = None

def get_vqa():
    global model_instance
    if model_instance is None:
        from ...models.blip_vqa import MedicalVQA
        ckpt = os.environ.get("BLIP_VQA_MODEL_PATH", "Salesforce/blip-vqa-base")
        model_instance = MedicalVQA(model_name_or_path=ckpt)
    return model_instance

router = APIRouter()

@router.post("/vqa", response_model=VQAResponse)
async def answer_question(question: str = Form(...), file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        vqa_model = get_vqa()
        answer = vqa_model.answer_question(image, question)
        
        return VQAResponse(question=question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
import io
import os
from ..schemas import CaptionResponse

# Lazy loading of models for faster API startup during dev
model_instance = None

def get_captioner():
    global model_instance
    if model_instance is None:
        from ...models.blip_captioner import MedicalImageCaptioner
        ckpt = os.environ.get("BLIP_CAPTION_MODEL_PATH", "Salesforce/blip-image-captioning-base")
        model_instance = MedicalImageCaptioner(model_name_or_path=ckpt)
    return model_instance

router = APIRouter()

@router.post("/caption", response_model=CaptionResponse)
async def caption_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        captioner = get_captioner()
        caption = captioner.generate_caption(image)
        
        return CaptionResponse(caption=caption)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

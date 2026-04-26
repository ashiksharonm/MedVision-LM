from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
import io
import os
from ..schemas import RetrievalResponse, RetrievalMatch

model_instance = None

def get_retriever():
    global model_instance
    if model_instance is None:
        from ...models.clip_retriever import crossModalRetriever
        
        # Determine paths
        index_path = os.environ.get("FAISS_INDEX_PATH", "data/artifacts/clip_faiss.index")
        metadata_path = os.environ.get("FAISS_METADATA_PATH", "data/artifacts/retrieval_metadata.csv")
        ckpt = os.environ.get("CLIP_MODEL_PATH", "openai/clip-vit-base-patch32")
        
        # Only load if mock files or real files exist
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            model_instance = crossModalRetriever(model_name_or_path=ckpt, index_path=index_path, metadata_path=metadata_path)
        else:
            # Fallback to an uninitialized retriever for local testing without index
            print("Warning: FAISS index not found. Retrieval will fail until index is built.")
            model_instance = crossModalRetriever(model_name_or_path=ckpt)
            
    return model_instance

router = APIRouter()

@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_similar(top_k: int = Form(3), file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        retriever = get_retriever()
        if retriever.index is None:
            raise HTTPException(status_code=503, detail="FAISS index not loaded. Please build the index first.")
            
        matches = retriever.retrieve_similar(image, k=top_k)
        
        pydantic_matches = [RetrievalMatch(**match) for match in matches]
        return RetrievalResponse(matches=pydantic_matches)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

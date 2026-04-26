import os
from fastapi import FastAPI
from dotenv import load_dotenv

from .routes import caption, vqa, retrieve, analyze
from .schemas import HealthResponse

load_dotenv()

app = FastAPI(
    title="MedVision-LM API",
    description="Multimodal AI Assistant for Medical Image Analysis",
    version="1.0.0"
)

# Includes routes
app.include_router(caption.router, prefix="/api/v1", tags=["Captioning"])
app.include_router(vqa.router, prefix="/api/v1", tags=["VQA"])
app.include_router(retrieve.router, prefix="/api/v1", tags=["Retrieval"])
app.include_router(analyze.router, prefix="/api/v1", tags=["Full Analysis"])

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        components={
            "api": "healthy"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

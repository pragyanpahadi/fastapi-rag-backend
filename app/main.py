from fastapi import FastAPI
from app.api.routes import ingestion

app = FastAPI(title="RAG Engine API", version="1.0")

app.include_router(ingestion.router, prefix="/api/v1", tags=["Ingestion"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "infrastructure": "running"}

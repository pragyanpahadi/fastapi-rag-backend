from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import ingestion, chat
from app.core.database import engine, Base

Base.metadata.create_all(bind=engine)

app = FastAPI(title="RAG Engine API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingestion.router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "infrastructure": "running"}

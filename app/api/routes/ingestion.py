from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from app.services.document_service import (
    extract_text_from_file,
    chunk_by_tokens,
    chunk_by_recursive_characters
)
from app.services.embedding_service import generate_embeddings
from app.services.qdrant_service import insert_vectors, init_qdrant
from app.core.database import get_db
from app.models.document import DocumentMetadata

router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    strategy: str = Form("token"),
    db: Session = Depends(get_db)
):
    file_bytes = await file.read()

    raw_text = extract_text_from_file(file_bytes, file.filename)

    if not raw_text.strip():
        raise HTTPException(
            status_code=400, detail="The uploaded document contains no readable text.")

    if strategy == "token":
        chunks = chunk_by_tokens(raw_text)
    elif strategy == "recursive":
        chunks = chunk_by_recursive_characters(raw_text)
    else:
        raise HTTPException(
            status_code=400, detail="Invalid strategy. Choose 'token' or 'recursive'.")

    init_qdrant()

    db_meta = DocumentMetadata(
        filename=file.filename,
        strategy_used=strategy,
        chunk_count=len(chunks)
    )
    db.add(db_meta)
    db.commit()
    db.refresh(db_meta)

    vectors = generate_embeddings(chunks)

    qdrant_meta = {
        "document_id": db_meta.id,
        "filename": file.filename,
        "strategy": strategy
    }
    insert_vectors(chunks, vectors, qdrant_meta)

    return {
        "status": "success",
        "document_id": db_meta.id,
        "metadata": {
            "filename": file.filename,
            "strategy_used": strategy,
            "total_chunks_generated": len(chunks)
        },
        "preview": chunks[:2]
    }

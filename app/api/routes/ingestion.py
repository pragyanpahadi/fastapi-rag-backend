from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.document_service import (
    extract_text_from_file,
    chunk_by_tokens,
    chunk_by_recursive_characters
)

router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    strategy: str = Form("token")
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

    return {
        "status": "success",
        "metadata": {
            "filename": file.filename,
            "strategy_used": strategy,
            "total_chunks_generated": len(chunks)
        },
        "preview": chunks[:2]
    }

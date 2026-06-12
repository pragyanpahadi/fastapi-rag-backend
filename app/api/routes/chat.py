from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from app.services.redis_service import get_chat_history, add_message_to_history
from app.services.rag_service import generate_rag_response

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    booking_intent: Optional[Dict[str, Any]] = None


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    history = get_chat_history(request.session_id)

    rag_result = await generate_rag_response(request.message, history)

    add_message_to_history(request.session_id, "user", request.message)
    add_message_to_history(
        request.session_id, "assistant", rag_result["answer"])

    return ChatResponse(
        answer=rag_result["answer"],
        booking_intent=rag_result.get("booking_intent")
    )

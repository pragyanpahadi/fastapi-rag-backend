import json
import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Optional
from app.services.embedding_service import generate_embeddings
from app.services.qdrant_service import client, COLLECTION_NAME

api_key = os.getenv("GEMINI_API_KEY", "mock-key")

class BookingIntent(BaseModel):
    name: Optional[str] = Field(default=None, description="User's name")
    email: Optional[str] = Field(default=None, description="User's email")
    date: Optional[str] = Field(default=None, description="Preferred date")
    time: Optional[str] = Field(default=None, description="Preferred time")

class RAGResponse(BaseModel):
    answer: str = Field(description="Your conversational response")
    booking_intent: Optional[BookingIntent] = Field(default=None, description="Extracted booking intent")


async def retrieve_context(query: str, top_k: int = 3) -> str:
    query_vector = generate_embeddings([query])[0]

    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )

    context_blocks = [hit.payload.get("text", "") for hit in search_result]
    return "\n\n".join(context_blocks)


async def generate_rag_response(query: str, history: list) -> dict:
    context = await retrieve_context(query)

    if api_key == "mock-key":
        lower_query = query.lower()
        if "book" in lower_query or "schedule" in lower_query or "interview" in lower_query:
            return {
                "answer": "I would be happy to help you book an interview. Could you please provide your name, email, and preferred date and time?",
                "booking_intent": {"name": "John Doe", "email": "john@example.com", "date": "2026-06-15", "time": "14:00"}
            }
        return {
            "answer": f"Simulated RAG Response based on context block preview: {context[:100]}...",
            "booking_intent": None
        }

    genai_client = genai.Client(api_key=api_key)
    
    system_prompt = f"""You are a helpful AI assistant for Palm Mind. 
Use the following retrieved context to answer the user's question.
Context:
{context}

If the user wants to book an interview, extract their Name, Email, Date, and Time."""

    messages = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        messages.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    
    messages.append(types.Content(role="user", parts=[types.Part.from_text(text=query)]))

    response = await genai_client.aio.models.generate_content(
        model="gemini-1.5-flash",
        contents=messages,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=RAGResponse,
            temperature=0.1
        )
    )

    try:
        return json.loads(response.text)
    except Exception:
        return {"answer": response.text, "booking_intent": None}

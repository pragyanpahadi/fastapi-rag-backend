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

    system_prompt = f"""You are a helpful AI assistant for Palm Mind. 
Use the following retrieved context to answer the user's question.
Context:
{context}

If the user wants to book an interview, extract their Name, Email, Date, and Time."""

    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    
    contents.append({"role": "user", "parts": [{"text": query}]})

    import requests
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}"
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
            "responseSchema": RAGResponse.model_json_schema()
        }
    }

    try:
        res = requests.post(url, json=payload)
        data = res.json()
        if "candidates" in data:
            text_response = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text_response)
        else:
            return {"answer": f"API Error: {data}", "booking_intent": None}
    except Exception as e:
        return {"answer": f"Exception: {str(e)}", "booking_intent": None}

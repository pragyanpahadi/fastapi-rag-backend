import json
import os
from openai import AsyncOpenAI
from app.services.embedding_service import generate_embeddings
from app.services.qdrant_service import client, COLLECTION_NAME

api_key = os.getenv("OPENAI_API_KEY", "mock-key")
llm_client = AsyncOpenAI(api_key=api_key)


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

If the user wants to book an interview, extract their Name, Email, Date, and Time.
Format your response strictly as a JSON object with two keys: 
"answer": "your conversational response",
"booking_intent": {{"name": "", "email": "", "date": "", "time": ""}} (or null if no booking is requested)."""

    messages = [{"role": "system", "content": system_prompt}]

    for msg in history:
        messages.append({
            "role": "user" if msg["role"] == "user" else "assistant",
            "content": msg["content"]
        })

    messages.append({"role": "user", "content": query})

    response = await llm_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        response_format={"type": "json_object"}
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except json.JSONDecodeError:
        return {"answer": response.choices[0].message.content, "booking_intent": None}

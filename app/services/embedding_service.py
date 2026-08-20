import os
from openai import OpenAI
from typing import List

# Use the same API key set in Render
api_key = os.getenv("OPENAI_API_KEY", "mock-key")
client = OpenAI(api_key=api_key)

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    if api_key == "mock-key":
        # Fallback for offline testing matching original 384 dimensions
        return [[0.1] * 384 for _ in texts]
        
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
        dimensions=384
    )
    return [data.embedding for data in response.data]

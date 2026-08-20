import os
from typing import List
from google import genai
from google.genai import types

api_key = os.getenv("GEMINI_API_KEY", "mock-key")

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    if api_key == "mock-key":
        return [[0.1] * 384 for _ in texts]
        
    client = genai.Client(api_key=api_key)
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=384)
    )
    return [emb.values for emb in response.embeddings]

import os
import requests
from typing import List

api_key = os.getenv("GEMINI_API_KEY", "mock-key")

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    if api_key == "mock-key":
        return [[0.1] * 384 for _ in texts]
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
    vectors = []
    
    for text in texts:
        payload = {
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": 384
        }
        response = requests.post(url, json=payload)
        data = response.json()
        
        if "embedding" in data:
            vectors.append(data["embedding"]["values"])
        else:
            raise Exception(f"Gemini API Error on single chunk: {data}")
            
    return vectors

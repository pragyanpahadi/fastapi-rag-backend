import os
import requests
from typing import List

api_key = os.getenv("GEMINI_API_KEY", "mock-key")

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    if api_key == "mock-key":
        return [[0.1] * 384 for _ in texts]
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={api_key}"
    
    requests_body = []
    for text in texts:
        requests_body.append({
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": 384
        })
        
    response = requests.post(url, json={"requests": requests_body})
    data = response.json()
    
    if "embeddings" in data:
        return [emb["values"] for emb in data["embeddings"]]
    else:
        raise Exception(f"Gemini API Error: {data}")

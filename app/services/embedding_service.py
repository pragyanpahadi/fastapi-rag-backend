from sentence_transformers import SentenceTransformer
from typing import List

#model = SentenceTransformer("all-MiniLM-L6-v2")

model = None 

def generate_embeddings(text):
    global model
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    return model.encode(text)

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()

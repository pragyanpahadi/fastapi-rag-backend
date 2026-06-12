from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from typing import List

client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "documents"


def init_qdrant():
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )


def insert_vectors(texts: List[str], vectors: List[List[float]], metadata: dict):
    points = []
    for text, vector in zip(texts, vectors):
        point_id = str(uuid.uuid4())
        payload = {
            "text": text,
            **metadata
        }
        points.append(
            PointStruct(id=point_id, vector=vector, payload=payload)
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

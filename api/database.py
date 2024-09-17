import os

from qdrant_client import QdrantClient, models
from qdrant_client.conversions.common_types import Distance, VectorParams


qdrant_url = os.environ["QDRANT_URL"]
qdrant_client = QdrantClient(url=f"http://{qdrant_url}:6333")


def create_collection(name, vector_size):
    qdrant_client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )


def clear_collection(name, vector_size):
    qdrant_client.delete_collection(collection_name=name)
    qdrant_client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )


def search(vector, limit):
    return qdrant_client.search(
        collection_name="embeddings", query_vector=vector, limit=limit
    )


def insert(points):
    qdrant_client.upsert(collection_name="embeddings", points=points)

from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel


class EmbedRequest(BaseModel):
    text: List[str]


client = QdrantClient(url="http://localhost:6333")

app = FastAPI(swagger_ui_parameters={"tryItOutEnabled": True})
model = SentenceTransformer('all-MiniLM-L6-v2')


@app.get("/vector/setup")
def setup_db():
    client.create_collection(
        collection_name="embeddings",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    return JSONResponse({"success": True})


def search():
    search_result = client.query_points(
        collection_name="embeddings", query=[0.2, 0.1, 0.9, 0.7], limit=3
    ).points
    return search_result


@app.get("/vector/search")
async def vector_search():
    return JSONResponse(content=jsonable_encoder(search()))


@app.post("/vector/embed")
async def vector_embed(embed: EmbedRequest):
    vector = model.encode(embed.text)
    print(vector)
    vector_list = vector.tolist()  # Convert NumPy array to list
    return vector_list  # FastAPI can handle serialization

#
# if __name__ == '__main__':
#     search()

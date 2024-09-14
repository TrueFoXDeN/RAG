import os
import uuid
from enum import Enum
from itertools import islice
from typing import List

import boto3
import nltk
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from starlette.responses import JSONResponse


class EmbedRequest(BaseModel):
    text: List[str]


from nltk.tokenize import sent_tokenize

client = QdrantClient(url="http://localhost:6333")

app = FastAPI(swagger_ui_parameters={"tryItOutEnabled": True})
model = SentenceTransformer('all-MiniLM-L6-v2')

nltk.download('punkt', download_dir='./.venv')
nltk.download('punkt_tab', download_dir='./.venv')

s3_client = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',  # Replace with your MinIO endpoint
    aws_access_key_id='JmdUeDDSoACtwAbJgjhj',  # Replace with your access key
    aws_secret_access_key='0seVgUd0S6myUpd55jPMJO3Bhx8tQrrposYTgUor',  # Replace with your secret key
    region_name='us-east-1',
)


def list_text_files(bucket_name: str, prefix: str) -> List[str]:
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    text_files = []
    for page in page_iterator:
        print(page)
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('.txt'):
                    text_files.append(key)
    return text_files


def read_text_file(bucket_name: str, key: str) -> str:
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    content = response['Body'].read().decode('utf-8')
    return content


def chunks(iterable, size):
    iterator = iter(iterable)
    for first in iterator:
        yield [first] + list(islice(iterator, size - 1))


class LanguageEnum(str, Enum):
    english = 'english'
    german = 'german'


@app.post("/vector/ingest")
async def ingest_vectors(language: LanguageEnum):
    bucket_name = 'rag'  # Replace with your bucket name
    prefix = ''
    text_files = list_text_files(bucket_name, prefix)
    print(text_files)
    points = []
    for key in text_files:
        content = read_text_file(bucket_name, key)
        sentences = sent_tokenize(content, language=language.value)
        embeddings = model.encode(sentences)
        for sentence, vector in zip(sentences, embeddings):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    'text': sentence,
                    'source_file': key
                }
            )
            points.append(point)
    # Insert points in batches
    batch_size = 1000
    for batch in chunks(points, batch_size):
        client.upsert(
            collection_name='embeddings',
            points=batch
        )
    return JSONResponse({"success": True, "ingested": len(points)})


@app.post("/vector/setup")
def setup_db():
    client.create_collection(
        collection_name="embeddings",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    return JSONResponse({"success": True})


@app.post("/vector/search")
async def vector_search(query: str):
    query_embedding = model.encode([query])[0].tolist()
    search_result = client.search(
        collection_name="embeddings",
        query_vector=query_embedding,
        limit=3
    )
    return JSONResponse(content=jsonable_encoder(search_result))


@app.post("/vector/embed")
async def vector_embed(embed: EmbedRequest):
    vector = model.encode(embed.text)
    print(vector)
    vector_list = vector.tolist()  # Convert NumPy array to list
    return vector_list  # FastAPI can handle serialization

import os
import shutil
import uuid
import xml.etree.ElementTree as ET
from enum import Enum
from itertools import islice
from typing import List

import boto3
import nltk
from fastapi import FastAPI
from grobid_client.grobid_client import GrobidClient
from nltk.tokenize import sent_tokenize
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from starlette.responses import JSONResponse
from bs4 import BeautifulSoup


class EmbedRequest(BaseModel):
    text: List[str]


openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

qdrant_client = QdrantClient(url="http://localhost:6333")

grobid_client = GrobidClient(config_path="./grobid_config.json")

app = FastAPI(swagger_ui_parameters={"tryItOutEnabled": True})
model = SentenceTransformer("all-MiniLM-L6-v2")

nltk.download("punkt", download_dir="./.venv")
nltk.download("punkt_tab", download_dir="./.venv")

s3_client = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="JmdUeDDSoACtwAbJgjhj",
    aws_secret_access_key="0seVgUd0S6myUpd55jPMJO3Bhx8tQrrposYTgUor",
    region_name="us-east-1",
)


def generate_answer_with_gpt(query: str, context: str):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Du bist ein intelligenter Assistent.
                            Basierend auf dem untenstehenden Kontext, erstelle
                            eine kohärente und informative Antwort 
                           auf die Frage des Nutzers.""",
            },
            {"role": "assistant", "content": f"Kontext: {context}"},
            {"role": "user", "content": query},
        ],
    )

    print(response)
    return response.choices[0].message


def list_files(bucket_name: str, prefix: str) -> List[str]:
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    files = []
    for page in page_iterator:
        print(page)
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                files.append(key)
    return files


def read_text_file(bucket_name: str, key: str) -> str:
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    content = response["Body"].read().decode("utf-8")
    return content


def save_pdf_file(bucket_name: str, key: str, destination_folder: str) -> None:
    # Get the file name from the S3 key
    file_name = os.path.basename(key)
    file_path = os.path.join(destination_folder, file_name)

    # Download the file from S3 and save it locally
    response = s3_client.get_object(Bucket=bucket_name, Key=key)

    # Read the content of the file
    pdf_content = response["Body"].read()

    # Save the content to the local file
    with open(file_path, "wb") as pdf_file:
        pdf_file.write(pdf_content)

    # print(f"Saved {file_name} to {file_path}")


def chunks(iterable, size):
    iterator = iter(iterable)
    for first in iterator:
        yield [first] + list(islice(iterator, size - 1))


class LanguageEnum(str, Enum):
    english = "english"
    german = "german"


@app.post("/vector/ingest")
async def ingest_vectors(language: LanguageEnum):
    bucket_name = "rag"  # Replace with your bucket name
    prefix = ""
    text_files = list_files(bucket_name, prefix)
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
                payload={"text": sentence, "source_file": key},
            )
            points.append(point)
    # Insert points in batches
    batch_size = 1000
    for batch in chunks(points, batch_size):
        qdrant_client.upsert(collection_name="embeddings", points=batch)
    return JSONResponse({"success": True, "ingested": len(points)})


def extract_text_from_xml(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Namespace used in the XML file
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

    # List to store the extracted text
    extracted_text = []

    # Find all div and figure elements
    div_elements = root.findall('.//tei:div', ns)
    figure_elements = root.findall('.//tei:figure', ns)

    # Helper function to remove HTML tags from text
    def remove_html_tags(text):
        soup = BeautifulSoup(text, "html.parser")

        # Remove specific tags like <code>, <blockquote>, and others
        for tag in soup.find_all(['code', 'blockquote', 'div', 'figure']):
            tag.decompose()  # Remove the tag and its contents

        # Return cleaned text without the specific tags
        return soup.get_text()

    # Extract text from div elements
    for div in div_elements:
        # Convert element to string and remove HTML tags
        div_text = remove_html_tags(ET.tostring(div, encoding='unicode', method='xml'))
        extracted_text.append(div_text)

    # Extract text from figure elements (if needed)
    for figure in figure_elements:
        # Convert element to string and remove HTML tags
        figure_text = remove_html_tags(ET.tostring(figure, encoding='unicode', method='xml'))
        extracted_text.append(figure_text)

    # Return all extracted and cleaned text
    return extracted_text


@app.post("/vector/ingest/pdf")
async def ingest_pdf():
    bucket_name = "rag"  # Replace with your bucket name
    prefix = ""
    files = list_files(bucket_name, prefix)
    print(files)

    input_folder_path = "temp"
    if os.path.exists(input_folder_path):
        shutil.rmtree(input_folder_path)
    os.makedirs(input_folder_path)

    output_folder_path = "output_xml"
    if os.path.exists(output_folder_path):
        shutil.rmtree(output_folder_path)
    os.makedirs(output_folder_path)

    output_txt_folder_path = "output_txt"
    if os.path.exists(output_txt_folder_path):
        shutil.rmtree(output_txt_folder_path)
    os.makedirs(output_txt_folder_path)

    for key in files:
        save_pdf_file(bucket_name, key, input_folder_path)

    grobid_client.process(
        "processFulltextDocument",
        input_folder_path,
        output=output_folder_path,
        consolidate_citations=True,
        tei_coordinates=True,
        force=True,
    )

    shutil.rmtree(input_folder_path)

    for file in os.listdir(output_folder_path):
        filename = os.fsdecode(file)
        res = extract_text_from_xml(f'{output_folder_path}/{filename}')
        text = " ".join(result for result in res)
        filename = filename.replace(".grobid.tei.xml", "")
        with open(f'{output_txt_folder_path}/{filename}.txt', 'w', encoding='utf-8') as f:
            f.write(text)

    return JSONResponse({"success": True, "ingested": len(files)})


@app.post("/vector/setup")
def setup_db():
    qdrant_client.create_collection(
        collection_name="embeddings",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    return JSONResponse({"success": True})


@app.post("/vector/search")
async def vector_search(query: str):
    query_embedding = model.encode([query])[0].tolist()
    search_result = qdrant_client.search(
        collection_name="embeddings", query_vector=query_embedding, limit=3
    )

    context = [
        {"file": result.payload["source_file"], "text": result.payload["text"]}
        for result in search_result
    ]

    generator_context = " ".join([result.payload["text"] for result in search_result])

    # Generiere die Antwort basierend auf dem Kontext und der Anfrage
    answer = generate_answer_with_gpt(query, generator_context).content

    # return {"query": query, "context": context, "answer": answer.content}
    return {"query": query, "context": context, "answer": answer}
    # return JSONResponse(content=jsonable_encoder(search_result))


@app.post("/vector/embed")
async def vector_embed(embed: EmbedRequest):
    vector = model.encode(embed.text)
    print(vector)
    vector_list = vector.tolist()  # Convert NumPy array to list
    return vector_list  # FastAPI can handle serialization

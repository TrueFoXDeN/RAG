import json
import logging
import uuid
from typing import Generator

from qdrant_client.conversions.common_types import PointStruct
from starlette.responses import JSONResponse

from api import (
    chunking,
    database_chats,
    database_vector,
    embedding,
    extract,
    generate,
    retrieve,
)
from api.schemas import ChatCollectionResponse, ChatRequest, MessageRequest


def query_service(query) -> Generator[str, None, None]:
    query_embedding = embedding.embed(query)
    search_result = database_vector.search(query_embedding, 3)

    context = [
        {
            "file": result.payload["file"],
            "text": result.payload["text"],
            "page": result.payload["page"],
        }
        for result in search_result
    ]

    logging.debug(json.dumps(context, ensure_ascii=False))
    generator_context = " ".join([result.payload["text"] for result in search_result])

    eol = "\n\n"

    yield f"data: {json.dumps({'context': context}, ensure_ascii=False)}{eol}"
    yield f"data: [CONTEXT-END]{eol}"

    for chunk in generate.gpt(query, generator_context):
        yield f"data: {chunk.replace('\n', '[NEWLINE]')}{eol}"

    logging.debug("done")
    yield f"data: [DONE]{eol}"
    yield f": stream closed{eol}"


def context_service(query, limit):
    query_embedding = embedding.embed(query)
    search_result = database_vector.search(query_embedding, limit)

    context = [
        {
            "file": result.payload["file"],
            "text": result.payload["text"],
            "page": result.payload["page"],
        }
        for result in search_result
    ]

    return context


def ingest_service():
    files = retrieve.download_bucket("rag")
    for file in files:
        print(f"chunking file: {file}")
        pages = extract.extract_from_pdf(file)
        chunks = chunking.split_chunks(pages)

        points = []
        page_count = 1
        for page in chunks:
            for chunk in page:
                vector = embedding.embed(chunk)
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={"text": chunk, "file": file, "page": page_count},
                    )
                )
            page_count += 1
        database_vector.insert(points)
    return JSONResponse({"success": True})


def db_setup_service():
    database_vector.create_collection("embeddings", 384)
    return JSONResponse({"success": True})


def db_clear_service():
    database_vector.clear_collection("embeddings", 384)
    return JSONResponse({"success": True})


def db_get_all_chats_service():
    res = database_chats.get_all_chats()
    return ChatCollectionResponse.model_validate({"chats": res})


def db_get_chat_service(chat_id):
    res = database_chats.get_chat(chat_id)
    return ChatRequest.model_validate(res)


def db_save_chat_service(messages: ChatRequest):
    database_chats.save_chat(messages)
    return JSONResponse({"success": True})


def db_save_message_service(message: MessageRequest):
    database_chats.save_message(message)
    return JSONResponse({"success": True})

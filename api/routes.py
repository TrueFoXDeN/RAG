from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from starlette.responses import JSONResponse

from api.schemas import ChatRequest
from api.services import (
    context_service,
    db_clear_service,
    db_save_chat_service,
    db_setup_service,
    ingest_service,
    query_service,
)

router = APIRouter()


@router.get("/", tags=["Management"])
async def root():
    return JSONResponse({"status": "running"})


@router.get("/health", tags=["Management"])
async def health_route():
    return JSONResponse({"status": "healthy"})


@router.post("/ingest", tags=["Rag"])
async def ingest_route():
    return ingest_service()


@router.get("/query", tags=["Rag"])
async def query_route(query: str):
    return StreamingResponse(query_service(query), media_type="text/event-stream")


@router.get("/context", tags=["Rag"])
async def context_route(query: str, limit: int = 3):
    return context_service(query, limit)


@router.post("/db/setup", tags=["Database"])
def db_setup_route():
    return db_setup_service()


@router.delete("/db/clear", tags=["Database"])
def db_clear_route():
    return db_clear_service()


@router.post("/db/chat", tags=["Database"])
def db_save_chat_route(messages: ChatRequest):
    return db_save_chat_service(messages)

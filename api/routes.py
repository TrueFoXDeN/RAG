from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from starlette.responses import JSONResponse

from api.schemas import ChatCollectionResponse, ChatRequest, MessageRequest
from api.services import (
    context_service,
    db_clear_service,
    db_get_all_chats_service,
    db_get_chat_service,
    db_save_chat_service,
    db_save_message_service,
    db_setup_service,
    ingest_service,
    query_service,
)

router = APIRouter()


@router.get("/", tags=["Management"], operation_id="root")
async def root():
    return JSONResponse({"status": "running"})


@router.get("/health", tags=["Management"], operation_id="health")
async def health_route():
    return JSONResponse({"status": "healthy"})


@router.post("/ingest", tags=["Rag"], operation_id="ingest")
async def ingest_route():
    return ingest_service()


@router.get("/query", tags=["Rag"], operation_id="query")
async def query_route(query: str):
    return StreamingResponse(query_service(query), media_type="text/event-stream")


@router.get("/context", tags=["Rag"], operation_id="context")
async def context_route(query: str, limit: int = 3):
    return context_service(query, limit)


@router.post("/db/setup", tags=["Database"], operation_id="setup")
def db_setup_route():
    return db_setup_service()


@router.delete("/db/clear", tags=["Database"], operation_id="clear")
def db_clear_route():
    return db_clear_service()


@router.get("/db/chat", tags=["Database"], operation_id="get_chat")
def db_get_chat_route() -> ChatCollectionResponse:
    return db_get_all_chats_service()


@router.get("/db/chat/{id}", tags=["Database"], operation_id="get_chat_by_id")
def db_get_chat_by_id_route(id: str) -> ChatRequest:
    return db_get_chat_service(id)


@router.post("/db/chat", tags=["Database"], operation_id="save_chat")
def db_save_chat_route(messages: ChatRequest):
    return db_save_chat_service(messages)


@router.post("/db/message", tags=["Database"], operation_id="save_message")
def db_save_messafe_route(message: MessageRequest):
    return db_save_message_service(message)

from fastapi import APIRouter

from api.services import (
    db_clear_service,
    db_setup_service,
    ingest_service,
    query_service,
)

router = APIRouter()


@router.post("/ingest")
async def ingest_route():
    return ingest_service()


@router.post("/query")
async def query_route(query: str):
    return query_service(query)


@router.post("/db/setup")
def db_setup_route():
    return db_setup_service()


@router.delete("/db/clear")
def db_clear_route():
    return db_clear_service()

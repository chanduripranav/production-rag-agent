from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.schemas import HealthResponse, QueryRequest, QueryResponse
from app.api.service import QueryService, get_query_service


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Simple health check endpoint.
    """
    return HealthResponse(status="ok")


@router.post("/query", response_model=QueryResponse)
def query_docs(
    payload: QueryRequest,
    service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    """
    Ask a question against the indexed documents.

    Keep this handler thin: the actual pipeline lives in `service.py`.
    """
    return service.run(payload.question)


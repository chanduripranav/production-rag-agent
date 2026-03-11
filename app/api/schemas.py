from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Health status")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")


class CitationOut(BaseModel):
    chunk_id: str
    source_file: str
    page_number: int
    text_preview: str


class ChunkOut(BaseModel):
    chunk_id: str
    source_file: str
    page_number: int
    text_preview: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citation_validation_passed: bool
    citations: List[CitationOut]
    top_chunks: List[ChunkOut]


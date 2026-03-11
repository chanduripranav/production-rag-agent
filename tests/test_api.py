from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.schemas import QueryResponse
from app.api.service import get_query_service


class FakeQueryService:
    def run(self, question: str) -> QueryResponse:  # type: ignore[override]
        # Return a tiny deterministic response for tests.
        return QueryResponse(
            question=question,
            answer="Mock answer [c1]",
            citation_validation_passed=True,
            citations=[
                {
                    "chunk_id": "c1",
                    "source_file": "a.pdf",
                    "page_number": 1,
                    "text_preview": "preview",
                }
            ],
            top_chunks=[
                {
                    "chunk_id": "c1",
                    "source_file": "a.pdf",
                    "page_number": 1,
                    "text_preview": "preview",
                }
            ],
        )


def test_health() -> None:
    app = create_app()
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_query_with_mocked_service() -> None:
    app = create_app()

    # Override the dependency so we don't run the real pipeline in tests.
    app.dependency_overrides[get_query_service] = lambda: FakeQueryService()

    client = TestClient(app)
    resp = client.post("/query", json={"question": "hello"})
    assert resp.status_code == 200

    data = resp.json()
    assert data["question"] == "hello"
    assert "answer" in data
    assert data["citation_validation_passed"] is True
    assert isinstance(data["citations"], list) and len(data["citations"]) == 1
    assert isinstance(data["top_chunks"], list) and len(data["top_chunks"]) == 1


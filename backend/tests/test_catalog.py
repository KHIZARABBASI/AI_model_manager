import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.catalog import normalize_model


def test_normalize_model_handles_missing_metadata() -> None:
    model = normalize_model({"id": "example/community-model"})

    assert model.id == "example/community-model"
    assert model.name == "community-model"
    assert model.publisher == "example"
    assert model.license is None
    assert model.task is None
    assert model.gated is False
    assert model.source_url == "https://huggingface.co/example/community-model"


def test_normalize_model_sums_repository_sizes() -> None:
    model = normalize_model(
        {
            "id": "example/model",
            "author": "example",
            "downloads": 123,
            "likes": 45,
            "pipeline_tag": "text-generation",
            "gated": True,
            "cardData": {"license": "apache-2.0"},
            "siblings": [{"size": 100}, {"size": 250}],
            "tags": ["text-generation", "gguf"],
        }
    )

    assert model.downloads == 123
    assert model.likes == 45
    assert model.task == "text-generation"
    assert model.license == "apache-2.0"
    assert model.gated is True
    assert model.estimated_repository_size_bytes == 350


def test_model_search_reports_catalog_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable_search(query: str, limit: int) -> list:
        raise RuntimeError("The Hugging Face catalog timed out.")

    monkeypatch.setattr("app.main.search_models", unavailable_search)

    response = TestClient(app).get("/models/search?query=phi&limit=3")

    assert response.status_code == 502
    assert "timed out" in response.json()["detail"]

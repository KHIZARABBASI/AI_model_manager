import os
from typing import Optional

import httpx
from pydantic import BaseModel, Field


HUGGING_FACE_API_URL = "https://huggingface.co/api/models"


class CatalogModel(BaseModel):
    id: str
    name: str
    publisher: Optional[str] = None
    task: Optional[str] = None
    license: Optional[str] = None
    downloads: int = 0
    likes: int = 0
    gated: bool = False
    tags: list[str] = Field(default_factory=list)
    source_url: str
    estimated_repository_size_bytes: Optional[int] = None


class ModelSearchResponse(BaseModel):
    models: list[CatalogModel]
    query: str
    limit: int


def _license_from_card_data(card_data: object) -> Optional[str]:
    if not isinstance(card_data, dict):
        return None

    license_value = card_data.get("license")
    if isinstance(license_value, str):
        return license_value
    if isinstance(license_value, list):
        licenses = [item for item in license_value if isinstance(item, str)]
        return ", ".join(licenses) if licenses else None
    return None


def normalize_model(raw_model: dict) -> CatalogModel:
    model_id = str(raw_model.get("id") or "").strip()
    publisher = raw_model.get("author")
    if not isinstance(publisher, str) or not publisher:
        publisher = model_id.split("/", 1)[0] if "/" in model_id else None

    siblings = raw_model.get("siblings")
    repository_size: Optional[int] = None
    if isinstance(siblings, list):
        sizes = [
            sibling.get("size")
            for sibling in siblings
            if isinstance(sibling, dict)
            and isinstance(sibling.get("size"), int)
            and sibling["size"] >= 0
        ]
        if sizes:
            repository_size = sum(sizes)

    tags = raw_model.get("tags")
    normalized_tags = (
        [tag for tag in tags if isinstance(tag, str)]
        if isinstance(tags, list)
        else []
    )
    gated_value = raw_model.get("gated")
    gated = gated_value is True or (
        isinstance(gated_value, str)
        and gated_value.lower() in {"true", "auto", "manual"}
    )

    task = raw_model.get("pipeline_tag")
    if not isinstance(task, str):
        task = None

    downloads = raw_model.get("downloads")
    likes = raw_model.get("likes")

    return CatalogModel(
        id=model_id,
        name=model_id.split("/", 1)[-1] if model_id else "Unknown model",
        publisher=publisher,
        task=task,
        license=_license_from_card_data(raw_model.get("cardData")),
        downloads=int(downloads) if isinstance(downloads, int) else 0,
        likes=int(likes) if isinstance(likes, int) else 0,
        gated=gated,
        tags=normalized_tags,
        source_url=f"https://huggingface.co/{model_id}" if model_id else "https://huggingface.co/models",
        estimated_repository_size_bytes=repository_size,
    )


def search_models(query: str, limit: int = 20) -> list[CatalogModel]:
    headers = {"Accept": "application/json"}
    token = os.getenv("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    params = {
        "search": query,
        "filter": "text-generation",
        "sort": "downloads",
        "direction": "-1",
        "limit": str(limit),
        "full": "true",
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                HUGGING_FACE_API_URL,
                params=params,
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.TimeoutException as error:
        raise RuntimeError("The Hugging Face catalog timed out.") from error
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        if status in (401, 403):
            raise RuntimeError(
                "Hugging Face rejected the catalog request. Authentication may be required."
            ) from error
        raise RuntimeError(
            f"Hugging Face catalog returned HTTP {status}."
        ) from error
    except (httpx.HTTPError, ValueError) as error:
        raise RuntimeError("Could not retrieve the Hugging Face catalog.") from error

    if not isinstance(payload, list):
        raise RuntimeError("Hugging Face returned an unexpected catalog response.")

    return [
        normalize_model(item)
        for item in payload
        if isinstance(item, dict)
    ]

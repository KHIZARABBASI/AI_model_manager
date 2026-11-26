import asyncio
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app import downloads
from app.main import app


MODEL_ID = "test-owner/demo-model"
MODEL_FILE = "model.gguf"
FILE_CONTENT = b"safe-model"


class SlowStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"partial"
        await asyncio.sleep(30)

    async def aclose(self) -> None:
        return None


class BytesStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self.content = content

    async def __aiter__(self):
        yield self.content

    async def aclose(self) -> None:
        return None


def install_transport(
    monkeypatch: pytest.MonkeyPatch,
    *,
    slow: bool = False,
    supports_range: bool = False,
    requested_ranges: list[str] | None = None,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/tree/main"):
            return httpx.Response(
                200,
                json=[
                    {
                        "type": "file",
                        "path": MODEL_FILE,
                        "size": len(FILE_CONTENT),
                    },
                    {"type": "file", "path": "modeling_custom.py", "size": 20},
                    {"type": "file", "path": "../escape.gguf", "size": 10},
                ],
            )
        if request.url.path.endswith("/resolve/main/model.gguf"):
            if slow:
                return httpx.Response(200, stream=SlowStream())
            range_header = request.headers.get("range")
            if range_header and supports_range:
                if requested_ranges is not None:
                    requested_ranges.append(range_header)
                start = int(range_header.removeprefix("bytes=").removesuffix("-"))
                return httpx.Response(
                    206,
                    stream=BytesStream(FILE_CONTENT[start:]),
                    headers={
                        "content-range": f"bytes {start}-{len(FILE_CONTENT) - 1}/{len(FILE_CONTENT)}"
                    },
                )
            return httpx.Response(200, stream=BytesStream(FILE_CONTENT))
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    def client_factory() -> httpx.AsyncClient:
        return original_client(transport=transport, follow_redirects=True)

    monkeypatch.setattr(downloads, "_create_http_client", client_factory)


def test_model_and_file_paths_reject_traversal_and_windows_special_names() -> None:
    for model_id in ("../outside/model", "owner/..", "CON/model"):
        with pytest.raises(ValueError):
            downloads.validate_model_id(model_id)

    for file_path in ("../outside.gguf", "folder\\model.gguf", "CON/data.gguf"):
        with pytest.raises(ValueError):
            downloads.validate_file_path(file_path)


def test_supported_file_filter_excludes_repository_scripts() -> None:
    assert downloads.is_supported_model_file("weights/model.safetensors")
    assert downloads.is_supported_model_file("tokenizer.json")
    assert not downloads.is_supported_model_file("modeling_custom.py")
    assert not downloads.is_supported_model_file("README.md")


def test_model_file_listing_excludes_scripts_and_unsafe_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_transport(monkeypatch)

    response = TestClient(app).get(
        "/models/files",
        params={"model_id": MODEL_ID},
    )

    assert response.status_code == 200
    assert response.json()["files"] == [
        {"path": MODEL_FILE, "size_bytes": len(FILE_CONTENT)}
    ]


def test_model_file_listing_follows_hugging_face_cursor_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("cursor") == "next-page":
            return httpx.Response(
                200,
                json=[{"type": "file", "path": "tokenizer.json", "size": 20}],
            )
        return httpx.Response(
            200,
            json=[{"type": "file", "path": MODEL_FILE, "size": 10}],
            headers={
                "link": (
                    '<https://huggingface.co/api/models/test-owner/demo-model/'
                    'tree/main?cursor=next-page>; rel="next"'
                )
            },
        )

    original_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        downloads,
        "_create_http_client",
        lambda: original_client(transport=transport, follow_redirects=True),
    )

    response = TestClient(app).get("/models/files", params={"model_id": MODEL_ID})

    assert response.status_code == 200
    assert [item["path"] for item in response.json()["files"]] == [
        MODEL_FILE,
        "tokenizer.json",
    ]


def test_download_completes_and_delete_requires_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MODEL_STORAGE_DIR", str(tmp_path))
    install_transport(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/downloads",
            json={"model_id": MODEL_ID, "files": [MODEL_FILE]},
        )
        assert response.status_code == 202
        job_id = response.json()["id"]

        status = response.json()
        for _ in range(100):
            status_response = client.get(f"/downloads/{job_id}")
            status = status_response.json()
            if status["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)

        assert status["status"] == "completed", status
        assert status["progress"] == 1
        installed = client.get("/downloads/installed")
        assert installed.status_code == 200
        assert installed.json()[0]["model_id"] == MODEL_ID

        model_path = tmp_path / "test-owner" / "demo-model" / MODEL_FILE
        assert model_path.read_bytes() == FILE_CONTENT

        delete_url = "/downloads/installed?model_id=test-owner%2Fdemo-model"
        assert client.delete(delete_url).status_code == 400
        assert model_path.exists()
        assert (
            client.post(
                "/downloads",
                json={"model_id": MODEL_ID, "files": [MODEL_FILE]},
            ).status_code
            == 409
        )
        assert client.delete(f"{delete_url}&confirm=true").status_code == 204
        assert not model_path.exists()


def test_cancelled_download_preserves_partial_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MODEL_STORAGE_DIR", str(tmp_path))
    install_transport(monkeypatch, slow=True)

    with TestClient(app) as client:
        response = client.post(
            "/downloads",
            json={"model_id": MODEL_ID, "files": [MODEL_FILE]},
        )
        job_id = response.json()["id"]

        status = response.json()
        for _ in range(100):
            status = client.get(f"/downloads/{job_id}").json()
            if status["status"] == "downloading" and status["downloaded_bytes"] > 0:
                break
            time.sleep(0.01)

        duplicate = client.post(
            "/downloads",
            json={"model_id": MODEL_ID, "files": [MODEL_FILE]},
        )
        assert duplicate.status_code == 409
        cancelled = client.delete(f"/downloads/{job_id}")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        time.sleep(0.05)

    partial_path = (
        tmp_path
        / "test-owner"
        / "demo-model"
        / f"{MODEL_FILE}.part"
    )
    assert partial_path.exists()
    assert partial_path.read_bytes() == b"partial"


def test_new_download_resumes_existing_partial_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MODEL_STORAGE_DIR", str(tmp_path))
    partial_path = (
        tmp_path
        / "test-owner"
        / "demo-model"
        / f"{MODEL_FILE}.part"
    )
    partial_path.parent.mkdir(parents=True)
    partial_path.write_bytes(FILE_CONTENT[:4])
    requested_ranges: list[str] = []
    install_transport(
        monkeypatch,
        supports_range=True,
        requested_ranges=requested_ranges,
    )

    with TestClient(app) as client:
        response = client.post(
            "/downloads",
            json={"model_id": MODEL_ID, "files": [MODEL_FILE]},
        )
        job_id = response.json()["id"]
        status = response.json()
        for _ in range(100):
            status = client.get(f"/downloads/{job_id}").json()
            if status["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)

    assert status["status"] == "completed", status
    assert requested_ranges == ["bytes=4-"]
    assert (partial_path.parent / MODEL_FILE).read_bytes() == FILE_CONTENT
    assert not partial_path.exists()

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app import ollama_runtime
from app.main import app


MODEL_ID = "test-owner/demo-model"
MODEL_FILE = "model.gguf"
MODEL_NAME = "aimm-test-model"


def install_ollama_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler,
) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    transport = httpx.MockTransport(recording_handler)
    monkeypatch.setattr(
        ollama_runtime,
        "_create_ollama_client",
        lambda: httpx.AsyncClient(
            base_url=ollama_runtime.OLLAMA_BASE_URL,
            transport=transport,
        ),
    )
    return requests


def test_status_lists_only_managed_models_and_matches_latest_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.5.0"})
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "aimm-test-model:latest", "size": 20},
                        {"name": "public-model:latest", "size": 30},
                    ]
                },
            )
        if request.url.path == "/api/ps":
            return httpx.Response(
                200,
                json={"models": [{"name": "aimm-test-model:latest"}]},
            )
        return httpx.Response(404)

    install_ollama_transport(monkeypatch, handler)
    monkeypatch.setattr(ollama_runtime.shutil, "which", lambda _: "ollama")

    status = asyncio.run(ollama_runtime.get_ollama_status())

    assert status.available
    assert status.cli_available
    assert status.version == "0.5.0"
    assert [model.name for model in status.models] == ["aimm-test-model:latest"]
    assert status.models[0].running


def test_status_reports_unavailable_service_without_claiming_runtime_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    install_ollama_transport(monkeypatch, handler)
    monkeypatch.setattr(ollama_runtime.shutil, "which", lambda _: None)

    status = asyncio.run(ollama_runtime.get_ollama_status())

    assert not status.available
    assert not status.cli_available
    assert "not running" in status.message.lower()
    assert status.models == []


def test_gguf_import_stages_link_and_keeps_managed_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MODEL_STORAGE_DIR", str(tmp_path))
    source = tmp_path / "test-owner" / "demo-model" / MODEL_FILE
    source.parent.mkdir(parents=True)
    source.write_bytes(b"gguf-data")
    monkeypatch.setattr(ollama_runtime, "_ollama_server_is_available", lambda: None)
    monkeypatch.setattr(ollama_runtime.shutil, "which", lambda _: "ollama")
    monkeypatch.setattr(ollama_runtime, "_installed_ollama_names", lambda: set())
    captured: dict[str, object] = {}

    def fake_run(command, *, cwd, capture_output, text, timeout, check):
        staging = Path(cwd)
        staged_model = staging / "model.gguf"
        captured["command"] = command
        captured["modelfile"] = (staging / "Modelfile").read_text(encoding="utf-8")
        captured["source_inode"] = source.stat().st_ino
        captured["staged_inode"] = staged_model.stat().st_ino
        captured["timeout"] = timeout
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ollama_runtime.subprocess, "run", fake_run)

    result = ollama_runtime.import_gguf_model(
        ollama_runtime.ImportModelRequest(model_id=MODEL_ID, file_path=MODEL_FILE)
    )

    assert result.model_name.startswith(ollama_runtime.OLLAMA_MODEL_PREFIX)
    assert captured["command"] == ["ollama", "create", result.model_name]
    assert captured["modelfile"] == "FROM ./model.gguf\n"
    assert captured["source_inode"] == captured["staged_inode"]
    assert captured["timeout"] == 3600
    assert source.read_bytes() == b"gguf-data"


def test_gguf_import_refuses_existing_default_latest_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MODEL_STORAGE_DIR", str(tmp_path))
    source = tmp_path / "test-owner" / "demo-model" / MODEL_FILE
    source.parent.mkdir(parents=True)
    source.write_bytes(b"gguf-data")
    request = ollama_runtime.ImportModelRequest(
        model_id=MODEL_ID,
        file_path=MODEL_FILE,
    )
    model_name = ollama_runtime._ollama_model_name(MODEL_ID, MODEL_FILE)
    monkeypatch.setattr(ollama_runtime, "_ollama_server_is_available", lambda: None)
    monkeypatch.setattr(ollama_runtime.shutil, "which", lambda _: "ollama")
    monkeypatch.setattr(
        ollama_runtime,
        "_installed_ollama_names",
        lambda: {f"{model_name}:latest"},
    )
    monkeypatch.setattr(
        ollama_runtime.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("Existing model must not be overwritten"),
    )

    with pytest.raises(FileExistsError, match="not overwritten"):
        ollama_runtime.import_gguf_model(request)


def test_split_gguf_import_uses_first_shard_as_modelfile_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MODEL_STORAGE_DIR", str(tmp_path))
    model_dir = tmp_path / "test-owner" / "demo-model"
    model_dir.mkdir(parents=True)
    for index in (1, 2):
        (model_dir / f"model-{index:05d}-of-00002.gguf").write_bytes(
            f"shard-{index}".encode()
        )
    monkeypatch.setattr(ollama_runtime, "_ollama_server_is_available", lambda: None)
    monkeypatch.setattr(ollama_runtime.shutil, "which", lambda _: "ollama")
    monkeypatch.setattr(ollama_runtime, "_installed_ollama_names", lambda: set())
    captured_modelfile: list[str] = []

    def fake_run(command, *, cwd, capture_output, text, timeout, check):
        staging = Path(cwd)
        captured_modelfile.append(
            (staging / "Modelfile").read_text(encoding="utf-8")
        )
        assert sorted(path.name for path in staging.glob("*.gguf")) == [
            "shard-00001-of-00002.gguf",
            "shard-00002-of-00002.gguf",
        ]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ollama_runtime.subprocess, "run", fake_run)

    result = ollama_runtime.import_gguf_model(
        ollama_runtime.ImportModelRequest(
            model_id=MODEL_ID,
            file_path="model-00001-of-00002.gguf",
        )
    )

    assert result.model_name.startswith(ollama_runtime.OLLAMA_MODEL_PREFIX)
    assert captured_modelfile == ["FROM ./shard-00001-of-00002.gguf\n"]


def test_local_model_actions_and_chat_use_only_loopback_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": f"{MODEL_NAME}:latest"}]},
            )
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "Hello locally"}},
            )
        return httpx.Response(200, json={})

    requests = install_ollama_transport(monkeypatch, handler)
    action = ollama_runtime.ModelActionRequest(model_name=MODEL_NAME)

    assert asyncio.run(ollama_runtime.load_ollama_model(action)).startswith("Model loaded")
    assert asyncio.run(ollama_runtime.stop_ollama_model(action)).startswith("Model stopped")
    chat = asyncio.run(
        ollama_runtime.chat_with_ollama(
            ollama_runtime.ChatRequest(
                model_name=MODEL_NAME,
                messages=[ollama_runtime.ChatMessage(role="user", content="Hello")],
            )
        )
    )
    assert chat.message == "Hello locally"

    with pytest.raises(ValueError, match="confirm=true"):
        asyncio.run(
            ollama_runtime.delete_ollama_model(
                ollama_runtime.DeleteModelRequest(model_name=MODEL_NAME)
            )
        )
    deleted = asyncio.run(
        ollama_runtime.delete_ollama_model(
            ollama_runtime.DeleteModelRequest(model_name=MODEL_NAME, confirm=True)
        )
    )
    assert "source files were kept" in deleted

    local_requests = [request for request in requests if request.url.path != "/api/tags"]
    assert {request.url.path for request in local_requests} == {
        "/api/generate",
        "/api/chat",
        "/api/delete",
    }
    assert all(request.url.host == "127.0.0.1" for request in requests)


def test_chat_surfaces_ollama_generation_error_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": f"{MODEL_NAME}:latest"}]},
            )
        return httpx.Response(
            500,
            json={
                "error": (
                    "llama-server process has terminated: "
                    "tensor 'blk.32.attn_norm.weight' not found"
                )
            },
        )

    install_ollama_transport(monkeypatch, handler)
    response = TestClient(app).post(
        "/runtime/ollama/chat",
        json={
            "model_name": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 502
    assert "blk.32.attn_norm.weight" in response.json()["detail"]

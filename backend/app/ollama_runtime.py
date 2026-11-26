import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Optional

import httpx
from pydantic import BaseModel, Field

from app.downloads import model_file_path, storage_root, validate_model_id, validate_file_path


OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL_PREFIX = "aimm-"
OLLAMA_MODEL_PATTERN = re.compile(r"^aimm-[a-z0-9-]{1,90}(?::[A-Za-z0-9._-]+)?$")
GGUF_SHARD_PATTERN = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)


class OllamaModel(BaseModel):
    name: str
    size_bytes: Optional[int] = None
    format: Optional[str] = None
    running: bool = False


class OllamaStatus(BaseModel):
    available: bool
    cli_available: bool
    version: Optional[str] = None
    models: list[OllamaModel] = Field(default_factory=list)
    message: str


class ImportModelRequest(BaseModel):
    model_id: str = Field(min_length=3, max_length=300)
    file_path: str = Field(min_length=1, max_length=1000)


class ImportModelResponse(BaseModel):
    model_name: str
    message: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    model_name: str = Field(min_length=5, max_length=120)
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)


class ChatResponse(BaseModel):
    model_name: str
    message: str


class ModelActionRequest(BaseModel):
    model_name: str = Field(min_length=5, max_length=120)


class DeleteModelRequest(ModelActionRequest):
    confirm: bool = False


def _create_ollama_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=OLLAMA_BASE_URL,
        timeout=httpx.Timeout(connect=3.0, read=None, write=30.0, pool=3.0),
    )


def _ollama_http_error(error: httpx.HTTPStatusError, operation: str) -> RuntimeError:
    try:
        payload = error.response.json()
    except ValueError:
        payload = {}
    detail = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(detail, str) or not detail.strip():
        detail = error.response.text.strip()
    detail = " ".join(detail.split())[:600] if detail else ""
    if not detail:
        detail = f"HTTP {error.response.status_code}"
    return RuntimeError(f"{operation}: {detail}")


def _is_managed_model_name(model_name: str) -> bool:
    return OLLAMA_MODEL_PATTERN.fullmatch(model_name) is not None


def _base_model_name(model_name: str) -> str:
    return model_name.split(":", 1)[0]


def _is_managed_model_entry(model: dict) -> bool:
    name = model.get("name") or model.get("model")
    return isinstance(name, str) and name.split(":", 1)[0].startswith(
        OLLAMA_MODEL_PREFIX
    )


async def get_ollama_status() -> OllamaStatus:
    cli_available = shutil.which("ollama") is not None
    try:
        async with _create_ollama_client() as client:
            version_response = await client.get("/api/version")
            version_response.raise_for_status()
            tags_response = await client.get("/api/tags")
            tags_response.raise_for_status()
            running_response = await client.get("/api/ps")
            running_response.raise_for_status()
            version_data = version_response.json()
            tags_data = tags_response.json()
            running_data = running_response.json()
    except httpx.ConnectError:
        return OllamaStatus(
            available=False,
            cli_available=cli_available,
            message=(
                "Ollama is not running. Install and start it yourself; "
                "AI Model Manager will not install software automatically."
            ),
        )
    except (httpx.HTTPError, ValueError):
        return OllamaStatus(
            available=False,
            cli_available=cli_available,
            message="The local service did not return a valid Ollama API response.",
        )

    raw_models = tags_data.get("models", []) if isinstance(tags_data, dict) else []
    running_data_models = (
        running_data.get("models", []) if isinstance(running_data, dict) else []
    )
    running_names = {
        _base_model_name(str(item.get("name") or item.get("model")))
        for item in running_data_models
        if isinstance(item, dict) and _is_managed_model_entry(item)
    }

    models = [
        OllamaModel(
            name=str(item.get("name") or item.get("model")),
            size_bytes=item.get("size") if isinstance(item.get("size"), int) else None,
            format=(
                item.get("details", {}).get("format")
                if isinstance(item.get("details"), dict)
                and isinstance(item.get("details", {}).get("format"), str)
                else None
            ),
            running=(
                _base_model_name(str(item.get("name") or item.get("model")))
                in running_names
            ),
        )
        for item in raw_models
        if isinstance(item, dict) and _is_managed_model_entry(item)
    ]
    version = version_data.get("version") if isinstance(version_data, dict) else None
    return OllamaStatus(
        available=True,
        cli_available=cli_available,
        version=version if isinstance(version, str) else None,
        models=models,
        message="Ollama is running locally. Chat requests go only to this device.",
    )


def _ollama_server_is_available() -> None:
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/version", timeout=3.0)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise RuntimeError(
            "Start Ollama locally before importing a model. The app does not install or start it."
        ) from error


def _installed_ollama_names() -> set[str]:
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10.0)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise RuntimeError("Could not check the models already installed in Ollama.") from error
    models = payload.get("models", []) if isinstance(payload, dict) else []
    return {
        str(model.get("name") or model.get("model"))
        for model in models
        if isinstance(model, dict)
    }


def _ollama_model_name(model_id: str, file_path: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", f"{model_id}-{Path(file_path).stem}".lower())
    base = base.strip("-")[:48].rstrip("-") or "local-model"
    suffix = hashlib.sha256(f"{model_id}\0{file_path}".encode()).hexdigest()[:10]
    return f"{OLLAMA_MODEL_PREFIX}{base}-{suffix}"


def _import_files(model_id: str, file_path: str) -> tuple[list[Path], bool]:
    validate_model_id(model_id)
    validate_file_path(file_path)
    if Path(file_path).suffix.lower() != ".gguf":
        raise ValueError("Ollama import currently supports GGUF files only.")

    source = model_file_path(model_id, file_path)
    if not source.is_file():
        raise FileNotFoundError("The selected GGUF file is not installed.")

    basename = Path(file_path).name
    shard_match = GGUF_SHARD_PATTERN.fullmatch(basename)
    if shard_match is None:
        return [source], False

    total = int(shard_match.group("total"))
    if total < 2 or total > 64:
        raise ValueError("The split GGUF shard count is invalid.")

    relative_parent = file_path.rsplit("/", 1)[0] if "/" in file_path else ""
    prefix = shard_match.group("prefix")
    shard_paths: list[Path] = []
    for index in range(1, total + 1):
        shard_name = f"{prefix}-{index:05d}-of-{total:05d}.gguf"
        relative = f"{relative_parent}/{shard_name}" if relative_parent else shard_name
        shard = model_file_path(model_id, relative)
        if not shard.is_file():
            raise ValueError(
                f"All {total} GGUF shards must be installed before importing this model."
            )
        shard_paths.append(shard)
    return shard_paths, True


def import_gguf_model(request: ImportModelRequest) -> ImportModelResponse:
    model_name = _ollama_model_name(request.model_id, request.file_path)
    sources, split = _import_files(request.model_id, request.file_path)
    _ollama_server_is_available()
    executable = shutil.which("ollama")
    if executable is None:
        raise RuntimeError(
            "The Ollama CLI was not found. Install Ollama yourself, then restart its local service."
        )
    if model_name in {
        _base_model_name(name) for name in _installed_ollama_names()
    }:
        raise FileExistsError(
            f"{model_name} already exists in Ollama. The existing model was not overwritten."
        )

    root = storage_root()
    import_root = root / ".runtime-imports"
    import_root.mkdir(parents=True, exist_ok=True)
    if import_root.is_symlink() or not import_root.resolve().is_relative_to(root):
        raise ValueError("Ollama import staging must stay inside managed model storage.")

    with tempfile.TemporaryDirectory(prefix="ollama-", dir=import_root) as temporary_dir:
        staging = Path(temporary_dir)
        if split:
            staged_names = [
                f"shard-{index:05d}-of-{len(sources):05d}.gguf"
                for index in range(1, len(sources) + 1)
            ]
            from_path = f"./shard-00001-of-{len(sources):05d}.gguf"
        else:
            staged_names = ["model.gguf"]
            from_path = "./model.gguf"

        for source, staged_name in zip(sources, staged_names, strict=True):
            os.link(source, staging / staged_name)
        (staging / "Modelfile").write_text(
            f"FROM {from_path}\n",
            encoding="utf-8",
        )

        try:
            result = subprocess.run(
                [executable, "create", model_name],
                cwd=staging,
                capture_output=True,
                text=True,
                timeout=3600,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "Ollama import timed out. Check the local Ollama service and try again."
            ) from error
        except OSError as error:
            raise RuntimeError("Could not start the installed Ollama CLI.") from error

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[-1000:]
            raise RuntimeError(
                f"Ollama could not import the GGUF model. {detail}".strip()
            )

    return ImportModelResponse(
        model_name=model_name,
        message="Model imported into Ollama. The managed GGUF source file was kept.",
    )


async def _ensure_managed_model_exists(model_name: str) -> None:
    if not _is_managed_model_name(model_name):
        raise ValueError("Choose an AI Model Manager-imported Ollama model.")
    try:
        async with _create_ollama_client() as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as error:
        raise RuntimeError("Ollama is not reachable on this device.") from error
    models = payload.get("models", []) if isinstance(payload, dict) else []
    if not any(
        isinstance(model, dict)
        and _is_managed_model_entry(model)
        and _base_model_name(str(model.get("name") or model.get("model")))
        == _base_model_name(model_name)
        for model in models
    ):
        raise FileNotFoundError("This model is no longer installed in Ollama.")


async def load_ollama_model(request: ModelActionRequest) -> str:
    await _ensure_managed_model_exists(request.model_name)
    try:
        async with _create_ollama_client() as client:
            response = await client.post(
                "/api/generate",
                json={
                    "model": request.model_name,
                    "keep_alive": "5m",
                    "stream": False,
                },
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise _ollama_http_error(error, "Ollama could not load this model") from error
    except httpx.HTTPError as error:
        raise RuntimeError("Ollama is not reachable on this device.") from error
    return "Model loaded locally and kept in memory for five minutes."


async def stop_ollama_model(request: ModelActionRequest) -> str:
    await _ensure_managed_model_exists(request.model_name)
    try:
        async with _create_ollama_client() as client:
            response = await client.post(
                "/api/generate",
                json={
                    "model": request.model_name,
                    "keep_alive": 0,
                    "stream": False,
                },
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise _ollama_http_error(error, "Ollama could not stop this model") from error
    except httpx.HTTPError as error:
        raise RuntimeError("Ollama is not reachable on this device.") from error
    return "Model stopped and unloaded from memory."


async def chat_with_ollama(request: ChatRequest) -> ChatResponse:
    await _ensure_managed_model_exists(request.model_name)
    try:
        async with _create_ollama_client() as client:
            response = await client.post(
                "/api/chat",
                json={
                    "model": request.model_name,
                    "messages": [
                        {"role": message.role, "content": message.content}
                        for message in request.messages
                    ],
                    "stream": False,
                },
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as error:
        raise _ollama_http_error(
            error,
            "Ollama could not generate a chat response",
        ) from error
    except (httpx.HTTPError, ValueError) as error:
        raise RuntimeError("Could not complete the local Ollama chat request.") from error

    message = payload.get("message") if isinstance(payload, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise RuntimeError("Ollama returned a response in an unexpected format.")
    return ChatResponse(model_name=request.model_name, message=content)


async def delete_ollama_model(request: DeleteModelRequest) -> str:
    if not request.confirm:
        raise ValueError(
            "Set confirm=true after confirming removal of the Ollama model."
        )
    await _ensure_managed_model_exists(request.model_name)
    try:
        async with _create_ollama_client() as client:
            response = await client.request(
                "DELETE",
                "/api/delete",
                json={"model": request.model_name},
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise _ollama_http_error(error, "Ollama could not remove this model") from error
    except httpx.HTTPError as error:
        raise RuntimeError("Ollama is not reachable on this device.") from error
    return "Ollama model removed. The managed source files were kept."

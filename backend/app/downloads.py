import asyncio
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, quote, urlsplit

import httpx
from pydantic import BaseModel, Field


HUGGING_FACE_ROOT = "https://huggingface.co"
MODEL_EXTENSIONS = {".gguf", ".safetensors"}
METADATA_FILES = {
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "vocab.txt",
}
DISK_RESERVE_BYTES = 256 * 1024 * 1024
MAX_TREE_PAGES = 100
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


class ModelFile(BaseModel):
    path: str
    size_bytes: Optional[int] = None


class ModelFilesResponse(BaseModel):
    model_id: str
    files: list[ModelFile]


class DownloadRequest(BaseModel):
    model_id: str = Field(min_length=3, max_length=300)
    files: list[str] = Field(min_length=1, max_length=64)


class DownloadStatus(BaseModel):
    id: str
    model_id: str
    status: str
    files: list[str]
    current_file: Optional[str] = None
    downloaded_bytes: int = 0
    total_bytes: Optional[int] = None
    progress: Optional[float] = None
    speed_bytes_per_second: Optional[float] = None
    eta_seconds: Optional[int] = None
    error: Optional[str] = None


class InstalledModel(BaseModel):
    model_id: str
    files: list[ModelFile]
    total_bytes: int


class DownloadJob:
    def __init__(self, model_id: str, files: list[ModelFile]) -> None:
        self.id = str(uuid.uuid4())
        self.model_id = model_id
        self.files = files
        self.status = "queued"
        self.current_file: Optional[str] = None
        self.downloaded_bytes = 0
        self.total_bytes = (
            sum(file.size_bytes for file in files if file.size_bytes is not None)
            if all(file.size_bytes is not None for file in files)
            else None
        )
        self.error: Optional[str] = None
        self.started_at = time.monotonic()
        self.task: Optional[asyncio.Task[None]] = None

    def snapshot(self) -> DownloadStatus:
        speed: Optional[float] = None
        eta: Optional[int] = None
        progress: Optional[float] = None
        elapsed = time.monotonic() - self.started_at
        if self.status == "downloading" and elapsed > 0:
            speed = self.downloaded_bytes / elapsed
            if self.total_bytes is not None and speed > 0:
                eta = max(
                    0,
                    int((self.total_bytes - self.downloaded_bytes) / speed),
                )
        if self.total_bytes is not None and self.total_bytes > 0:
            progress = min(1.0, self.downloaded_bytes / self.total_bytes)
        if self.status == "completed":
            progress = 1.0
        return DownloadStatus(
            id=self.id,
            model_id=self.model_id,
            status=self.status,
            files=[file.path for file in self.files],
            current_file=self.current_file,
            downloaded_bytes=self.downloaded_bytes,
            total_bytes=self.total_bytes,
            progress=progress,
            speed_bytes_per_second=speed,
            eta_seconds=eta,
            error=self.error,
        )


_jobs: dict[str, DownloadJob] = {}


def validate_model_id(model_id: str) -> str:
    if not MODEL_ID_PATTERN.fullmatch(model_id):
        raise ValueError("Model ID must be in publisher/model format.")
    parts = model_id.split("/")
    if any(part in {".", ".."} or _is_windows_reserved_name(part) for part in parts):
        raise ValueError("Model ID contains an invalid path segment.")
    return model_id


def _is_windows_reserved_name(segment: str) -> bool:
    stem = segment.split(".", 1)[0].upper()
    return stem in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }


def validate_file_path(file_path: str) -> str:
    if not file_path or "\\" in file_path or "\x00" in file_path:
        raise ValueError("Model file path is invalid.")
    parts = file_path.split("/")
    if any(
        part in {"", ".", ".."}
        or any(character in part for character in '<>:"|?*')
        or part.endswith((" ", "."))
        or _is_windows_reserved_name(part)
        or any(ord(character) < 32 for character in part)
        for part in parts
    ):
        raise ValueError("Model file path must stay inside the model repository.")
    return file_path


def is_supported_model_file(file_path: str) -> bool:
    basename = file_path.rsplit("/", 1)[-1].lower()
    suffix = Path(basename).suffix
    return (
        suffix in MODEL_EXTENSIONS
        or basename.endswith(".safetensors.index.json")
        or basename in METADATA_FILES
    )


def storage_root() -> Path:
    configured = os.getenv("MODEL_STORAGE_DIR")
    root = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".ai-model-manager" / "models"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _assert_contained(path: Path, parent: Path) -> Path:
    resolved_parent = parent.resolve()
    relative = path.relative_to(parent)
    cursor = parent
    for segment in relative.parts:
        cursor = cursor / segment
        if cursor.is_symlink():
            raise ValueError("Symbolic links are not allowed in managed model storage.")
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_parent):
        raise ValueError("Resolved storage path escapes the managed model directory.")
    return resolved_path


def model_directory(model_id: str, *, create: bool = False) -> Path:
    validate_model_id(model_id)
    root = storage_root()
    target = root.joinpath(*model_id.split("/"))
    _assert_contained(target, root)
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return _assert_contained(target, root)


def model_file_path(model_id: str, file_path: str, *, create: bool = False) -> Path:
    validate_file_path(file_path)
    model_dir = model_directory(model_id, create=create)
    target = model_dir.joinpath(*file_path.split("/"))
    if create:
        target.parent.mkdir(parents=True, exist_ok=True)
    return _assert_contained(target, model_dir)


def _headers() -> dict[str, str]:
    token = os.getenv("HF_TOKEN")
    headers = {"Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _model_api_url(model_id: str) -> str:
    validate_model_id(model_id)
    return f"{HUGGING_FACE_ROOT}/api/models/{quote(model_id, safe='/')}/tree/main"


async def list_model_files(model_id: str) -> ModelFilesResponse:
    url = _model_api_url(model_id)
    try:
        async with _create_http_client() as client:
            params = {
                "recursive": "true",
                "expand": "true",
                "limit": "100",
            }
            payload: list[object] = []
            seen_cursors: set[str] = set()
            for page_number in range(MAX_TREE_PAGES):
                response = await client.get(url, params=params, headers=_headers())
                response.raise_for_status()
                page = response.json()
                if not isinstance(page, list):
                    raise RuntimeError(
                        "Hugging Face returned an unexpected file listing."
                    )
                payload.extend(page)

                next_link = response.links.get("next", {}).get("url")
                if not next_link:
                    break
                parsed_link = urlsplit(next_link)
                parsed_api_url = urlsplit(url)
                cursor_values = parse_qs(parsed_link.query).get("cursor", [])
                if (
                    parsed_link.scheme != "https"
                    or parsed_link.netloc != parsed_api_url.netloc
                    or parsed_link.path != parsed_api_url.path
                    or not cursor_values
                    or cursor_values[0] in seen_cursors
                ):
                    raise RuntimeError(
                        "Hugging Face returned an invalid file-list pagination link."
                    )
                seen_cursors.add(cursor_values[0])
                params["cursor"] = cursor_values[0]
            else:
                raise RuntimeError(
                    "The model repository has too many files to list safely."
                )
    except httpx.TimeoutException as error:
        raise RuntimeError("Hugging Face file listing timed out.") from error
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        if status in {401, 403, 404}:
            raise RuntimeError(
                "The model files are unavailable. The repository may be gated, private, or missing."
            ) from error
        raise RuntimeError(f"Hugging Face returned HTTP {status} while listing files.") from error
    except (httpx.HTTPError, ValueError) as error:
        raise RuntimeError("Could not retrieve the model file list.") from error

    files: list[ModelFile] = []
    for item in payload:
        if not isinstance(item, dict) or item.get("type") != "file":
            continue
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not is_supported_model_file(raw_path):
            continue
        try:
            safe_path = validate_file_path(raw_path)
        except ValueError:
            continue
        size = item.get("size")
        files.append(
            ModelFile(
                path=safe_path,
                size_bytes=size if isinstance(size, int) and size >= 0 else None,
            )
        )
    return ModelFilesResponse(model_id=model_id, files=files)


def _create_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=20.0, read=60.0, write=20.0, pool=20.0),
        follow_redirects=True,
    )


def _link_without_overwrite(part_path: Path, final_path: Path) -> None:
    _assert_contained(part_path, part_path.parent)
    _assert_contained(final_path, final_path.parent)
    try:
        os.link(part_path, final_path)
    except FileExistsError as error:
        raise FileExistsError("A completed file already exists; it was not overwritten.") from error
    except OSError as error:
        raise RuntimeError("Could not safely finalize the downloaded file.") from error
    part_path.unlink()


async def _download_file(job: DownloadJob, model_file: ModelFile) -> None:
    final_path = model_file_path(job.model_id, model_file.path, create=True)
    part_path = model_file_path(
        job.model_id,
        f"{model_file.path}.part",
        create=True,
    )
    if final_path.exists():
        raise FileExistsError(
            f"{model_file.path} already exists. Existing model files are never overwritten."
        )

    offset = part_path.stat().st_size if part_path.exists() else 0
    if part_path.exists() and part_path.stat().st_nlink != 1:
        raise RuntimeError(
            f"Partial file {model_file.path} has an unexpected hard-link count and was not modified."
        )
    if model_file.size_bytes is not None and offset > model_file.size_bytes:
        raise RuntimeError(
            f"Partial file {model_file.path} is larger than the listed file; it was preserved."
        )
    if (
        part_path.exists()
        and model_file.size_bytes is not None
        and offset == model_file.size_bytes
    ):
        _link_without_overwrite(part_path, final_path)
        return

    url = (
        f"{HUGGING_FACE_ROOT}/{quote(job.model_id, safe='/')}"
        f"/resolve/main/{quote(model_file.path, safe='/')}"
    )
    headers = _headers()
    if offset:
        headers["Range"] = f"bytes={offset}-"

    async with _create_http_client() as client:
        async with client.stream("GET", url, headers=headers) as response:
            if response.status_code not in {200, 206}:
                if response.status_code in {401, 403, 404}:
                    raise RuntimeError(
                        f"Access to {model_file.path} was denied. The model may be gated or require an HF_TOKEN."
                    )
                response.raise_for_status()
                raise RuntimeError(
                    f"Hugging Face returned HTTP {response.status_code} for {model_file.path}."
                )

            append = offset > 0 and response.status_code == 206
            if response.status_code == 206:
                content_range = response.headers.get("content-range", "")
                if not content_range.startswith(f"bytes {offset}-"):
                    raise RuntimeError(
                        "The file server returned an invalid resume range; the partial file was preserved."
                    )
            elif offset:
                job.downloaded_bytes = max(0, job.downloaded_bytes - offset)
                offset = 0

            mode = "ab" if append else "wb"
            with part_path.open(mode) as output:
                async for chunk in response.aiter_raw():
                    if not chunk:
                        continue
                    output.write(chunk)
                    job.downloaded_bytes += len(chunk)

    actual_size = part_path.stat().st_size
    if model_file.size_bytes is not None and actual_size != model_file.size_bytes:
        raise RuntimeError(
            f"Downloaded size for {model_file.path} does not match Hugging Face metadata. "
            "The partial file is preserved so the download can be resumed."
        )
    _link_without_overwrite(part_path, final_path)


async def _run_job(job: DownloadJob) -> None:
    job.status = "downloading"
    try:
        for model_file in job.files:
            job.current_file = model_file.path
            await _download_file(job, model_file)
        job.status = "completed"
        job.current_file = None
    except asyncio.CancelledError:
        job.status = "cancelled"
        job.current_file = None
        raise
    except Exception as error:
        job.status = "failed"
        job.error = str(error)
        job.current_file = None


async def start_download(request: DownloadRequest) -> DownloadStatus:
    validate_model_id(request.model_id)
    if any(
        job.model_id == request.model_id
        and job.status in {"queued", "downloading"}
        for job in _jobs.values()
    ):
        raise FileExistsError("A download for this model is already active.")
    if len(set(request.files)) != len(request.files):
        raise ValueError("The file selection contains duplicates.")
    if len(request.files) > 64:
        raise ValueError("A download can include at most 64 files.")

    listed_files = await list_model_files(request.model_id)
    available = {model_file.path: model_file for model_file in listed_files.files}
    selected: list[ModelFile] = []
    for path in request.files:
        validate_file_path(path)
        if path not in available:
            raise ValueError(
                f"{path} is not an allowed downloadable file in this model repository."
            )
        selected.append(available[path])

    root = storage_root()
    model_directory(request.model_id, create=True)
    existing_partials = 0
    required_bytes = 0
    all_sizes_known = True
    for model_file in selected:
        final_path = model_file_path(request.model_id, model_file.path, create=True)
        if final_path.exists():
            raise FileExistsError(
                f"{model_file.path} is already installed. Existing files are never overwritten."
            )
        part_path = model_file_path(
            request.model_id,
            f"{model_file.path}.part",
            create=True,
        )
        partial_size = part_path.stat().st_size if part_path.exists() else 0
        existing_partials += partial_size
        if model_file.size_bytes is None:
            all_sizes_known = False
        else:
            required_bytes += max(0, model_file.size_bytes - partial_size)

    if all_sizes_known:
        free_bytes = shutil.disk_usage(root).free
        if free_bytes < required_bytes + DISK_RESERVE_BYTES:
            raise OSError(
                "There may not be enough free disk space. Free space or choose another managed storage location."
            )

    job = DownloadJob(request.model_id, selected)
    job.downloaded_bytes = existing_partials
    _jobs[job.id] = job
    job.task = asyncio.create_task(_run_job(job))
    return job.snapshot()


def get_download(job_id: str) -> DownloadStatus:
    job = _jobs.get(job_id)
    if job is None:
        raise KeyError("Download job was not found.")
    return job.snapshot()


def cancel_download(job_id: str) -> DownloadStatus:
    job = _jobs.get(job_id)
    if job is None:
        raise KeyError("Download job was not found.")
    if job.status in {"completed", "failed", "cancelled"}:
        raise ValueError(f"Cannot cancel a {job.status} download.")
    if job.task is not None:
        job.task.cancel()
    job.status = "cancelled"
    return job.snapshot()


def list_installed_models() -> list[InstalledModel]:
    root = storage_root()
    installed: list[InstalledModel] = []
    for publisher_dir in root.iterdir():
        if not publisher_dir.is_dir() or publisher_dir.is_symlink():
            continue
        for model_dir in publisher_dir.iterdir():
            if not model_dir.is_dir() or model_dir.is_symlink():
                continue
            model_id = f"{publisher_dir.name}/{model_dir.name}"
            if not MODEL_ID_PATTERN.fullmatch(model_id):
                continue
            files: list[ModelFile] = []
            for path in model_dir.rglob("*"):
                if not path.is_file() or path.is_symlink() or path.name.endswith(".part"):
                    continue
                try:
                    relative = path.relative_to(model_dir).as_posix()
                    _assert_contained(path, model_dir)
                    if is_supported_model_file(relative):
                        files.append(
                            ModelFile(path=relative, size_bytes=path.stat().st_size)
                        )
                except (OSError, ValueError):
                    continue
            if files:
                installed.append(
                    InstalledModel(
                        model_id=model_id,
                        files=files,
                        total_bytes=sum(file.size_bytes or 0 for file in files),
                    )
                )
    return installed


def delete_installed_model(model_id: str) -> None:
    validate_model_id(model_id)
    if any(
        job.model_id == model_id and job.status in {"queued", "downloading"}
        for job in _jobs.values()
    ):
        raise ValueError("Cancel active downloads before removing this model.")

    root = storage_root()
    model_dir = model_directory(model_id)
    if not model_dir.exists():
        raise FileNotFoundError("No managed model files were found for this model.")
    _assert_contained(model_dir, root)
    if model_dir.is_symlink():
        raise ValueError("Refusing to remove a symbolic link from model storage.")
    shutil.rmtree(model_dir)

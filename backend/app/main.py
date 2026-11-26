from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import HTTPException, Query

from app.hardware import HardwareInfo, detect_hardware
from app.catalog import ModelSearchResponse, search_models
from app.compatibility import CompatibilityRequest, CompatibilityResponse, estimate_compatibility
from app.downloads import (
    DownloadRequest,
    DownloadStatus,
    InstalledModel,
    ModelFilesResponse,
    cancel_download,
    delete_installed_model,
    get_download,
    list_installed_models,
    list_model_files,
    start_download,
)
from app.ollama_runtime import (
    ChatRequest,
    ChatResponse,
    DeleteModelRequest,
    ImportModelRequest,
    ImportModelResponse,
    ModelActionRequest,
    OllamaStatus,
    chat_with_ollama,
    delete_ollama_model,
    get_ollama_status,
    import_gguf_model,
    load_ollama_model,
    stop_ollama_model,
)


class HealthResponse(BaseModel):
    status: str
    service: str


class RuntimeMessageResponse(BaseModel):
    message: str


app = FastAPI(
    title="AI Model Manager",
    description="Local-first API for discovering and managing AI models.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="ai-model-manager-backend")


@app.get("/hardware", response_model=HardwareInfo)
def hardware() -> HardwareInfo:
    return detect_hardware()


@app.get("/models/search", response_model=ModelSearchResponse)
def model_search(
    query: str = Query(default="", max_length=200),
    limit: int = Query(default=20, ge=1, le=100),
) -> ModelSearchResponse:
    try:
        models = search_models(query.strip(), limit)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return ModelSearchResponse(
        models=models,
        query=query.strip(),
        limit=limit,
    )


@app.post("/compatibility/estimate", response_model=CompatibilityResponse)
def compatibility_estimate(request: CompatibilityRequest) -> CompatibilityResponse:
    return estimate_compatibility(detect_hardware(), request)


@app.get("/models/files", response_model=ModelFilesResponse)
async def model_files(model_id: str = Query(min_length=3, max_length=300)) -> ModelFilesResponse:
    try:
        return await list_model_files(model_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/downloads", response_model=DownloadStatus, status_code=202)
async def create_download(request: DownloadRequest) -> DownloadStatus:
    try:
        return await start_download(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(status_code=507, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.get("/downloads/installed", response_model=list[InstalledModel])
def installed_models() -> list[InstalledModel]:
    return list_installed_models()


@app.delete("/downloads/installed", status_code=204)
def remove_installed_model(
    model_id: str = Query(min_length=3, max_length=300),
    confirm: bool = Query(default=False),
) -> None:
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Set confirm=true to permanently remove this managed model.",
        )
    try:
        delete_installed_model(model_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(status_code=500, detail="Could not remove managed model files.") from error


@app.get("/downloads/{job_id}", response_model=DownloadStatus)
def download_status(job_id: str) -> DownloadStatus:
    try:
        return get_download(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.delete("/downloads/{job_id}", response_model=DownloadStatus)
def stop_download(job_id: str) -> DownloadStatus:
    try:
        return cancel_download(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/runtime/ollama", response_model=OllamaStatus)
async def ollama_status() -> OllamaStatus:
    return await get_ollama_status()


@app.post("/runtime/ollama/import", response_model=ImportModelResponse)
def ollama_import(request: ImportModelRequest) -> ImportModelResponse:
    try:
        return import_gguf_model(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(status_code=500, detail="Ollama model import failed.") from error


@app.post("/runtime/ollama/load", response_model=RuntimeMessageResponse)
async def ollama_load(request: ModelActionRequest) -> RuntimeMessageResponse:
    try:
        return RuntimeMessageResponse(message=await load_ollama_model(request))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/runtime/ollama/stop", response_model=RuntimeMessageResponse)
async def ollama_stop(request: ModelActionRequest) -> RuntimeMessageResponse:
    try:
        return RuntimeMessageResponse(message=await stop_ollama_model(request))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/runtime/ollama/chat", response_model=ChatResponse)
async def ollama_chat(request: ChatRequest) -> ChatResponse:
    try:
        return await chat_with_ollama(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.delete("/runtime/ollama/models", response_model=RuntimeMessageResponse)
async def ollama_delete(request: DeleteModelRequest) -> RuntimeMessageResponse:
    try:
        return RuntimeMessageResponse(message=await delete_ollama_model(request))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

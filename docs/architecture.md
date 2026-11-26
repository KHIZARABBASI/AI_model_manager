# Architecture

## Current foundation

The application is split into independently startable processes:

- `frontend/`: Next.js user interface. It communicates with the API through `NEXT_PUBLIC_API_BASE_URL`.
- `backend/`: FastAPI application. It binds to loopback by default and exposes `/health`.

The backend also exposes `GET /hardware`, which reports operating-system,
CPU, RAM, disk, and detectable GPU information. GPU VRAM is marked with its
source, and missing or uncertain values are returned as `null` with detection
warnings rather than estimates.

The backend also exposes `GET /models/search?query=&limit=20`, which
proxies the supported Hugging Face model API. It restricts discovery to
text-generation models and normalizes licenses, downloads, repository size,
gating status, and source links. Hugging Face tokens are optional and may be
provided through the `HF_TOKEN` environment variable; they must never be
committed or logged.

`POST /compatibility/estimate` combines detected hardware with repository
size metadata to produce a labeled status: likely to run comfortably,
may run but could be slow, likely to run out of memory, or not enough
information. Estimates use explicit memory and disk overhead assumptions,
prefer detected NVIDIA VRAM when available, and do not treat parameter count
or advertised file size as measured runtime memory.

Managed downloads use `GET /models/files` to list only selected model formats
and data files, `POST /downloads` to begin a transfer, `GET /downloads/{id}`
to poll progress, and `DELETE /downloads/{id}` to cancel. A cancelled or
interrupted transfer keeps its partial file for a later Range-based resume.
Completed files are finalized without overwriting existing files and are
stored under the resolved managed model directory (`MODEL_STORAGE_DIR`, or
`%USERPROFILE%\\.ai-model-manager\\models` by default). Repository scripts and
unsupported formats are not offered for download or executed. Installed
models can be listed with `GET /downloads/installed`; deleting one requires
an explicit confirmation parameter and UI confirmation.

## Ollama runtime

The optional adapter in `backend/app/ollama_runtime.py` connects only to
Ollama's loopback API at `127.0.0.1:11434`; model import invokes the separately
installed `ollama` CLI. The user must install and start Ollama. The app does
not install, launch, or update the runtime.

`GET /runtime/ollama` reports runtime availability and lists only models
using the AI Model Manager naming prefix. `POST /runtime/ollama/import`
accepts GGUF files from managed storage, creates a temporary staging directory
under that storage with hardlinks to the source files, and runs `ollama create`
with a generated Modelfile. Split GGUF imports require all shards. The staging
directory is removed after import; managed source files remain. Existing
Ollama model names are not overwritten.

The API also exposes local load, stop, and chat endpoints. Model removal
requires explicit confirmation and removes only the Ollama copy, not its
managed source. Chat history stays in the frontend page's memory, and chat
requests go only to the loopback Ollama service. Import is limited to GGUF;
other model formats are not loaded through this adapter. Runtime availability
has been verified; import and live inference still require a managed GGUF
model to exercise.

No model files, chat history, telemetry, or credentials are stored in the repository.

## API boundary

FastAPI owns hardware detection, model discovery, compatibility estimation, downloads, runtime management, and chat. The frontend should treat the API as the source of truth for application state and avoid direct filesystem or Hugging Face access.

## Module boundaries

The backend is organized into replaceable modules for:

- hardware detection
- Hugging Face catalog access
- compatibility estimation
- managed downloads
- runtime adapters
- local chat and settings

Ollama is the initial runtime adapter. Additional runtimes can implement the
same user-facing model-management and chat workflows.

## Privacy and networking

The services use loopback URLs for local development. The frontend may call the backend; the backend will not send prompts or telemetry outside the device unless a future explicit feature requires network access. Hugging Face browsing and model downloads are expected to require internet and must be shown clearly in the UI.

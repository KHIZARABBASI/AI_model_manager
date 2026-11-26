# AI Model Manager

A local-first application for discovering, downloading, managing, and running AI models on a user's own computer. The MVP currently includes hardware reporting, Hugging Face model discovery, compatibility estimates, managed downloads, and optional Ollama integration.

## Technology

- Frontend: Next.js, React, and TypeScript
- Backend: Python, FastAPI, and Uvicorn
- Local runtime: Ollama (manual installation required; GGUF import and local chat are supported)
- Model discovery: Hugging Face APIs

## Prerequisites

- Node.js 20 or newer
- npm
- Python 3.12 or newer

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:3000. Copy `frontend/.env.example` to `frontend/.env.local` if a backend URL other than `http://127.0.0.1:8000` is needed.

## Backend

Create and use a local virtual environment:

```powershell
cd backend
py -3.14 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install "httpx>=0.28,<1" "pytest>=8,<9"
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API documentation is available at http://127.0.0.1:8000/docs. The health endpoint is `GET /health`.
The local hardware summary endpoint is `GET /hardware`.
Model discovery is available at `GET /models/search`.
Compatibility estimates are available at `POST /compatibility/estimate`.
Managed downloads are available through the `GET /models/files`,
`POST /downloads`, `GET /downloads/{id}`, and
`DELETE /downloads/{id}` endpoints. Model files are stored under
`%USERPROFILE%\.ai-model-manager\models` by default; set
`MODEL_STORAGE_DIR` before starting the backend to choose another location.
Only SafeTensors, GGUF, and selected tokenizer/configuration files are offered.
Downloads resume from preserved `.part` files when the Hugging Face file
server supports byte ranges. Existing completed files are never overwritten.
Removing a managed model requires explicit confirmation in the UI and the
backend request. The downloader stores files only; it does not execute or load
repository code.
Active job status is held in backend memory. If the backend restarts, partial
files remain on disk; list the model files and start the download again to
resume where the upstream server supports it.

## Local Ollama runtime

Install Ollama yourself from [ollama.com/download](https://ollama.com/download)
and start its local service. The Ollama CLI must be available on `PATH`; the
backend connects only to `127.0.0.1:11434`. AI Model Manager does not install
or start Ollama. Check availability at `GET /runtime/ollama`.

The UI can import installed GGUF files into Ollama, load or stop app-managed
models, and chat locally. Split GGUF files are supported when all shards are
installed. SafeTensors and other formats are not importable through this
Ollama path. Importing creates a separate Ollama-managed copy; removing that
copy requires confirmation and leaves the original managed files untouched.
Chat history is kept in page memory only. Prompts are sent to the local Ollama
service and are not sent to a cloud API by this feature.

## Tests

Frontend:

```powershell
cd frontend
npm run lint
npm run build
```

Backend:

```powershell
cd backend
python -m pytest
```

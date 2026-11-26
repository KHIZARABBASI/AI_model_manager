# Development Roadmap

## Phase 1 - MVP: Local Model Manager

- Repository and independent frontend/backend startup
- Hardware and operating-system detection
- Hugging Face model search
- Model detail metadata and license display
- Conservative compatibility estimates (first implementation complete)
- Managed model downloads with progress, cancellation, and resume (first
  implementation complete)
- Managed installed-model library
- Ollama runtime integration for managed GGUF files (first implementation complete)
- Local chat with app-managed Ollama models (first implementation complete)
- Core tests and user documentation

## Phase 2 - Improved Local Experience

- Additional runtime adapters such as llama.cpp
- Improved quantization recommendations
- Hardware benchmarks
- Configurable model storage
- Desktop packaging after the web prototype is stable

## Phase 3 - Cloud Lab

- User-controlled notebook generation
- Explicit provider setup and usage instructions
- No permanent or unrestricted free-GPU hosting workflow

## Phase 4 - Additional AI Tasks

Only after the core MVP is stable:

- Embeddings
- Vision
- Speech
- RAG
- Agent workflows

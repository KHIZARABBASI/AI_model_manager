# AGENTS.md

## 1. Project Purpose

Build an open-source tool that makes it easy for people to discover, download, manage, and run AI models on their own computers.

The product should help users who do not understand model formats, quantization, RAM/VRAM requirements, or inference runtimes. It should recommend models their hardware can realistically run and automate setup wherever possible.

**Core product idea:** "Find the best AI model my computer can run, install it, and use it without needing paid API access."

Cloud notebook support (such as user-controlled Colab/Kaggle notebook workflows) is a later feature. Do not build workflows that misuse free notebook services as permanent, always-on backend infrastructure or bypass their usage restrictions.

## 2. Initial Product Scope (MVP)

Prioritize a Windows-friendly local application/prototype with these capabilities:

1. Detect hardware and operating-system details: CPU, system RAM, GPU(s), VRAM where detectable, free disk space.
2. Search and browse Hugging Face models, initially prioritizing text-generation/LLM models.
3. Show model metadata: model name, publisher, task, license (when available), download size, formats, quantization, and source link.
4. Estimate whether a model can run on the detected hardware. Estimates must be labeled as estimates and explain their assumptions.
5. Recommend suitable model variants/quantizations when possible.
6. Download models with progress, cancellation, and resumable behavior where supported.
7. Manage installed models: list, inspect, launch, stop, and remove.
8. Run local models by integrating with an existing inference runtime rather than implementing inference from scratch.
9. Provide a simple chat UI for supported models.
10. Clearly show when a model is running locally and whether any data is sent outside the device.

Do not expand the MVP into image generation, speech, embeddings, agents, RAG, or cloud execution until the core workflow works end-to-end. Keep the architecture extensible for these later capabilities.

## 3. Product Principles

- Local-first: local execution and user data should be the default.
- No paid API is required for the MVP.
- Honest resource estimates: never promise a model will run based only on parameter count.
- Explain recommendations in plain language.
- Respect model licenses, source terms, and runtime licenses.
- Never silently upload user prompts, files, telemetry, or model data.
- Make network activity explicit. Hugging Face browsing/downloads and any optional cloud workflows require internet.
- Avoid unnecessary complexity. Prefer reliable, existing tools and libraries over custom infrastructure.
- Do not pretend Colab/Kaggle free GPU access is guaranteed, permanent, or unrestricted.

## 4. Suggested Architecture

Keep modules separated so components can be replaced without rewriting the application.

### Frontend
- Next.js
- TypeScript
- Simple, accessible UI
- Model discovery, hardware summary, download manager, installed-model library, and chat

### Backend
- Python
- FastAPI
- API for hardware detection, model catalog, compatibility estimates, downloads, runtime management, and chat

### Local inference
- Integrate an existing runtime.
- Initial integration target: Ollama, if installed or installable through a documented and user-approved process.
- Keep a runtime adapter/interface so other runtimes (for example llama.cpp) can be added later.
- Do not build a custom inference engine.

### Model discovery
- Use Hugging Face APIs and/or supported metadata endpoints.
- Do not scrape pages when a supported API is available.
- Handle missing metadata, gated models, authentication requirements, and licenses explicitly.
- Never store Hugging Face tokens in source code or logs.

### Storage
- Store application settings, model metadata, download state, and chat history locally.
- Make model storage location configurable.
- Do not delete or overwrite user model files without explicit confirmation.
- Keep secrets out of persistent logs.

### Desktop packaging
- Initially, prioritize a working local web application.
- Keep frontend/backend boundaries suitable for later packaging as a desktop app.
- Desktop packaging (for example, Tauri or another suitable shell) is a later milestone; do not add it prematurely.

## 5. MVP User Workflow

The intended happy path is:

1. User launches the application.
2. Application detects available hardware and storage.
3. User searches or browses models.
4. User opens a model detail page and sees its license, size, format, and estimated hardware requirements.
5. Application recommends a compatible option, including quantized variants where available.
6. User downloads the model and sees progress.
7. User launches the model using the supported local runtime.
8. User chats with it through the app.
9. User can stop the model or remove it from the library.

Keep this workflow functional before adding secondary features.

## 6. Hardware and Compatibility Estimation

Compatibility estimates must be conservative and transparent.

Consider, where data is available:
- Model weight size and quantization
- RAM
- VRAM
- CPU/GPU capability
- Context length and KV-cache overhead
- Runtime overhead
- Disk space for download and installation
- Whether GPU offloading is supported by the selected runtime

Display statuses such as:
- **Likely to run comfortably**
- **May run, but could be slow**
- **Likely to run out of memory**
- **Not enough information**

Explain why each recommendation is made. Do not treat parameter count alone as a reliable memory estimate. Do not imply that a model's advertised file size equals total runtime memory usage.

## 7. Download and Model Management Requirements

- Show download progress, speed, downloaded bytes, total bytes (if known), and estimated remaining time (if calculable).
- Support cancellation.
- Support resuming interrupted downloads where the source/client supports it.
- Validate files where checksums or other integrity metadata are available.
- Handle disk-space shortages and network errors with useful messages.
- Avoid duplicate downloads where possible.
- Never remove files outside the application's managed model directory unless the user explicitly chooses a location and confirms the action.
- Never execute arbitrary code from a model repository as part of downloading or loading a model.
- Warn about unsafe or custom model formats and recommend safer formats/runtime paths.

## 8. Privacy, Security, and Safety

- Local chat prompts and chat history remain local by default.
- No telemetry by default. If telemetry is ever proposed, it must be opt-in and documented.
- Do not expose the local API to the public network by default.
- Bind local services to loopback unless the user explicitly changes the setting.
- Validate paths and prevent path traversal during model downloads/extraction.
- Do not place API keys, access tokens, or credentials in repository files.
- Redact credentials from logs and error reports.
- Treat model cards, repository content, downloaded metadata, and model outputs as untrusted input.
- Do not run model-repository scripts or install arbitrary dependencies automatically without a clear user-consent step.
- Confirm destructive actions (model deletion, reset, overwrite, changing storage locations).
- Use safe defaults and provide actionable error messages.

## 9. Development Rules for AI Agents

Before making changes:

1. Read this `AGENTS.md` completely.
2. Inspect the existing repository, current code, configuration, and documentation.
3. Identify the current state of the project. Do not assume the repository is empty or that previous work is correct.
4. If expected files or folders are missing, create only what is needed for the task.
5. Follow existing patterns and naming conventions unless there is a clear reason to improve them.
6. Give a short implementation plan for multi-step tasks before making significant changes.

While implementing:

- Make small, coherent changes.
- Prefer readable, maintainable code over clever abstractions.
- Avoid unnecessary dependencies.
- Do not create placeholder implementations that appear functional but are not.
- Do not silently change unrelated functionality.
- Keep secrets, local environment files, model files, and generated artifacts out of version control.
- Add/update tests for meaningful behavior.
- Handle errors explicitly.
- Keep user-facing messages understandable to nontechnical users.

After implementing:

1. Run relevant tests, linting, type checks, and/or build commands that are available.
2. Report exactly what was changed.
3. Report commands/tests run and their results.
4. Clearly state anything not implemented, not tested, or requiring manual setup.
5. Update documentation only when the change makes existing documentation inaccurate or a new feature/setup step needs documenting.

## 10. Documentation Maintenance

Documentation should stay accurate but should not be rewritten on every task.

- Update documentation when architecture, setup, environment variables, commands, APIs, features, limitations, or user workflows change.
- Do not edit documentation merely to make an unnecessary cosmetic change.
- Keep `AGENTS.md` focused on stable project-wide instructions and principles.
- Put setup instructions in `README.md`.
- Put architecture decisions and diagrams in `docs/architecture.md` when needed.
- Put API documentation or endpoint details in the appropriate API docs when needed.
- Put active development plans in `docs/roadmap.md` or task-specific plans, if such documents exist.
- If a referenced documentation file does not exist and the current task needs it, create it.
- When changing a documented feature, update the relevant section instead of duplicating information in multiple files.
- If a documentation file becomes obsolete, explain why and propose removal; do not delete it without a clear reason or user authorization where appropriate.

## 11. Proposed Repository Layout

Adjust to the repository's actual state and avoid creating every folder upfront.

```text
project-root/
├── AGENTS.md
├── README.md
├── .gitignore
├── frontend/
│   └── ...
├── backend/
│   └── ...
├── docs/
│   ├── architecture.md
│   └── roadmap.md
└── tests/
    └── ...
```

This is a suggested structure, not a requirement to create all folders immediately.

## 12. Roadmap (High Level)

### Phase 1 — MVP: Local Model Manager
- Repository setup and basic UI/API
- Hardware detection
- Hugging Face model search
- Model details and license display
- Compatibility estimator
- Model download management
- Ollama integration (initial runtime)
- Local chat
- Basic tests and documentation

### Phase 2 — Improved Local Experience
- Additional runtime integration (such as llama.cpp)
- Better quantization and model recommendations
- Benchmarks
- Model storage management
- Desktop packaging after validating the web-based local prototype

### Phase 3 — Cloud Lab (User-Controlled)
- Generate notebook files for supported providers
- Provide setup and inference instructions
- User opens and runs notebooks in their own account
- Respect each provider's quotas, policies, session limits, and terms
- No attempt to bypass restrictions or create permanent free GPU hosting

### Phase 4 — More AI Tasks
Only after the core MVP is stable:
- Embedding models
- Vision models
- Speech models
- RAG workflows
- Agent workflows
- Optional integrations

## 13. Definition of Done

A task is not complete merely because code was written.

For each task, as applicable:
- Feature works through its intended workflow.
- Relevant error cases are handled.
- Tests are added or updated.
- Existing tests and checks pass, or failures are reported accurately.
- Documentation is updated if behavior/setup has changed.
- No secrets or unintended generated files are committed.
- The agent reports changed files, verification performed, known limitations, and next steps.

## 14. Agent Communication

Keep progress and final reports concise and factual.

For significant work, report:
- **Plan**
- **Changes made**
- **Verification**
- **Limitations / follow-up**

Do not claim a feature, test, build, download, or integration works unless it was actually verified.

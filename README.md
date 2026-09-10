# Open 3D Model Generator — Local AI Edition

An open-source, privacy-first **text-to-3D** application. The repository is intentionally tiny; large AI weights are downloaded and cached on the user's machine at runtime.

## Real local AI

The primary engine is **OpenAI Shap-E**, an actual text-conditioned 3D generation model. It runs through the local Python backend rather than a hosted generation API.

- Text → 3D locally
- CPU fallback is possible, but GPU is strongly recommended
- Model weights are cached locally after first use
- Generated GLB files are saved locally
- No prompt or model is sent to a remote service by the application

Shap-E's official repository documents text-conditional 3D generation and is MIT licensed:
https://github.com/openai/shap-e

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
```

### 2. Install the local engine

Install a suitable PyTorch build for your machine first, then:

```bash
pip install -r backend/requirements.txt
```

### 3. Start the local server

```bash
python run-local-ai.py
```

Open `index.html` in a browser.

The UI talks only to `127.0.0.1:8787`.

## Architecture

```text
Browser UI
   │
   ├── local REST API ───────────────┐
   │                                 ▼
   │                         FastAPI local server
   │                                 │
   │                    ┌────────────┴────────────┐
   │                    ▼                         ▼
   │                 Shap-E               Procedural fallback
   │                    │
   ▼                    ▼
Three.js/model-viewer       local GLB output
```

## Model storage

Do **not** commit model weights to Git. They can be many gigabytes, while this source repository stays well below 40 MB.

You can point Hugging Face / model caches to another drive using the normal cache environment variables supported by the model stack.

## Planned engines

The backend is deliberately adapter-based so additional local engines can be added without changing the UI. Candidates include newer text-to-3D pipelines and image-to-3D systems such as Hunyuan3D and TRELLIS. Their model weights and licenses must be reviewed separately before bundling or redistributing anything.

## Security

The default server binds to `127.0.0.1`, not the public network. Keep it local unless you intentionally add authentication and network controls.

## License

This application code is MIT licensed. Third-party engines and model weights retain their own licenses; see `THIRD-PARTY-NOTICES.md`.

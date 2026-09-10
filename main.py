import os
import uuid
import traceback
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

app = FastAPI(title="Open 3D Model Generator Local AI", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=2, max_length=2000)
    engine: str = "shap-e"
    seed: int = 0
    guidance_scale: float = 15.0
    steps: int = 64

@app.get("/api/health")
def health():
    import torch
    return {
        "ok": True,
        "local": True,
        "cuda": bool(torch.cuda.is_available()),
        "engines": ["shap-e", "procedural"],
    }

@app.get("/api/engines")
def engines():
    return {
        "engines": [
            {"id":"shap-e","name":"Shap-E","type":"text-to-3d","local":True,
             "note":"Real local text-conditioned 3D generation; weights download on first run."},
            {"id":"procedural","name":"Procedural fallback","type":"text-to-mesh","local":True,
             "note":"Instant fallback for offline/no-PyTorch machines."}
        ]
    }

@app.post("/api/generate")
def generate(req: GenerateRequest):
    job = uuid.uuid4().hex[:12]
    out = OUT / f"{job}.glb"
    try:
        if req.engine == "shap-e":
            from backend.engines.shape_e import ShapeEEngine
            engine = ShapeEEngine()
            mesh = engine.generate(
                req.prompt, seed=req.seed,
                guidance_scale=req.guidance_scale,
                karras_steps=req.steps
            )
            engine.save_glb(mesh, out)
        elif req.engine == "procedural":
            from backend.engines.procedural import ProceduralEngine
            mesh = ProceduralEngine().generate(req.prompt)
            mesh.export(out)
        else:
            raise HTTPException(400, "Unknown engine")
        return {"job_id": job, "url": f"/api/files/{out.name}", "engine": req.engine}
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            500,
            f"Generation failed: {type(exc).__name__}: {exc}. "
            "Install the backend requirements and ensure enough RAM/VRAM is available."
        )

@app.get("/api/files/{name}")
def files(name: str):
    p = OUT / Path(name).name
    if not p.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(p, media_type="model/gltf-binary", filename=p.name)

"""
Real local text-to-3D engine using OpenAI Shap-E.

The Shap-E weights are downloaded on first use and cached outside the repository.
This keeps the GitHub repository small while allowing fully local generation after
the initial model download.
"""
from pathlib import Path
import io
import os
import torch

class ShapeEEngine:
    name = "shap-e"
    description = "Local text-to-3D generation with Shap-E"

    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir or os.getenv("MODEL_CACHE", "./models"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ready = False

    def load(self):
        if self._ready:
            return
        from shap_e.diffusion.sample import sample_latents
        from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
        from shap_e.models.download import load_model, load_config
        self.torch = torch
        self.sample_latents = sample_latents
        self.diffusion_from_config = diffusion_from_config
        self.xm = load_model("transmitter", device=self.device)
        self.text_model = load_model("text300M", device=self.device)
        self.diffusion = diffusion_from_config(load_config("diffusion"))
        self._ready = True

    @property
    def device(self):
        requested = os.getenv("DEVICE", "auto")
        if requested == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def generate(self, prompt, seed=0, guidance_scale=15.0, karras_steps=64):
        self.load()
        from shap_e.util.notebooks import create_pan_cameras, decode_latent_images
        from shap_e.util.collections import TriMesh
        from shap_e.util.notebooks import decode_latent_mesh
        torch.manual_seed(int(seed))

        latents = self.sample_latents(
            batch_size=1,
            model=self.text_model,
            diffusion=self.diffusion,
            guidance_scale=float(guidance_scale),
            model_kwargs={"texts": [prompt]},
            progress=False,
            clip_denoised=True,
            use_fp16=self.device == "cuda",
            karras_steps=int(karras_steps),
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        mesh = decode_latent_mesh(self.xm, latents[0]).tri_mesh()
        return mesh

    def save_glb(self, mesh, path):
        # Shap-E's mesh is convertible to trimesh through its vertex/face arrays.
        import trimesh
        vertices = mesh.verts.numpy()
        faces = mesh.faces.numpy()
        tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        tm.remove_unreferenced_vertices()
        tm.export(path)
        return path

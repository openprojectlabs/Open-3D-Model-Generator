"""Fast deterministic fallback for machines without AI dependencies."""
import math
import trimesh

class ProceduralEngine:
    name = "procedural"
    description = "Instant local procedural fallback"

    def generate(self, prompt):
        p = prompt.lower()
        if "sphere" in p or "ball" in p:
            return trimesh.creation.icosphere(subdivisions=3, radius=1)
        if "cylinder" in p:
            return trimesh.creation.cylinder(radius=0.8, height=1.8, sections=48)
        if "cone" in p:
            return trimesh.creation.cone(radius=0.9, height=1.8, sections=48)
        if "torus" in p or "ring" in p:
            return trimesh.creation.torus(major_radius=0.9, minor_radius=0.25)
        if "capsule" in p:
            return trimesh.creation.capsule(radius=0.45, height=1.5)
        return trimesh.creation.box(extents=(1.4, 1.0, 1.4))

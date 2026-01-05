from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import tempfile
import os
import numpy as np
import trimesh

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    file_data: str
    file_name: str
    restoration_type: str = "crown"

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    try:
        file_bytes = base64.b64decode(request.file_data)
        ext = os.path.splitext(request.file_name)[1].lower()
        
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(file_bytes)
            temp_path = f.name
        
        mesh = trimesh.load(temp_path)
        os.unlink(temp_path)
        
        # Calculate metrics
        convergence = calculate_convergence(mesh)
        occlusal = calculate_occlusal_reduction(mesh)
        finish_line = calculate_finish_line(mesh)
        undercuts = detect_undercuts(mesh)
        
        score = int((convergence["score"] + occlusal["score"] + finish_line["score"] + undercuts["score"]) / 4)
        
        return {
            "score": score,
            "convergence": convergence,
            "occlusalReduction": occlusal,
            "finishLine": finish_line,
            "undercuts": undercuts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def calculate_convergence(mesh):
    normals = mesh.face_normals
    z_axis = np.array([0, 0, 1])
    angles = np.arccos(np.clip(np.dot(normals, z_axis), -1, 1))
    avg_angle = np.degrees(np.mean(angles))
    
    if 4 <= avg_angle <= 8:
        return {"value": round(avg_angle, 1), "score": 95, "status": "success", "message": "Ideal taper angle"}
    elif 2 <= avg_angle <= 12:
        return {"value": round(avg_angle, 1), "score": 75, "status": "warning", "message": "Acceptable taper"}
    return {"value": round(avg_angle, 1), "score": 50, "status": "error", "message": "Taper needs adjustment"}

def calculate_occlusal_reduction(mesh):
    bounds = mesh.bounds
    height = bounds[1][2] - bounds[0][2]
    
    if height >= 1.5:
        return {"value": round(height, 2), "score": 95, "status": "success", "message": "Adequate reduction"}
    elif height >= 1.0:
        return {"value": round(height, 2), "score": 70, "status": "warning", "message": "Minimal reduction"}
    return {"value": round(height, 2), "score": 40, "status": "error", "message": "Insufficient reduction"}

def calculate_finish_line(mesh):
    edges = mesh.edges_unique_length
    smoothness = 1 - (np.std(edges) / np.mean(edges)) if np.mean(edges) > 0 else 0
    clarity = min(100, max(0, int(smoothness * 100)))
    
    if clarity >= 80:
        return {"clarity": clarity, "score": 95, "status": "success", "message": "Clear margin"}
    elif clarity >= 60:
        return {"clarity": clarity, "score": 70, "status": "warning", "message": "Margin needs refinement"}
    return {"clarity": clarity, "score": 40, "status": "error", "message": "Unclear margin"}

def detect_undercuts(mesh):
    normals = mesh.face_normals
    z_axis = np.array([0, 0, 1])
    dots = np.dot(normals, z_axis)
    undercut_faces = np.sum(dots < -0.1)
    
    if undercut_faces == 0:
        return {"detected": False, "depth": 0, "score": 100, "status": "success", "message": "No undercuts"}
    elif undercut_faces < len(normals) * 0.05:
        return {"detected": True, "depth": 0.2, "score": 70, "status": "warning", "message": "Minor undercuts"}
    return {"detected": True, "depth": 0.5, "score": 40, "status": "error", "message": "Significant undercuts"}

@app.get("/health")
async def health():
    return {"status": "ok"}

import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any

from model.inference import ChestXrayPipeline
from model.model import generate_dummy_weights
from agent.report_generator import ReportGeneratorAgent

# Initialize directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "output")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "densenet_dummy.pth")
THRESHOLDS_PATH = os.path.join(BASE_DIR, "model", "thresholds.json")
GUIDELINES_PATH = os.path.join(BASE_DIR, "agent", "rag_guidelines.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Generate dummy weights if they don't exist
if not os.path.exists(WEIGHTS_PATH):
    print("Generating dummy weights...")
    generate_dummy_weights(WEIGHTS_PATH)

# Initialize pipeline and agent
pipeline = ChestXrayPipeline(weights_path=WEIGHTS_PATH, thresholds_path=THRESHOLDS_PATH)
agent = ReportGeneratorAgent(guidelines_path=GUIDELINES_PATH)

app = FastAPI(title="Chest X-Ray Disease Classifier & Report Generator API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (uploads and output heatmaps)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))

class ReportRequest(BaseModel):
    detected_findings: List[str]
    triage: str
    predictions: Dict[str, Any]

@app.post("/api/predict")
async def predict_xray(file: UploadFile = File(...)):
    """
    Upload an X-ray image, run classification, and output Grad-CAM heatmap.
    """
    # Verify file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, and PNG images are supported.")
        
    # Save input image
    file_id = str(uuid.uuid4())
    input_filename = f"{file_id}{ext}"
    input_path = os.path.join(UPLOAD_DIR, input_filename)
    
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
        
    # Run prediction and Grad-CAM
    try:
        result = pipeline.predict(input_path, output_heatmap_dir=OUTPUT_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        
    # Build URL pathways for static file access
    heatmap_filename = f"gradcam_{input_filename}"
    heatmap_url = f"/static/output/{heatmap_filename}"
    
    return {
        "success": True,
        "input_image": f"/static/uploads/{input_filename}",
        "heatmap_image": heatmap_url if result["heatmap_path"] else None,
        "triage": result["triage"],
        "detected_findings": result["detected_findings"],
        "visualized_finding": result["visualized_finding"],
        "predictions": result["predictions"]
    }

@app.post("/api/report")
async def generate_clinical_report(request: ReportRequest):
    """
    Generate the agentic report (Clinical + Patient English/Hindi summaries).
    """
    try:
        reports = agent.generate_report(request.dict())
        return reports
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate reports: {str(e)}")

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "classes": len(pipeline.thresholds)}

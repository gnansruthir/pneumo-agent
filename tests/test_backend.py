import os
import sys
import pytest
from PIL import Image
from fastapi.testclient import TestClient

# Add base path to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app
from model.inference import ChestXrayPipeline

client = TestClient(app)

@pytest.fixture
def dummy_image(tmp_path):
    """Generates a dummy 224x224 grayscale/RGB chest x-ray mockup image for testing."""
    img_path = tmp_path / "dummy_xray.png"
    img = Image.new("RGB", (224, 224), color="gray")
    img.save(img_path)
    return str(img_path)

def test_pipeline_inference(dummy_image):
    """Tests that the inference pipeline correctly processes the image and outputs results."""
    from api.main import WEIGHTS_PATH, THRESHOLDS_PATH, OUTPUT_DIR
    
    pipeline = ChestXrayPipeline(weights_path=WEIGHTS_PATH, thresholds_path=THRESHOLDS_PATH)
    result = pipeline.predict(dummy_image, output_heatmap_dir=OUTPUT_DIR)
    
    assert "predictions" in result
    assert "triage" in result
    assert result["triage"] in ["Clear", "Follow-up", "Urgent"]
    assert "detected_findings" in result
    assert "heatmap_path" in result

def test_api_health():
    """Tests the health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["classes"] == 15

def test_api_predict(dummy_image):
    """Tests uploading an image to the predict endpoint."""
    with open(dummy_image, "rb") as f:
        response = client.post(
            "/api/predict",
            files={"file": ("test.png", f, "image/png")}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "triage" in data
    assert "detected_findings" in data
    assert "predictions" in data

def test_api_report():
    """Tests the report generation agent endpoint."""
    payload = {
        "detected_findings": ["Tuberculosis", "Effusion"],
        "triage": "Urgent",
        "predictions": {
            "Tuberculosis": {"probability": 0.85, "threshold": 0.22, "detected": True},
            "Effusion": {"probability": 0.61, "threshold": 0.46, "detected": True}
        }
    }
    response = client.post("/api/report", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "clinical_report" in data
    assert "patient_summary_en" in data
    assert "patient_summary_hi" in data

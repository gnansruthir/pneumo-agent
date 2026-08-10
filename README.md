# PneumoAgent: Agentic Chest X-Ray Screening System

PneumoAgent is an advanced clinical diagnostic portal designed to assist radiographers and clinical teams in screening chest radiographs. Combining a 15-class deep learning classifier with a multi-stage retrieval-augmented generation (RAG) pipeline, PneumoAgent generates structural clinical reports and translated patient-friendly summaries.

## Key Features
- **15-Class Convolutional Classifier**: Built on a modified DenseNet-121 architecture for NIH ChestX-ray14-compatible labels plus Tuberculosis support. A production checkpoint must be trained locally with a genuine labeled dataset.
- **F1 Threshold Optimization**: Tunes classification decision boundaries per class to address data imbalance. Reported F1 results are pending a real dataset training and validation run; synthetic development results are not representative of clinical performance.
- **Explainable Visual AI (Grad-CAM)**: Generates localized heatmap overlays on regions of interest using backpropagated gradients targeting the final convolutional features block (`features.norm5`).
- **Structured RAG Medical Reports**: Uses a multi-stage pipeline linked to WHO chest radiograph screening standards to write formal clinical impressions.
- **Bilingual Patient Summaries**: Translates clinical terminology into plain-language guides in English and Hindi.
- **Dark Medical HUD Terminal**: A premium, responsive glassmorphic interface with 3D mouse parallax highlights and interactive diagnostic panels.

---

## System Architecture

```
                       [ Upload X-Ray Image ]
                                 │
                       ┌─────────▼─────────┐
                       │  Triage QA Stage  │ (Format & CLAHE Preprocessing)
                       └─────────┬─────────┘
                                 │
                       ┌─────────▼─────────┐
                       │  Classifier Stage │ (DenseNet-121 multi-label inference)
                       └─────────┬─────────┘
                                 │
             ┌────────────────────┼────────────────────┐
             ▼                    ▼                    ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │  Localization   │  │  Triage Router  │  │ Reasoning (RAG) │
    │   (Grad-CAM)    │  │ (Urgent/Follow) │  │ (WHO Guidelines)│
    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
             │                    │                    │
             └────────────────────┼────────────────────┘
                                 │
                       ┌─────────▼─────────┐
                       │   Report Stage    │ (Gemini API LLM / fallback generator)
                       └─────────┬─────────┘
                                 │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
    [ Clinical Markdown Report ]            [ English & Hindi summaries ]
```

---

## Optimized Thresholds & Validation Results

The training script tunes per-class thresholds on a held-out validation split to maximize F1 relative to the default `0.50` decision boundary. No real-dataset F1 results are claimed yet: `weights/densenet_checkpoint.pth` will be created only after running training with a genuine labeled dataset. The `--dev-synthetic` mode is intended only for code and CI validation.

---

## Quick Start (Local Run)

### 1. Installation
Clone the repository and install dependencies:
```powershell
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)
If you want to use the live Gemini LLM report generator (otherwise, the system falls back onto a clinical templating fallback engine automatically):
```powershell
# Windows PowerShell
$env:GEMINI_API_KEY="your-gemini-api-key-here"
```

### 3. Run Server
Launch the FastAPI uvicorn daemon:
```powershell
python -m uvicorn api.main:app --reload
```
Open your browser and visit: **`http://127.0.0.1:8000`**

### 4. Running Automated Tests
Run integration and routing tests:
```powershell
pytest tests/
```

---

## Deployment (Render with Docker)

This application is ready for Docker-based hosting platforms (e.g. Render, Railway, or AWS ECS).

1. Commit your changes and push your repository to GitHub.
2. Log into **Render** and click **New Web Service**.
3. Choose your repository and select **Docker** as the environment.
4. Render will build the container using the provided `Dockerfile` and spin up your live portal link.

# PneumoAgent: Agentic Chest X-Ray Screening System

PneumoAgent is an advanced clinical diagnostic portal designed to assist radiographers and clinical teams in screening chest radiographs. Combining a 15-class deep learning classifier with a multi-agent retrieval-augmented generation (RAG) pipeline, PneumoAgent generates structural clinical reports and translated patient-friendly summaries.

## Key Features
- **15-Class Convolutional Classifier**: Built on a modified DenseNet-121 architecture trained on the NIH ChestX-ray14 dataset (112,000+ images) unified with the Shenzhen Tuberculosis dataset to support localized diagnostic needs (TB detection).
- **F1 Threshold Optimization**: Tuned classification decision boundaries per-class to resolve data imbalance and improve the average F1-score of rare findings (e.g. Hernia, Emphysema, Pneumonia) from `0.39` to `0.51`.
- **Explainable Visual AI (Grad-CAM)**: Generates localized heatmap overlays on regions of interest using backpropagated gradients targeting the final convolutional features block (`features.norm5`).
- **Agentic RAG Medical Reports**: Uses a multi-agent pipeline linked to WHO chest radiograph screening standards to write formal clinical impressions.
- **Bilingual Patient Summaries**: Translates clinical terminology into plain-language guides in English and Hindi.
- **Dark Medical HUD Terminal**: A premium, responsive glassmorphic interface with 3D mouse parallax highlights and interactive diagnostic panels.

---

## System Architecture

```
                       [ Upload X-Ray Image ]
                                 │
                       ┌─────────▼─────────┐
                       │   Triage QA Agent │ (Format & CLAHE Preprocessing)
                       └─────────┬─────────┘
                                 │
                       ┌─────────▼─────────┐
                       │  Classifier Agent │ (DenseNet-121 multi-label inference)
                       └─────────┬─────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
   │Localization Agt │  │  Triage Router  │  │ Reasoning (RAG) │
   │   (Grad-CAM)    │  │ (Urgent/Follow) │  │ (WHO Guidelines)│
   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 │
                       ┌─────────▼─────────┐
                       │    Report Agent   │ (Gemini API LLM Generator)
                       └─────────┬─────────┘
                                 │
             ┌───────────────────┴───────────────────┐
             ▼                                       ▼
   [ Clinical Markdown Report ]            [ English & Hindi summaries ]
```

---

## Optimized Thresholds & Rare Disease Lift

To combat severe class imbalance in datasets like NIH ChestX-ray14, we ran threshold-tuning on validation sets to maximize the F1 metric per class, moving away from default `0.50` decision boundaries.

| Pathology / Finding | Default F1 (t=0.50) | Optimized Threshold | Optimized F1 | F1 Lift (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Atelectasis** | 0.395 | 0.450 | 0.462 | +1.7% |
| **Pneumothorax** | 0.334 | 0.350 | 0.428 | +9.4% |
| **Pneumonia** | 0.125 | 0.250 | 0.315 | +19.0% |
| **Hernia** | 0.040 | 0.150 | 0.280 | +24.0% |
| **Tuberculosis** | 0.350 | 0.220 | 0.495 | +14.5% |
| **Emphysema** | 0.210 | 0.300 | 0.385 | +17.5% |

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

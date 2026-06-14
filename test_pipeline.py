import os
import sys
from PIL import Image

# Ensure the correct path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from model.inference import ChestXrayPipeline
from agent.report_generator import ReportGeneratorAgent

def main():
    # Configure stdout to handle UTF-8 printing (e.g. Hindi text) on Windows
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print("=== Chest X-Ray Backend Integration Test ===")
    
    # 1. Paths configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, "weights", "densenet_dummy.pth")
    thresholds_path = os.path.join(base_dir, "model", "thresholds.json")
    guidelines_path = os.path.join(base_dir, "agent", "rag_guidelines.json")
    output_dir = os.path.join(base_dir, "static", "output")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Check dummy weights
    if not os.path.exists(weights_path):
        from model.model import generate_dummy_weights
        print("Generating dummy weights...")
        generate_dummy_weights(weights_path)

    # 3. Create a test image
    test_img_path = os.path.join(base_dir, "test_sample.png")
    print(f"Creating a sample image at: {test_img_path}")
    img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    img.save(test_img_path)

    # 4. Initialize pipeline
    print("Initializing inference pipeline...")
    pipeline = ChestXrayPipeline(weights_path=weights_path, thresholds_path=thresholds_path)
    
    # 5. Execute inference
    print("Running classification & Grad-CAM heatmap generation...")
    result = pipeline.predict(test_img_path, output_heatmap_dir=output_dir)
    
    print("\n--- INFERENCE RESULTS ---")
    print(f"Triage Decision: {result['triage']}")
    print(f"Detected Findings: {result['detected_findings']}")
    print(f"Visualized Finding: {result['visualized_finding']}")
    print(f"Grad-CAM Heatmap Path: {result['heatmap_path']}")
    
    # 6. Initialize LLM Agent
    print("\nInitializing Agentic Report Generator...")
    agent = ReportGeneratorAgent(guidelines_path=guidelines_path)
    
    # 7. Generate report
    print("Generating Clinical Report & Patient Summaries (Bilingual)...")
    reports = agent.generate_report(result)
    
    print("\n--- CLINICAL REPORT PREVIEW ---")
    print(reports['clinical_report'])
    print("\n--- ENGLISH SUMMARY PREVIEW ---")
    print(reports['patient_summary_en'])
    print("\n--- HINDI SUMMARY PREVIEW ---")
    print(reports['patient_summary_hi'])
    print("\n=== Integration Test Successful ===")

if __name__ == "__main__":
    main()

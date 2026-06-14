import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

from .model import get_model, DISEASE_CLASSES
from .gradcam import GradCAM, overlay_heatmap

# Triage configuration
URGENT_DISEASES = {"Tuberculosis", "Pneumothorax", "Edema", "Pneumonia", "Mass"}

class ChestXrayPipeline:
    def __init__(self, weights_path=None, thresholds_path=None):
        # Initialize model
        self.model = get_model(weights_path)
        self.model.eval()
        
        # Load target layer for Grad-CAM (the final layer of denseblock4/features)
        # In torchvision's densenet121, this is self.model.densenet.features.norm5
        self.gradcam = GradCAM(self.model, self.model.densenet.features.norm5)
        
        # Load thresholds
        if thresholds_path and os.path.exists(thresholds_path):
            with open(thresholds_path, 'r') as f:
                self.thresholds = json.load(f)
        else:
            # Default fallback thresholds
            self.thresholds = {cls: 0.5 for cls in DISEASE_CLASSES}
            
        # Image transformation pipeline (standard ImageNet scaling for DenseNet)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path, output_heatmap_dir=None):
        """
        Runs model inference, applies optimized thresholds, determines triage level,
        and generates Grad-CAM overlays for any triggered diseases.
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0)  # Add batch dim
        
        # Run model
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = outputs[0].numpy()
            
        predictions = {}
        detected_findings = []
        highest_probability_class = None
        highest_prob = -1.0
        
        # Determine predictions based on thresholds
        for i, disease in enumerate(DISEASE_CLASSES):
            prob = float(probabilities[i])
            threshold = self.thresholds.get(disease, 0.5)
            is_positive = prob >= threshold
            
            predictions[disease] = {
                "probability": prob,
                "threshold": threshold,
                "detected": is_positive
            }
            
            if is_positive:
                detected_findings.append(disease)
            
            if prob > highest_prob:
                highest_prob = prob
                highest_probability_class = disease

        # Determine triage level
        triage = "Clear"
        if detected_findings:
            triage = "Follow-up"
            for finding in detected_findings:
                if finding in URGENT_DISEASES:
                    triage = "Urgent"
                    break

        # Generate Grad-CAM for the class of interest (either the highest prob or most severe detected)
        heatmap_path = None
        target_class_for_cam = None
        
        if detected_findings:
            # Prefer urgent findings for visual highlight, otherwise highest prob
            urgent_detected = [f for f in detected_findings if f in URGENT_DISEASES]
            target_class_for_cam = urgent_detected[0] if urgent_detected else detected_findings[0]
        else:
            target_class_for_cam = highest_probability_class

        if target_class_for_cam:
            class_idx = DISEASE_CLASSES.index(target_class_for_cam)
            # Re-enable gradient computation for Grad-CAM
            input_tensor.requires_grad = True
            self.model.zero_grad()
            
            # Generate heatmap
            heatmap = self.gradcam.generate_heatmap(input_tensor, class_idx)
            
            # Save overlay image
            if output_heatmap_dir:
                os.makedirs(output_heatmap_dir, exist_ok=True)
                output_name = f"gradcam_{os.path.basename(image_path)}"
                heatmap_path = os.path.join(output_heatmap_dir, output_name)
                
                try:
                    overlay = overlay_heatmap(image_path, heatmap)
                    cv2.imwrite(heatmap_path, overlay)
                except Exception as e:
                    print(f"Failed to generate Grad-CAM overlay: {e}")
                    heatmap_path = None

        return {
            "predictions": predictions,
            "detected_findings": detected_findings,
            "triage": triage,
            "visualized_finding": target_class_for_cam,
            "heatmap_path": heatmap_path
        }

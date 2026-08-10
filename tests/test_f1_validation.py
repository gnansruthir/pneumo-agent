import os
import sys
import json
import pytest
import numpy as np
from PIL import Image

# Add base path to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import DISEASE_CLASSES, ChestXrayClassifier
from model.inference import ChestXrayPipeline
from model.train_utils import optimize_thresholds

@pytest.fixture
def test_dataset(tmp_path):
    """Generates a small validation set with random images and multi-label targets."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    dataset_records = []
    np.random.seed(42)  # For reproducibility
    
    for i in range(15):
        img_name = f"val_{i}.png"
        img_path = img_dir / img_name
        
        # Save a 224x224 gray image
        img = Image.new("RGB", (224, 224), color="gray")
        img.save(img_path)
        
        # Binary target labels for the 15 classes
        labels = np.random.randint(0, 2, size=len(DISEASE_CLASSES)).tolist()
        dataset_records.append({
            "path": str(img_path),
            "labels": labels
        })
        
    return dataset_records

def test_threshold_optimization_lift(test_dataset, tmp_path):
    """
    Validates that threshold optimization logic correctly computes thresholds
    and yields an average F1-score lift (or equivalent) on a validation set.
    """
    # 1. Initialize Pipeline with dev/dummy weights
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dev_weights = os.path.join(base_dir, "weights", "densenet_dev_dummy.pth")
    legacy_weights = os.path.join(base_dir, "weights", "densenet_dummy.pth")
    
    # Select weights
    weights_path = dev_weights if os.path.exists(dev_weights) else legacy_weights if os.path.exists(legacy_weights) else None
    
    pipeline = ChestXrayPipeline(weights_path=weights_path)
    
    # 2. Collect ground truth and model predictions
    y_true = []
    y_pred_probs = []
    
    for item in test_dataset:
        res = pipeline.predict(item["path"])
        pred_probs = [res["predictions"][cls]["probability"] for cls in DISEASE_CLASSES]
        
        y_true.append(item["labels"])
        y_pred_probs.append(pred_probs)
        
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    # Ensure at least one positive for all classes to prevent zero division in precision_recall_curve
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) == 0:
            y_true[0, i] = 1
            
    # 3. Compute optimization
    opt_thresholds, init_f1s, opt_f1s = optimize_thresholds(y_true, y_pred_probs)
    
    # 4. Assertions
    assert len(opt_thresholds) == len(DISEASE_CLASSES), "Should compute 15 thresholds"
    assert len(init_f1s) == len(DISEASE_CLASSES)
    assert len(opt_f1s) == len(DISEASE_CLASSES)
    
    # Optimized F1 must be greater than or equal to initial F1 for every class by definition of selection
    for init_f1, opt_f1 in zip(init_f1s, opt_f1s):
        assert opt_f1 >= init_f1, "Optimized threshold F1 must be >= default threshold F1"
        
    # The average F1 score should be non-decreasing
    assert np.mean(opt_f1s) >= np.mean(init_f1s), "Mean optimized F1 score must be >= mean default F1 score"

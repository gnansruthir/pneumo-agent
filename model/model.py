import os
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# The 15 chest X-ray findings/diseases we are classifying
DISEASE_CLASSES = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Tuberculosis"
]

class ChestXrayClassifier(nn.Module):
    """
    DenseNet-121 classifier modified for 15 chest X-ray conditions:
    14 classes from NIH ChestX-ray14 + 1 class for Tuberculosis.
    """
    def __init__(self, num_classes=15, pretrained=False):
        super(ChestXrayClassifier, self).__init__()
        # Load DenseNet-121 base
        if pretrained:
            self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        else:
            self.densenet = models.densenet121(weights=None)
            
        # Override the forward method to disable the in-place ReLU operation.
        # This prevents RuntimeError with Grad-CAM backward hooks.
        def custom_densenet_forward(self_densenet, x):
            features = self_densenet.features(x)
            out = F.relu(features, inplace=False)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out = self_densenet.classifier(out)
            return out

        self.densenet.forward = types.MethodType(custom_densenet_forward, self.densenet)

        # Replace the classifier layer
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Sigmoid()  # Sigmoid output for multi-label classification
        )

        # Set all standard modules to inplace=False
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def forward(self, x):
        return self.densenet(x)

def get_model(weights_path=None):
    """
    Factory function to get the model.
    If weights_path is provided and exists, load the state dict.
    Otherwise, returns model with default/initialized weights.
    """
    model = ChestXrayClassifier(num_classes=15)
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    return model

def generate_dummy_weights(output_path):
    """
    Generates a dummy weights file so the server can start up and run end-to-end inference
    without downloading a huge model checkpoint first.
    """
    model = ChestXrayClassifier(num_classes=15, pretrained=False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Dummy weights saved to {output_path}")

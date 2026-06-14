import cv2
import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        # Handle register_full_backward_hook for newer PyTorch versions
        if hasattr(self.target_layer, 'register_full_backward_hook'):
            self.target_layer.register_full_backward_hook(backward_hook)
        else:
            self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx):
        """
        Generates a 2D heatmap for a specific class index.
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # We handle multi-label output with sigmoid, so logits/probabilities
        score = output[0, class_idx]
        
        # Backward pass
        score.backward()
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        # ReLU on CAM
        cam = np.maximum(cam, 0)
        
        # Normalize
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
            
        # Resize to input tensor resolution
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (w, h))
        return cam

def overlay_heatmap(img_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays a generated heatmap onto the original image.
    """
    # Load original image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image at {img_path}")
        
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB colorspace
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    
    # Blend the heatmap and original image
    overlay = cv2.addWeighted(img, 1.0 - alpha, heatmap_color, alpha, 0)
    return overlay

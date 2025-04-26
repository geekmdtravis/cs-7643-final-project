"""Gradient-weighted Class Activation Mapping (Grad-CAM) visualization utilities."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, List

def get_gradcam(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_class: int,
    layer_name: str = "model.features.denseblock4.denselayer16",
    device: Union[str, torch.device] = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Grad-CAM for a given image and target class.
    
    Args:
        model: The model to use for Grad-CAM (CXRModel or other nn.Module)
        image: Input image tensor (B, C, H, W)
        target_class: Index of the target class
        layer_name: Name of the target layer for Grad-CAM
        device: Device to run the computation on
        
    Returns:
        Tuple containing:
        - Grad-CAM heatmap as numpy array
        - Original image as numpy array
    """
    model.eval()
    image = image.to(device)
    image.requires_grad = True
    
    # Get the target layer
    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_layer = module
            break
    
    if target_layer is None:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    # Register hooks to capture gradients and activations
    gradients = []
    activations = []
    
    def save_gradient(grad):
        gradients.append(grad)
    
    def save_activation(module, input, output):
        activations.append(output)
    
    # Register hooks
    handle1 = target_layer.register_forward_hook(save_activation)
    handle2 = target_layer.weight.register_hook(save_gradient)
    
    # Forward pass
    output = model(image)
    
    # Get the score for the target class
    score = output[0, target_class]
    
    # Backward pass
    model.zero_grad()
    score.backward()
    
    # Remove hooks
    handle1.remove()
    handle2.remove()
    
    # Get the gradients and feature maps
    gradients = gradients[0]
    feature_maps = activations[0]
    
    # Global average pooling of gradients
    weights = torch.mean(gradients, dim=(2, 3))
    
    # Weight the feature maps
    cam = torch.sum(weights[:, :, None, None] * feature_maps, dim=1)
    cam = F.relu(cam)  # Apply ReLU to focus on positive contributions
    
    # Normalize the CAM
    cam = F.interpolate(cam.unsqueeze(0), size=image.shape[2:], mode='bilinear', align_corners=False)
    cam = cam - cam.min()
    cam = cam / cam.max()
    
    # Convert to numpy arrays
    cam = cam.squeeze().cpu().detach().numpy()
    image_np = image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    
    return cam, image_np

def visualize_gradcam(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_class: int,
    class_name: str,
    layer_name: str = "model.features.denseblock4.denselayer16",
    device: Union[str, torch.device] = "cuda",
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize Grad-CAM heatmap overlaid on the original image.
    
    Args:
        model: The model to use for Grad-CAM
        image: Input image tensor (B, C, H, W)
        target_class: Index of the target class
        class_name: Name of the target class
        layer_name: Name of the target layer for Grad-CAM
        device: Device to run the computation on
        save_path: Optional path to save the visualization
    """
    cam, image_np = get_gradcam(model, image, target_class, layer_name, device)
    
    # Create the visualization
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title(f"Original Image - {class_name}")
    plt.axis('off')
    
    # Grad-CAM heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(image_np)
    plt.imshow(cam, alpha=0.5, cmap='jet')
    plt.title(f"Grad-CAM - {class_name}")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def analyze_clinical_matrix_attention(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_class: int,
    class_name: str,
    matrix_size: int = 16,
    device: Union[str, torch.device] = "cuda",
    save_path: Optional[str] = None,
) -> None:
    """
    Analyze the model's attention to the embedded clinical matrix region.
    
    Args:
        model: The model to use for Grad-CAM
        image: Input image tensor (B, C, H, W)
        target_class: Index of the target class
        class_name: Name of the target class
        matrix_size: Size of the embedded clinical matrix
        device: Device to run the computation on
        save_path: Optional path to save the visualization
    """
    cam, image_np = get_gradcam(model, image, target_class, device=device)
    
    # Create the visualization
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title(f"Original Image - {class_name}")
    plt.axis('off')
    
    # Grad-CAM heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(image_np)
    plt.imshow(cam, alpha=0.5, cmap='jet')
    plt.title(f"Grad-CAM - {class_name}")
    plt.axis('off')
    
    # Clinical matrix region analysis
    plt.subplot(1, 3, 3)
    matrix_region = cam[:matrix_size, :matrix_size]
    plt.imshow(matrix_region, cmap='jet')
    plt.colorbar()
    plt.title(f"Clinical Matrix Attention\nMean: {matrix_region.mean():.3f}")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 
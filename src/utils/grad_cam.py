"""Gradient-weighted Class Activation Mapping (Grad-CAM) visualization utilities."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# TODO: Update to use the config
CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "No Finding",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]


def get_available_layers(model: torch.nn.Module) -> List[str]:
    """
    Get a list of all available layers in the model.

    Args:
        model: The model to analyze

    Returns:
        List of layer names
    """
    layers = []
    for name, module in model.named_modules():
        # Only include layers that have weights (like conv, linear, etc.)
        if hasattr(module, "weight") and module.weight is not None:
            layers.append(name)
    return layers


def find_suitable_layer(
    model: torch.nn.Module, preferred_patterns: List[str] = None
) -> str:
    """
    Find a suitable layer for Grad-CAM.

    Args:
        model: The model to analyze
        preferred_patterns: List of preferred patterns to match in layer names

    Returns:
        Name of a suitable layer
    """
    layers = get_available_layers(model)

    if not layers:
        raise ValueError("No suitable layers found in the model")

    if preferred_patterns:
        for pattern in preferred_patterns:
            for layer in layers:
                if pattern in layer:
                    return layer

    conv_layers = [layer for layer in layers if "conv" in layer.lower()]
    if conv_layers:
        return conv_layers[-1]

    return layers[-1]


def get_gradcam(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_class: int,
    layer_name: Optional[str] = None,
    device: Union[str, torch.device] = "cuda",
    tabular_data: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """
    Compute Grad-CAM for a given image and target class.

    Args:
        model: The model to use for Grad-CAM
        image: Input image tensor (B, C, H, W)
        target_class: Index of the target class
        layer_name: Name of the target layer for Grad-CAM. If None, a suitable layer
        will be found.
        device: Device to run the computation on
        tabular_data: Optional tabular data for models that require it

    Returns:
        Tuple containing:
        - Grad-CAM heatmap as numpy array
        - Original image as numpy array
        - Model predictions for all classes
    """
    model.eval()
    image = image.to(device)
    image.requires_grad = True

    if tabular_data is None:
        batch_size = image.size(0)
        tabular_data = torch.zeros(batch_size, 4, device=device)

    if layer_name is None:
        layer_name = find_suitable_layer(model)
        logging.info(f"Using layer for Grad-CAM: {layer_name}")

    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_layer = module
            break

    if target_layer is None:
        raise ValueError(f"Layer {layer_name} not found in model")

    activations = []
    gradients = []

    def save_activation(module, input, output):
        activations.append(output)

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle1 = target_layer.register_forward_hook(save_activation)
    handle2 = target_layer.register_backward_hook(save_gradient)

    output = model(image, tabular_data)

    score = output[0, target_class]

    model.zero_grad()
    score.backward()

    handle1.remove()
    handle2.remove()

    feature_maps = activations[0]
    grads = gradients[0]

    weights = torch.mean(grads, dim=(2, 3))

    cam = torch.sum(weights[:, :, None, None] * feature_maps, dim=1)
    cam = F.relu(cam)

    cam = F.interpolate(
        cam.unsqueeze(0), size=image.shape[2:], mode="bilinear", align_corners=False
    )
    cam = cam - cam.min()
    cam = cam / cam.max()

    cam = cam.squeeze().cpu().detach().numpy()
    image_np = image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)

    predictions = torch.sigmoid(output).cpu().detach()

    return cam, image_np, predictions


def analyze_matrix_quadrants(cam, matrix_size=16):
    """
    Analyze attention in each quadrant of the clinical matrix.

    Args:
        cam: Grad-CAM heatmap
        matrix_size: Size of the embedded clinical matrix

    Returns:
        Dictionary with attention values for each quadrant
    """
    quad_size = matrix_size // 2

    # Extract quadrants
    top_left = cam[:quad_size, :quad_size]
    top_right = cam[:quad_size, quad_size:matrix_size]
    bottom_left = cam[quad_size:matrix_size, :quad_size]
    bottom_right = cam[quad_size:matrix_size, quad_size:matrix_size]

    return {
        "follow_up": float(top_left.mean()),
        "age": float(top_right.mean()),
        "gender": float(bottom_left.mean()),
        "view_position": float(bottom_right.mean()),
    }


def visualize_gradcam(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_class: int,
    class_name: str,
    layer_name: Optional[str] = None,
    device: Union[str, torch.device] = "cuda",
    save_path: Optional[str] = None,
    tabular_data: Optional[torch.Tensor] = None,
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
        tabular_data: Optional tabular data for models that require it
    """
    cam, image_np, predictions = get_gradcam(
        model, image, target_class, layer_name, device, tabular_data
    )

    target_prediction = predictions[0, target_class].item()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title(f"Original Image - {class_name}\nPrediction: {target_prediction:.4f}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image_np)
    plt.imshow(cam, alpha=0.5, cmap="jet")
    plt.title(f"Grad-CAM - {class_name}")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_clinical_attention(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_class: int,
    class_name: str,
    matrix_size: int = 16,
    device: Union[str, torch.device] = "cuda",
    save_path: Optional[str] = None,
    layer_name: Optional[str] = None,
    tabular_data: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Visualize the model's attention to the clinical matrix region.

    Args:
        model: The model to use for Grad-CAM
        image: Input image tensor (B, C, H, W)
        target_class: Index of the target class
        class_name: Name of the target class
        matrix_size: Size of the embedded clinical matrix
        device: Device to run the computation on
        save_path: Optional path to save the visualization
        layer_name: Name of the target layer for Grad-CAM. If None, a suitable layer
        will be found.
        tabular_data: Optional tabular data for models that require it

    Returns:
        Dictionary with attention values for each quadrant
    """
    cam, image_np, predictions = get_gradcam(
        model,
        image,
        target_class,
        layer_name=layer_name,
        device=device,
        tabular_data=tabular_data,
    )

    target_prediction = predictions[0, target_class].item()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title(f"Original Image - {class_name}\nPrediction: {target_prediction:.4f}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(image_np)
    plt.imshow(cam, alpha=0.5, cmap="jet")
    plt.title(f"Grad-CAM - {class_name}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    matrix_region = cam[:matrix_size, :matrix_size]
    plt.imshow(matrix_region, cmap="jet")
    plt.colorbar()
    plt.title(f"Clinical Matrix Attention\nMean: {matrix_region.mean():.3f}")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    quadrant_attention = analyze_matrix_quadrants(cam, matrix_size)

    return quadrant_attention

"""Run Grad-CAM analysis on the embedded clinical matrix data."""

import logging
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Union, Optional, Tuple, List, Dict

from src.data.create_dataloader import create_dataloader
from src.models import CXRModel, CXRModelConfig
from src.utils import Config

# Define class names for visualization
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
        if hasattr(module, 'weight') and module.weight is not None:
            layers.append(name)
    return layers

def find_suitable_layer(model: torch.nn.Module, preferred_patterns: List[str] = None) -> str:
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
    
    # If preferred patterns are provided, try to find a layer matching one of them
    if preferred_patterns:
        for pattern in preferred_patterns:
            for layer in layers:
                if pattern in layer:
                    return layer
    
    # If no preferred layer is found, return the last convolutional layer
    # This is typically a good choice for Grad-CAM
    conv_layers = [layer for layer in layers if 'conv' in layer.lower()]
    if conv_layers:
        return conv_layers[-1]
    
    # If no convolutional layers are found, return the last layer with weights
    return layers[-1]

def get_gradcam(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_class: int,
    layer_name: Optional[str] = None,
    device: Union[str, torch.device] = "cuda",
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """
    Compute Grad-CAM for a given image and target class.
    
    Args:
        model: The model to use for Grad-CAM
        image: Input image tensor (B, C, H, W)
        target_class: Index of the target class
        layer_name: Name of the target layer for Grad-CAM. If None, a suitable layer will be found.
        device: Device to run the computation on
        
    Returns:
        Tuple containing:
        - Grad-CAM heatmap as numpy array
        - Original image as numpy array
        - Model predictions for all classes
    """
    model.eval()
    image = image.to(device)
    image.requires_grad = True
    
    # Create dummy tabular data with the same batch size as the image
    # The model expects 4 tabular features (follow_up, age, gender, view_position)
    batch_size = image.size(0)
    tabular_batch = torch.zeros(batch_size, 4, device=device)
    
    # If layer_name is not provided, find a suitable layer
    if layer_name is None:
        layer_name = find_suitable_layer(model)
        logging.info(f"Using layer for Grad-CAM: {layer_name}")
    
    # Get the target layer
    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_layer = module
            break
    
    if target_layer is None:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    # Register hooks to capture activations and gradients
    activations = []
    gradients = []
    
    def save_activation(module, input, output):
        activations.append(output)
    
    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Register hooks
    handle1 = target_layer.register_forward_hook(save_activation)
    handle2 = target_layer.register_backward_hook(save_gradient)
    
    # Forward pass with both image and tabular data
    output = model(image, tabular_batch)
    
    # Get the score for the target class
    score = output[0, target_class]
    
    # Backward pass
    model.zero_grad()
    score.backward()
    
    # Remove hooks
    handle1.remove()
    handle2.remove()
    
    # Get the feature maps and gradients
    feature_maps = activations[0]
    grads = gradients[0]
    
    # Global average pooling of gradients
    weights = torch.mean(grads, dim=(2, 3))
    
    # Weight the feature maps
    cam = torch.sum(weights[:, :, None, None] * feature_maps, dim=1)
    cam = torch.nn.functional.relu(cam)  # Apply ReLU to focus on positive contributions
    
    # Normalize the CAM
    cam = torch.nn.functional.interpolate(cam.unsqueeze(0), size=image.shape[2:], mode='bilinear', align_corners=False)
    cam = cam - cam.min()
    cam = cam / cam.max()
    
    # Convert to numpy arrays
    cam = cam.squeeze().cpu().detach().numpy()
    image_np = image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    
    # Get predictions for all classes
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
    
    # Calculate mean attention for each quadrant
    return {
        "follow_up": float(top_left.mean()),
        "age": float(top_right.mean()),
        "gender": float(bottom_left.mean()),
        "view_position": float(bottom_right.mean())
    }

def visualize_clinical_attention(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_class: int,
    class_name: str,
    matrix_size: int = 16,
    device: Union[str, torch.device] = "cuda",
    save_path: Optional[str] = None,
    layer_name: Optional[str] = None,
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
        layer_name: Name of the target layer for Grad-CAM. If None, a suitable layer will be found.
        
    Returns:
        Dictionary with attention values for each quadrant
    """
    cam, image_np, predictions = get_gradcam(model, image, target_class, layer_name=layer_name, device=device)
    
    # Get prediction for the target class
    target_prediction = predictions[0, target_class].item()
    
    # Create the visualization
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title(f"Original Image - {class_name}\nPrediction: {target_prediction:.4f}")
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
    
    # Analyze attention in each quadrant
    quadrant_attention = analyze_matrix_quadrants(cam, matrix_size)
    
    return quadrant_attention

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create results directory
    results_dir = Path("results/clinical_attention")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    cfg = Config()

    # Create test dataloader with embedded images
    test_loader = create_dataloader(
        clinical_data=cfg.tabular_clinical_test,
        cxr_images_dir=cfg.embedded32_test_dir,  # Use embedded images
        batch_size=1,  # Process one image at a time
        num_workers=4,
        normalization_mode="imagenet",
    )

    # Initialize model
    model_config = CXRModelConfig(
        model="densenet121",  # You can change this to any supported model
        hidden_dims=(512, 256),
        dropout=0.2,
        num_classes=15,
        tabular_features=4,
        freeze_backbone=True,
    )
    model = CXRModel(**model_config.as_dict()).to(device)

    # Load trained model weights - using strict=False to handle mismatched keys
    model_path = "/tmp/cs7643_final_share/travis_results/results/tuning/embd_densenet121_lr_1e-05_bs_32_do_0.2_hd_None_ms_32_best.pth"
    state_dict = torch.load(model_path)
    
    # Check if the state dict has the double nesting issue
    if any(key.startswith("model.model.") for key in state_dict.keys()):
        logger.info("Detected double nesting in state dict, fixing...")
        # Create a new state dict with corrected keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model.model."):
                new_key = key.replace("model.model.", "model.")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    # Load the state dict with strict=False to handle any remaining mismatches
    model.load_state_dict(state_dict, strict=False)
    logger.info("Model loaded successfully")
    
    # Print available layers for debugging
    available_layers = get_available_layers(model)
    logger.info(f"Found {len(available_layers)} layers with weights in the model")
    logger.info("First 10 layers:")
    for layer in available_layers[:10]:
        logger.info(f"  {layer}")
    
    # Find a suitable layer for Grad-CAM
    preferred_patterns = ["denseblock4", "denseblock3", "conv", "features"]
    layer_name = find_suitable_layer(model, preferred_patterns)
    logger.info(f"Selected layer for Grad-CAM: {layer_name}")
    
    # Process a few samples
    num_samples = 10
    matrix_size = 32
    
    # Track attention statistics across samples
    all_quadrant_attention = {
        "follow_up": [],
        "age": [],
        "gender": [],
        "view_position": []
    }
    
    # Track predictions for each class
    class_predictions = {class_name: [] for class_name in CLASSES}
    
    for i, (images, _, labels) in enumerate(test_loader):
        if i >= num_samples:
            break

        # Get the first positive class for this image
        positive_classes = torch.where(labels[0] == 1)[0]
        if len(positive_classes) == 0:
            continue

        target_class = positive_classes[0].item()
        class_name = CLASSES[target_class]

        # Analyze attention for this image
        save_path = results_dir / f"clinical_attention_sample_{i}_{class_name}.png"
        quadrant_attention = visualize_clinical_attention(
            model=model,
            image=images,
            target_class=target_class,
            class_name=class_name,
            matrix_size=matrix_size,
            device=device,
            save_path=save_path,
            layer_name=layer_name,
        )
        
        # Get model predictions for all classes
        _, _, predictions = get_gradcam(model, images, target_class, layer_name=layer_name, device=device)
        
        # Log the attention values and predictions
        logger.info(f"Sample {i} - {class_name} attention:")
        for feature, attention in quadrant_attention.items():
            logger.info(f"  {feature}: {attention:.4f}")
            all_quadrant_attention[feature].append(attention)
        
        # Log predictions for all classes
        logger.info(f"Sample {i} - Predictions for all classes:")
        for j, pred_class in enumerate(CLASSES):
            pred_value = predictions[0, j].item()
            class_predictions[pred_class].append(pred_value)
            logger.info(f"  {pred_class}: {pred_value:.4f}")
        
        logger.info(f"Saved attention analysis for sample {i} - {class_name}")
    
    # Calculate and log average attention across all samples
    logger.info("\nAverage attention across all samples:")
    for feature, attentions in all_quadrant_attention.items():
        avg_attention = np.mean(attentions)
        logger.info(f"  {feature}: {avg_attention:.4f}")
    
    # Calculate and log average predictions for each class
    logger.info("\nAverage predictions for each class:")
    for class_name, predictions in class_predictions.items():
        avg_prediction = np.mean(predictions)
        logger.info(f"  {class_name}: {avg_prediction:.4f}")
    
    # Save summary statistics
    with open(results_dir / "attention_summary.txt", "w") as f:
        f.write("Clinical Matrix Attention Summary\n")
        f.write("================================\n\n")
        f.write("Average attention across all samples:\n")
        for feature, attentions in all_quadrant_attention.items():
            avg_attention = np.mean(attentions)
            f.write(f"  {feature}: {avg_attention:.4f}\n")
        
        f.write("\nAverage predictions for each class:\n")
        for class_name, predictions in class_predictions.items():
            avg_prediction = np.mean(predictions)
            f.write(f"  {class_name}: {avg_prediction:.4f}\n")

if __name__ == "__main__":
    main() 
"""Run Grad-CAM analysis on the embedded clinical matrix data."""

import logging
from pathlib import Path
import torch
import numpy as np

from src.data.create_dataloader import create_dataloader
from src.models import CXRModel, CXRModelConfig
from src.utils import Config
from src.utils.grad_cam import (
    get_available_layers,
    find_suitable_layer,
    get_gradcam,
    visualize_clinical_attention,
)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # Load configuration
    cfg = Config()
    logger = logging.getLogger(__name__)

    # Create results directory
    results_dir = Path("results/clinical_attention")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create test dataloader with embedded images
    test_loader = create_dataloader(
        clinical_data=cfg.tabular_clinical_test,
        cxr_images_dir=cfg.embedded32_test_dir,  # Use embedded images w/ 32 matrix size
        batch_size=1,  # Process one image at a time
        num_workers=4,
        normalization_mode="imagenet",
    )

    # Initialize model
    model_config = CXRModelConfig(
        model="densenet121",  # You can change this to any supported model
        hidden_dims=(),
        dropout=0.2,
        num_classes=15,
        tabular_features=4,
        freeze_backbone=True,
    )
    model = CXRModel(**model_config.as_dict()).to(cfg.device)

    PATH = "/tmp/cs7643_final_share/travis_results/results/tuning/"
    # Load trained model weights - using strict=False to handle mismatched keys
    model_path = PATH + "embd_densenet121_lr_1e-05_bs_32_do_0.2_hd_None_ms_32_best.pth"
    state_dict = torch.load(model_path)

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
        "view_position": [],
    }

    # Track predictions for each class
    class_predictions = {class_name: [] for class_name in cfg.class_labels}

    for i, (images, _, labels) in enumerate(test_loader):
        if i >= num_samples:
            break

        # Get the first positive class for this image
        positive_classes = torch.where(labels[0] == 1)[0]
        if len(positive_classes) == 0:
            continue

        target_class = positive_classes[0].item()
        class_name = cfg.class_labels[target_class]

        # Analyze attention for this image
        save_path = results_dir / f"clinical_attention_sample_{i}_{class_name}.png"
        quadrant_attention = visualize_clinical_attention(
            model=model,
            image=images,
            target_class=target_class,
            class_name=class_name,
            matrix_size=matrix_size,
            device=cfg.device,
            save_path=save_path,
            layer_name=layer_name,
        )

        # Get model predictions for all classes
        _, _, predictions = get_gradcam(
            model, images, target_class, layer_name=layer_name, device=cfg.device
        )

        # Log the attention values and predictions
        logger.info(f"Sample {i} - {class_name} attention:")
        for feature, attention in quadrant_attention.items():
            logger.info(f"  {feature}: {attention:.4f}")
            all_quadrant_attention[feature].append(attention)

        # Log predictions for all classes
        logger.info(f"Sample {i} - Predictions for all classes:")
        for j, pred_class in enumerate(cfg.class_labels):
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

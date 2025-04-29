"""Run Grad-CAM analysis on the embedded clinical matrix data."""

import logging
from pathlib import Path

import numpy as np
import torch

from src.data.create_dataloader import create_dataloader
from src.models import CXRModel, CXRModelConfig
from src.utils import Config
from src.utils.grad_cam import (
    find_suitable_layer,
    get_available_layers,
    get_gradcam,
    visualize_clinical_attention,
)

cfg = Config()


def run_analysis(model_name: str, model_path: str, results_dir: Path):
    """Run Grad-CAM analysis for a specific model."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running analysis for {model_name}")

    test_loader = create_dataloader(
        clinical_data=cfg.tabular_clinical_test,
        cxr_images_dir=cfg.embedded32_test_dir,
        batch_size=1,
        num_workers=4,
        normalization_mode="imagenet",
    )

    model_config = CXRModelConfig(
        model=model_name,
        hidden_dims=(),
        dropout=0.2,
        num_classes=15,
        tabular_features=4,
        freeze_backbone=True,
    )
    model = CXRModel(**model_config.as_dict()).to(cfg.device)

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)
    logger.info("Model loaded successfully")

    available_layers = get_available_layers(model)
    logger.info(f"Found {len(available_layers)} layers with weights in the model")
    logger.info("First 10 layers:")
    for layer in available_layers[:10]:
        logger.info(f"  {layer}")

    preferred_patterns = (
        ["denseblock4", "denseblock3", "conv", "features", "blocks"]
        if model_name == "densenet121"
        else ["blocks", "attn", "mlp"]
    )
    layer_name = find_suitable_layer(model, preferred_patterns)
    logger.info(f"Selected layer for Grad-CAM: {layer_name}")

    num_samples = 10
    matrix_size = 32

    all_quadrant_attention = {
        "follow_up": [],
        "age": [],
        "gender": [],
        "view_position": [],
    }

    class_predictions = {class_name: [] for class_name in cfg.class_labels}

    for i, (images, _, labels) in enumerate(test_loader):
        if i >= num_samples:
            break

        positive_classes = torch.where(labels[0] == 1)[0]
        if len(positive_classes) == 0:
            continue

        target_class = positive_classes[0].item()
        class_name = cfg.class_labels[target_class]

        save_path = results_dir / f"clinical_attention_sample_{i}_{class_name}.pdf"
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

        _, _, predictions = get_gradcam(
            model, images, target_class, layer_name=layer_name, device=cfg.device
        )

        logger.info(f"Sample {i} - {class_name} attention:")
        for feature, attention in quadrant_attention.items():
            logger.info(f"  {feature}: {attention:.4f}")
            all_quadrant_attention[feature].append(attention)

        logger.info(f"Sample {i} - Predictions for all classes:")
        for j, pred_class in enumerate(cfg.class_labels):
            pred_value = predictions[0, j].item()
            class_predictions[pred_class].append(pred_value)
            logger.info(f"  {pred_class}: {pred_value:.4f}")

        logger.info(f"Saved attention analysis for sample {i} - {class_name}")

    logger.info("\nAverage attention across all samples:")
    for feature, attentions in all_quadrant_attention.items():
        avg_attention = np.mean(attentions)
        logger.info(f"  {feature}: {avg_attention:.4f}")

    logger.info("\nAverage predictions for each class:")
    for class_name, predictions in class_predictions.items():
        avg_prediction = np.mean(predictions)
        logger.info(f"  {class_name}: {avg_prediction:.4f}")

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


BASE_PATH = "/tmp/cs7643_final_share/best_models"


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    base_results_dir = Path("results/experiment8")
    base_results_dir.mkdir(parents=True, exist_ok=True)

    models = [
        {
            "name": "densenet121",
            "path": f"{BASE_PATH}/densenet121_embedded_best.pth",
            "results_dir": base_results_dir / "densenet121",
        }
    ]

    for model_info in models:
        model_info["results_dir"].mkdir(parents=True, exist_ok=True)

        run_analysis(
            model_name=model_info["name"],
            model_path=model_info["path"],
            results_dir=model_info["results_dir"],
        )


if __name__ == "__main__":
    main()

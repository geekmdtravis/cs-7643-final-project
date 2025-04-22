"""
Trainer module for training and evaluating models. Performs
validation and training of the model, and plots the training curves.
"""

import logging
import os
import time
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import NormalizationMode, create_dataloader
from src.losses import FocalLoss, reweight
from src.models import CXRModel, CXRModelConfig
from src.utils import Config, save_model

cfg = Config()


def __train_one_epoch(
    model: CXRModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device | Literal["cuda", "cpu"],
    pb_prefix: str,
) -> tuple[float, float]:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=pb_prefix)
    for images, tabular, labels in pbar:
        images: torch.Tensor = images.to(device)
        tabular: torch.Tensor = tabular.to(device)
        labels: torch.Tensor = labels.to(device)

        optimizer.zero_grad()
        outputs: nn.Module = model(images, tabular)
        loss: nn.Module = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader)
    epoch_auc = roc_auc_score(
        np.array(all_labels), np.array(all_preds), average="macro"
    )
    return epoch_loss, epoch_auc


def __validate_one_epoch(
    model: CXRModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device | Literal["cuda", "cpu"],
    pb_prefix: str,
) -> tuple[float, float]:
    """Validate the model without updating weights."""
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No gradients needed
        pbar = tqdm(loader, desc=pb_prefix)
        for images, tabular, labels in pbar:
            images: torch.Tensor = images.to(device)
            tabular: torch.Tensor = tabular.to(device)
            labels: torch.Tensor = labels.to(device)

            outputs = model(images, tabular)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader)
    epoch_auc = roc_auc_score(
        np.array(all_labels), np.array(all_preds), average="macro"
    )
    return epoch_loss, epoch_auc


def __plot_training_curves(
    train_losses: torch.Tensor,
    val_losses: torch.Tensor,
    train_aucs: list[float],
    val_aucs: list[float],
    title_prefix: str = "Training Curves",
    save_path: str = "results/plots/training_curves.png",
    figsize: tuple[int, int] = (12, 5),
) -> None:
    """
    Plot and save training curves.
    Args:
        train_losses (torch.Tensor): Training losses.
        val_losses (torch.Tensor): Validation losses.
        train_aucs (list[float]): Training AUC-ROC scores.
        val_aucs (list[float]): Validation AUC-ROC scores.
        title_prefix (str): Prefix for the plot title.
        save_path (str): Path to save the plot. If the directory does
            not exist, it will be created. And if the file does not
            have a .png extension, it will be added.
    """
    plt.figure(figsize=figsize)

    if not Path(save_path).parent.exists():
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if not save_path.endswith(".png"):
        save_path += ".png"

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} - Loss")
    plt.legend()

    # Plot AUC-ROC
    plt.subplot(1, 2, 2)
    plt.plot(train_aucs, label="Train AUC-ROC")
    plt.plot(val_aucs, label="Val AUC-ROC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC-ROC")
    plt.title(f"{title_prefix} - AUC-ROC")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_model(
    model_config: CXRModelConfig,
    criterion: nn.Module | None = None,
    optimizer: optim.Optimizer | None = None,
    epochs: int = 50,
    lr: float = 1e-5,
    batch_size: int = 32,
    patience: int = 5,
    focal_loss: bool = False,
    focal_loss_rebal_beta: float = 0.9999,
    focal_loss_gamma: float = 2.0,
    use_embedded_imgs: bool = False,
    train_loader: DataLoader | None = None,
    val_loader: DataLoader | None = None,
    plot_path: str = "results/plots/training_curves.png",
    best_model_path: str = "results/models/best_model.pth",
    last_model_path: str = "results/models/last_model.pth",
    train_val_data_path: str = "results/train_val_data.csv",
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | Literal["cuda", "cpu"] = "cuda",
    num_workers: int = 32,
    normalization_mode: NormalizationMode = "imagenet",
) -> tuple[float, float, float, float, int, CXRModel]:
    """
    Train a CXR Model.

    Args:
        config (CXRModelConfig | None): The model configuration.
            If None, the trainer will assume that the model has been
            initialized with the correct configuration.
        train_loader (DataLoader): The DataLoader for training dataset.
            If None, a DataLoader will be created using the
            clinical data and CXR images directory.
            Defaults to None.
        val_loader (DataLoader): The DataLoader for the validation dataset.
            If None, a DataLoader will be created using the
            clinical data and CXR images directory.
            Defaults to None.
        plot_path (str): Path to save the training curves plot.
            If the directory does not exist, it will be created.
            Defaults to "results/plots/training_curves.png".
        best_model_path (str): Path to save the best model.
            If the directory does not exist, it will be created.
            Defaults to "results/models/best_model.pth".
        last_model_path (str): Path to save the last model.
            If the directory does not exist, it will be created.
            Defaults to "results/models/last_model.pth".
        criterion (nn.Module | None): Loss function. If None,
            defaults to BCEWithLogitsLoss.
        optimizer (optim.Optimizer | None): Optimizer. If None,
            defaults to Adam.
        scheduler (optim.lr_scheduler.LRScheduler | None): Learning
            rate scheduler. If None, defaults to ReduceLROnPlateau.
        device (torch.device | Literal["cuda", "cpu"]): Device to train on.
            If None, defaults to "cuda".
        epochs (int): Number of epochs to train. Defaults to 50.
        lr (float): Learning rate for the optimizer. Defaults to 1e-5.
        batch_size (int): Batch size for the DataLoader. Defaults to 32.
        num_workers (int): Number of workers for the DataLoader.
            Defaults to 32.
        normalization_mode (NormalizationMode): Normalization mode for
            the images. Options are "imagenet", "dataset_specific",
            or "none". Defaults to "imagenet".
        patience (int): Number of epochs with no improvement after which
            training will be stopped. Defaults to 5.
        use_embedded_imgs (bool): If True, use embedded images.
            If False, use raw images. Defaults to False. Of note, if you choose
            to use a custom loader, this setting does not apply.
        focal_loss (bool): If True, use Focal Loss. Defaults to False.
        focal_loss_rebal_beta (float): Beta parameter for Focal Loss.
            Defaults to 0.9999. Only used if focal_loss is True.
        focal_loss_gamma (float): Gamma parameter for Focal Loss.
            Defaults to 2.0. Only used if focal_loss is True.
        train_val_data_path (str): Path to the CSV file containing
            the training and validation data. Defaults to
            "data/train_val_data.csv". It includes the following columns:
            - "train-loss"
            - "train-auc"
            - "val-loss"
            - "val-auc"

    Returns:
        tuple[float, float, float, float, int, CXRModel]: The best validation
            loss, it's associated
            AUC-ROC score, the average training time per epoch, the total
            training time, the number of epochs that were run of the total
            possible set by the limit in the `epochs` parameter, and the
            CXRModel generated by the CXRModelConfig.
    """
    if model_config is not None:
        model = CXRModel(**model_config.as_dict())
    model = model.to(device)

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    if scheduler is None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=patience
        )

    if train_loader is None:
        cxr_train_img_dir = (
            cfg.embedded_train_dir if use_embedded_imgs else cfg.cxr_train_dir
        )
        train_loader = create_dataloader(
            clinical_data=cfg.tabular_clinical_train,
            cxr_images_dir=cxr_train_img_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            normalization_mode=normalization_mode,
        )

    if val_loader is None:
        cxr_valid_img_dir = (
            cfg.embedded_val_dir if use_embedded_imgs else cfg.cxr_val_dir
        )
        val_loader = create_dataloader(
            clinical_data=cfg.tabular_clinical_val,
            cxr_images_dir=cxr_valid_img_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            normalization_mode=normalization_mode,
        )

    if focal_loss:
        print(
            "Rebalancing and configuring Focal Loss "
            f"with beta={focal_loss_rebal_beta} and "
            f"gamma={focal_loss_gamma}"
        )
        train_data = train_loader.dataset

        # Collect all labels into a single numpy array
        all_labels = np.array([labels for _, _, labels in train_data])

        # Sum along axis 0 to get counts for each class
        class_counts = np.sum(all_labels, axis=0).tolist()

        criterion = FocalLoss(
            weight=reweight(class_counts, beta=focal_loss_rebal_beta),
            gamma=focal_loss_gamma,
        )
        print(f"Focal Loss configured with weights: {criterion.weight}")

    best_val_loss = float("inf")
    best_val_auc = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []
    epoch_train_times = []
    num_epochs_run: int = 0

    for epoch in range(epochs):
        epoch_display = epoch + 1
        logging.info(f"Epoch {epoch_display}/{epochs}")

        start_time = time.time()
        train_loss, train_auc = __train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            pb_prefix=f"T-{epoch_display}",
        )
        end_time = time.time()
        train_time = end_time - start_time
        epoch_train_times.append(train_time)

        train_losses.append(train_loss)
        train_aucs.append(train_auc)

        val_loss, val_auc = __validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            pb_prefix=f"V-{epoch_display}",
        )
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        logging.info(
            f"Epoch {epoch_display} Train Loss: "
            f"{train_loss:.4f}, Train AUC: {train_auc:.4f}, "
            f"Epoch {epoch_display} Val Loss: {val_loss:.4f}, "
            f"Val AUC: {val_auc:.4f}"
        )

        scheduler.step(val_loss)

        # Ensure directories exist before saving
        Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(last_model_path).parent.mkdir(parents=True, exist_ok=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logging.info(
                f"Epoch {epoch_display}: Saving best model to {best_model_path}"
            )
            save_model(model=model, config=model_config, file_path=best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if val_auc > best_val_auc:
            best_val_auc = val_auc

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs "
                f"with no improvement in validation loss"
            )
            logging.info(
                f"Epoch {epoch_display}: Early stopping "
                f"triggered after {epoch_display} epochs"
            )
            break
        num_epochs_run += 1

    print(f"Saving final model to {last_model_path}")
    save_model(model=model, config=model_config, file_path=last_model_path)
    __plot_training_curves(
        train_losses,
        val_losses,
        train_aucs,
        val_aucs,
        title_prefix="Training Curves",
        save_path=plot_path,
    )
    print(f"Saving train/val data to {train_val_data_path}")
    # Create DataFrame with training/validation metrics
    training_data = pd.DataFrame(
        {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_auc": train_aucs,
            "val_auc": val_aucs,
        }
    )

    # Ensure file extension is .csv
    if not train_val_data_path.endswith(".csv"):
        file_name, _ = os.path.splitext(train_val_data_path)
        train_val_data_path = file_name + ".csv"

    # Ensure directory exists
    Path(train_val_data_path).parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV with headers
    training_data.to_csv(train_val_data_path, index=False)
    print("Training completed!")
    return (
        best_val_loss,
        best_val_auc,
        np.mean(epoch_train_times),
        np.sum(epoch_train_times),
        num_epochs_run,
        model,
    )

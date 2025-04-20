"""
Trainer module for training and evaluating models. Performs
validation and training of the model, and plots the training curves.
"""

import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import NormalizationMode, create_dataloader
from src.models import CXRModel
from src.utils import Config

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
    model: CXRModel,
    criterion: nn.Module | None = None,
    optimizer: optim.Optimizer | None = None,
    epochs: int = 50,
    lr: float = 1e-5,
    batch_size: int = 32,
    patience: int = 5,
    train_loader: DataLoader | None = None,
    val_loader: DataLoader | None = None,
    plot_path: str = "results/plots/training_curves.png",
    best_model_path: str = "results/models/best_model.pth",
    last_model_path: str = "results/models/last_model.pth",
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | Literal["cuda", "cpu"] = "cuda",
    num_workers: int = 32,
    normalization_mode: NormalizationMode = "imagenet",
) -> tuple[float, float]:
    """
    Train a CXR Model.

    Args:
        model (CXRModel): The model to train.
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

    Returns:
        tuple[float, float]: The best validation loss and it's associated
            AUC-ROC score.
    """
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
        train_loader = create_dataloader(
            clinical_data=cfg.tabular_clinical_train,
            cxr_images_dir=cfg.embedded_train_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            normalization_mode=normalization_mode,
        )

    if val_loader is None:
        val_loader = create_dataloader(
            clinical_data=cfg.tabular_clinical_val,
            cxr_images_dir=cfg.embedded_val_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            normalization_mode=normalization_mode,
        )

    best_val_loss = float("inf")
    best_val_auc = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []

    for epoch in range(epochs):
        epoch_display = epoch + 1
        logging.info(f"Epoch {epoch_display}/{epochs}")

        train_loss, train_auc = __train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            pb_prefix=f"T-{epoch_display}",
        )

        train_losses.append(train_loss)
        train_aucs.append(train_auc)

        val_loss, val_auc = __train_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
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
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if val_auc > best_val_auc:
            best_val_auc = val_auc

        if patience_counter >= patience:
            logging.info(
                f"Epoch {epoch_display}: Early stopping "
                f"triggered after {epoch_display} epochs"
            )
            break

    logging.info(f"Saving final model to {last_model_path}")
    torch.save(model.state_dict(), last_model_path)
    __plot_training_curves(
        train_losses,
        val_losses,
        train_aucs,
        val_aucs,
        title_prefix="Training Curves",
        save_path=plot_path,
    )
    logging.info("Training completed!")
    return best_val_loss, best_val_auc

"""Train a DenseNet-121 model on the NIH Chest X-ray dataset."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import random_split
from tqdm import tqdm

from src.data.create_dataloader import create_dataloader
from src.models.densenet_121_vanilla import DenseNet121Vanilla
from src.utils import Config

cfg = Config()


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc="Training")
    for images, _, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(train_loader)
    epoch_auc = roc_auc_score(
        np.array(all_labels), np.array(all_preds), average="macro"
    )
    return epoch_loss, epoch_auc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, _, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(val_loader)
    val_auc = roc_auc_score(np.array(all_labels), np.array(all_preds), average="macro")
    return val_loss, val_auc


def plot_training_curves(train_losses, val_losses, train_aucs, val_aucs):
    """Plot and save training curves."""
    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Plot AUC-ROC
    plt.subplot(1, 2, 2)
    plt.plot(train_aucs, label="Train AUC-ROC")
    plt.plot(val_aucs, label="Validation AUC-ROC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC-ROC")
    plt.title("Training and Validation AUC-ROC")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/plots/training_curves.png")
    plt.close()


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create directories if they don't exist
    Path("results/plots").mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    BATCH_SIZE = 32
    NUM_WORKERS = 16  # Increased due to 32 threads available
    NUM_EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-5

    # Create dataset and split into train/val
    dataset = create_dataloader(
        clinical_data=cfg.tabular_clinical_train,
        cxr_images_dir=cfg.cxr_train_dir,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        normalization_mode="imagenet",
    ).dataset

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize model, criterion, optimizer
    model = DenseNet121Vanilla(num_classes=15, freeze_backbone=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )

    # Training tracking
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []

    # Training loop
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}")

        # Train
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_aucs.append(train_auc)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        # Log metrics
        logger.info(
            f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}"
        )

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            file_path = cfg.artifacts / "densenet121_vanilla_best.pth"
            logger.info(f"Saving best model to {file_path}")
            torch.save(model.state_dict(), file_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Plot current curves
        plot_training_curves(train_losses, val_losses, train_aucs, val_aucs)

    logger.info("Training completed!")

    # Save final model
    final_model_path = cfg.artifacts / "densenet121_vanilla_final.pth"
    logger.info(f"Saving final model to {final_model_path}")
    torch.save(model.state_dict(), final_model_path)


if __name__ == "__main__":
    main()

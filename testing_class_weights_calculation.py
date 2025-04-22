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
from src.losses import FocalLoss, reweight

from src.data.create_dataloader import create_dataloader
#from src.models.densenet_121_vanilla import DenseNet121Vanilla
import random
import ViT_Base
print("Line 18:")
print("ViT_Base is loaded from:", ViT_Base.__file__)

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
    NUM_EPOCHS = 1 #50
    LR = 1e-4
    WEIGHT_DECAY = 1e-5

    # Setting Seeds for same split:
    # Set random seeds for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setting Seeds for Same split:

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

    # Class names for the 15 conditions
    CLASS_NAMES = [
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
        "Pleural Thickening",
        "Pneumonia",
        "Pneumothorax",
    ]
    # Calculate class weights using actual distribution
    print("\nCalculating class weights...")
    '''class_counts = []
    for i in range(15):
        count = sum(1 for _, _, labels in train_dataset if labels[i] == 1)
        class_counts.append(count)
        print(f"{CLASS_NAMES[i]}: {count} samples")'''
    
    '''# Preload all labels from the training dataset
    all_labels = []

    for _, _, label in train_dataset:
        all_labels.append(label.unsqueeze(0))  # shape: [1, 15]

    # Stack into a single tensor: shape [N, 15]
    all_labels_tensor = torch.cat(all_labels, dim=0)

    # Sum over rows to get per-class count
    class_counts = all_labels_tensor.sum(dim=0).tolist()

    # Optional: Round and cast to int for display
    class_counts = [int(x) for x in class_counts]'''
    # Collect all labels into a single numpy array
    all_labels = np.array([labels for _, _, labels in train_dataset])

    # Sum along axis 0 to get counts for each class
    class_counts = np.sum(all_labels, axis=0).tolist()

    # Print class counts
    print("\nClass counts:")
    for i, count in enumerate(class_counts):
        print(f"{CLASS_NAMES[i]}: {count} samples")


    class_weights = reweight(class_counts, beta=0.9999)
    print("\nClass weights:")
    for i, weight in enumerate(class_weights):
        print(f"{CLASS_NAMES[i]}: {weight:.4f}")
    print("class_weights = \n")
    print(class_weights)

if __name__ == "__main__":
    main()
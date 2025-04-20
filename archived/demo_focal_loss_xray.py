"""Demo script for training a model with Focal Loss on X-ray images."""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from src.losses.focal_loss import FocalLoss, reweight
from src.models.densenet_201_vanilla import DenseNet201Vanilla
from src.data.dataset import ChestXrayDataset
from src.utils import Config

# Initialize config
cfg = Config()

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


def calculate_metrics(outputs, labels):
    """Calculate per-class and average metrics."""
    # Convert logits to probabilities
    probs = torch.sigmoid(outputs).cpu().detach().numpy()
    labels = labels.cpu().numpy()

    # Calculate per-class metrics
    per_class_auc = []
    per_class_ap = []

    for i in range(len(CLASS_NAMES)):
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
            ap = average_precision_score(labels[:, i], probs[:, i])
            per_class_auc.append(auc)
            per_class_ap.append(ap)
        except ValueError:
            # Handle case where a class has no positive samples
            per_class_auc.append(0.0)
            per_class_ap.append(0.0)

    # Calculate mean metrics
    mean_auc = np.mean(per_class_auc)
    mean_ap = np.mean(per_class_ap)

    return {
        "per_class_auc": per_class_auc,
        "per_class_ap": per_class_ap,
        "mean_auc": mean_auc,
        "mean_ap": mean_ap,
    }


def train_model(
    model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5
):
    """Train the model using Focal Loss with validation."""
    model.train()
    train_losses = []
    val_metrics = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, _, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_outputs = []
        val_labels = []

        with torch.no_grad():
            for images, _, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                val_outputs.append(outputs)
                val_labels.append(labels)

        # Concatenate all batches
        val_outputs = torch.cat(val_outputs, dim=0)
        val_labels = torch.cat(val_labels, dim=0)

        # Calculate metrics
        metrics = calculate_metrics(val_outputs, val_labels)
        val_metrics.append(metrics)

        # Print epoch results
        print(f"\nEpoch {epoch+1}:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Mean AUC: {metrics['mean_auc']:.4f}")
        print(f"Validation Mean AP: {metrics['mean_ap']:.4f}")

        # Print per-class metrics for a few classes as example
        print("\nPer-class metrics (first 3 classes):")
        for i in range(3):
            print(f"{CLASS_NAMES[i]}:")
            print(f"  AUC: {metrics['per_class_auc'][i]:.4f}")
            print(f"  AP: {metrics['per_class_ap'][i]:.4f}")

    return train_losses, val_metrics


def plot_training_curves(train_losses, val_metrics):
    """Plot training curves and metrics."""
    # Plot training loss
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot validation metrics
    plt.subplot(1, 2, 2)
    mean_aucs = [m["mean_auc"] for m in val_metrics]
    mean_aps = [m["mean_ap"] for m in val_metrics]
    plt.plot(mean_aucs, label="Mean AUC")
    plt.plot(mean_aps, label="Mean AP")
    plt.title("Validation Metrics over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("focal_loss_training_curves.png")
    plt.close()


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transforms for DenseNet201
    transform = transforms.Compose(
        [
            transforms.Resize(256),  # DenseNet201 expects 256x256 input
            transforms.CenterCrop(224),  # Then crop to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_dataset = ChestXrayDataset(
        clinical_data=cfg.tabular_clinical_train,
        cxr_images_dir=cfg.cxr_train_dir,
        transform=transform,
    )

    val_dataset = ChestXrayDataset(
        clinical_data=cfg.tabular_clinical_test,
        cxr_images_dir=cfg.cxr_test_dir,
        transform=transform,
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Calculate class weights using actual distribution
    print("\nCalculating class weights...")
    class_counts = []
    for i in range(15):
        count = sum(1 for _, _, labels in train_dataset if labels[i] == 1)
        class_counts.append(count)
        print(f"{CLASS_NAMES[i]}: {count} samples")

    class_weights = reweight(class_counts, beta=0.9999)
    print("\nClass weights:")
    for i, weight in enumerate(class_weights):
        print(f"{CLASS_NAMES[i]}: {weight:.4f}")

    # Initialize model, loss functions, and optimizer
    model = DenseNet201Vanilla(num_classes=15).to(device)
    focal_criterion = FocalLoss(weight=class_weights, gamma=2.0)
    # bce_criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train with Focal Loss
    print("\nTraining with Focal Loss...")
    focal_losses, focal_metrics = train_model(
        model, train_loader, val_loader, focal_criterion, optimizer, device
    )

    # Plot training curves
    plot_training_curves(focal_losses, focal_metrics)

    # Save the model
    # torch.save(model.state_dict(), 'focal_loss_xray_model.pth')
    print("\nTraining completed and model saved!")


if __name__ == "__main__":
    main()

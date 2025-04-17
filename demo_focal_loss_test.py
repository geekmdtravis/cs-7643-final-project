"""Demo script to test Focal Loss implementation with DenseNet201Vanilla model."""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.models.densenet_201_vanilla import DenseNet201Vanilla
from src.losses.focal_loss import FocalLoss, reweight


def create_imbalanced_data(batch_size, num_classes, imbalance_ratio=10):
    """Create imbalanced data for testing.

    Args:
        batch_size: Total number of samples
        num_classes: Number of classes
        imbalance_ratio: Ratio between most and least frequent class
    """
    # Calculate samples per class to create imbalance
    samples_per_class = []
    for i in range(num_classes):
        # Create exponential decay in class frequencies
        samples = int(batch_size / (imbalance_ratio ** (i / (num_classes - 1))))
        samples_per_class.append(samples)

    # Create data and labels
    all_data = []
    all_labels = []

    for class_idx, num_samples in enumerate(samples_per_class):
        # Create random data for this class
        class_data = torch.randn(num_samples, 3, 224, 224)
        all_data.append(class_data)

        # Create one-hot labels for this class
        class_labels = torch.zeros(num_samples, num_classes)
        class_labels[:, class_idx] = 1.0
        all_labels.append(class_labels)

    # Concatenate all classes
    data = torch.cat(all_data, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Shuffle the data
    indices = torch.randperm(len(data))
    data = data[indices]
    labels = labels[indices]

    return data, labels, samples_per_class


def test_focal_loss():
    """Test Focal Loss implementation with DenseNet201Vanilla model."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create imbalanced dataset with smaller size
    batch_size = 100  # Reduced from 1000 to 100
    num_classes = 15
    imbalance_ratio = (
        10  # Most frequent class will have 10x more samples than least frequent
    )

    print("\nCreating imbalanced dataset...")
    data, labels, samples_per_class = create_imbalanced_data(
        batch_size, num_classes, imbalance_ratio
    )

    # Print class distribution
    print("\nClass distribution:")
    for i, count in enumerate(samples_per_class):
        class_name = "normal" if i == 0 else f"class_{i}"
        print(f"{class_name}: {count} samples")

    # Plot class distribution
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_classes), samples_per_class)
    plt.title("Class Distribution in Synthetic Dataset")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.xticks(
        range(num_classes),
        ["normal"] + [f"class_{i}" for i in range(1, num_classes)],
        rotation=45,
    )
    plt.tight_layout()
    plt.savefig("synthetic_class_distribution.png")
    plt.close()

    # Move data to device
    data = data.to(device)
    labels = labels.to(device)

    # Initialize model
    model = DenseNet201Vanilla(num_classes=num_classes).to(device)
    model.eval()  # Set to evaluation mode

    # Calculate class weights using actual distribution
    class_weights = reweight(samples_per_class, beta=0.9999)
    print("\nClass weights:")
    for i, weight in enumerate(class_weights):
        class_name = "normal" if i == 0 else f"class_{i}"
        print(f"{class_name}: {weight:.4f}")

    # Initialize Focal Loss
    criterion = FocalLoss(weight=class_weights, gamma=2.0)

    try:
        # Forward pass
        print("\nRunning forward pass...")
        with torch.no_grad():
            output = model(data)
            print(f"Model output shape: {output.shape}")
            print(f"Target shape: {labels.shape}")

        # Compute loss
        print("\nComputing Focal Loss...")
        loss = criterion(output, labels)
        print(f"Focal Loss value: {loss.item():.4f}")

        # Test with different gamma values
        print("\nTesting different gamma values...")
        for gamma in [0.5, 1.0, 2.0, 3.0]:
            criterion = FocalLoss(weight=class_weights, gamma=gamma)
            loss = criterion(output, labels)
            print(f"Gamma {gamma}: Loss = {loss.item():.4f}")

        print("\nAll tests completed successfully!")
        return True

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return False


if __name__ == "__main__":
    test_focal_loss()

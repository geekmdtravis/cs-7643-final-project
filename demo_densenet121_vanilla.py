"""Demo script for DenseNet121Vanilla model inference."""

from io import BytesIO
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image

from src.models import DenseNet121Vanilla  # Updated import


def download_image(url) -> Image.Image:
    """Download an image from a URL."""
    response = requests.get(url, timeout=10_000)
    if response.status_code != 200:
        raise requests.RequestException(f"Failed to download image from {url}")
    return Image.open(BytesIO(response.content))


def load_imagenet_labels():
    """Load ImageNet class labels."""
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url, timeout=10_000)
    if response.status_code != 200:
        raise requests.RequestException(f"Failed to load labels from {url}")
    labels = response.text.strip().split("\n")
    return labels


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for DenseNet inference."""
    mean_transform = [0.485, 0.456, 0.406]
    std_transform = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_transform, std=std_transform),
        ]
    )
    return transform(image).unsqueeze(0)


def run_inference():
    """Run inference on a sample image using DenseNet121Vanilla."""
    # Initialize model and set to evaluation mode
    model = DenseNet121Vanilla(num_classes=1000)  # Updated model
    model.eval()

    # Sample image URL (fluffy white dog)
    image_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"

    try:
        # Download and preprocess image
        print(f"Downloading image from: {image_url}")
        image = download_image(image_url)
        input_tensor = preprocess_image(image)

        # Load labels
        labels = load_imagenet_labels()

        # Run inference
        print("Running inference...")
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get top 5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5)

        # Print results
        print("\nTop 5 predictions:")
        for i in range(5):
            print(f"{labels[top5_idx[i]]:>20}: {top5_prob[i].item()*100:.2f}%")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    run_inference()

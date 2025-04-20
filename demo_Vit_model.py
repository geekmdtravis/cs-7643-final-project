from io import BytesIO
import requests
import torch
from PIL import Image
import timm
from timm.data import resolve_model_data_config, create_transform


def download_image(url) -> Image.Image:
    """Download an image from a URL."""
    response = requests.get(url, timeout=10_000)
    if response.status_code != 200:
        raise requests.RequestException(f"Failed to download image from {url}")
    return Image.open(BytesIO(response.content)).convert("RGB")


def load_imagenet_labels():
    """Load ImageNet class labels."""
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url, timeout=10_000)
    if response.status_code != 200:
        raise requests.RequestException(f"Failed to load labels from {url}")
    return response.text.strip().split("\n")


def preprocess_image(image: Image.Image, model) -> torch.Tensor:
    """Preprocess image using Timm's model-specific transforms."""
    config = resolve_model_data_config(model)
    transform = create_transform(**config, is_training=False)
    return transform(image).unsqueeze(0)  # Add batch dimension


def run_inference():
    """Run inference using Timm ViT model."""
    #model_name = "vit_small_patch16_224.augreg_in1k"
    #model = timm.create_model(model_name, pretrained=True)
    model = timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=True)
    model.eval()

    image_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"

    try:
        print(f"Downloading image from: {image_url}")
        image = download_image(image_url)

        input_tensor = preprocess_image(image, model)
        labels = load_imagenet_labels()

        print("Running inference...")
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        top5_prob, top5_idx = torch.topk(probabilities, 5)

        print("\nTop 5 predictions:")
        for i in range(5):
            print(f"{labels[top5_idx[i]]:>20}: {top5_prob[i].item()*100:.2f}%")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    run_inference()

import torch
import torch.nn as nn
from torchvision.models import ViT_B_32_Weights, vit_b_32

class ViTB32Vanilla(nn.Module):
    def __init__(self, num_classes: int = 15):
        super(ViTB32Vanilla, self).__init__()
        self.model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
        num_features = self.model.heads.head.in_features

        # Replace the classification head
        if num_classes != 1000:
            self.model.heads = nn.Sequential(nn.Linear(num_features, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

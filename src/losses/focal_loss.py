import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight  # Tensor of class weights

    def forward(self, input, target):
        # input: (batch_size, num_classes)
        # target: (batch_size, num_classes)

        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(input)

        # Compute focal factor
        focal_weight = torch.pow(
            (1 - probs) * target + probs * (1 - target), self.gamma
        )

        # Apply focal weight and class weight (if given)
        loss = focal_weight * BCE_loss

        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0).to(input.device)

        return loss.mean()


def reweight(cls_num_list, beta=0.9999):
    cls_num_list = np.array(cls_num_list)
    effective_num = 1.0 - np.power(beta, cls_num_list)
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * len(cls_num_list)
    per_cls_weights = torch.from_numpy(weights).float()
    return per_cls_weights

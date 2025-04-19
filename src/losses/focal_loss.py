import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the focal loss between input and target.
        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, num_classes).
            target (torch.Tensor): The target tensor of shape (batch_size, num_classes).
        Returns:
            torch.Tensor: The computed focal loss.
        """

        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")

        probs = torch.sigmoid(input)

        focal_weight = torch.pow(
            (1 - probs) * target + probs * (1 - target), self.gamma
        )

        loss = focal_weight * BCE_loss

        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0).to(input.device)

        return loss.mean()


def reweight(cls_num_list: list[int], beta: float = 0.9999):
    """
    Reweight the classes based on effective number of samples.
    Args:
        cls_num_list (list[int]): List of class sample counts.
        beta (float): The beta parameter for effective number calculation.
    Returns:
        torch.Tensor: The computed class weights.
    """
    cls_num_list = np.array(cls_num_list)
    effective_num = 1.0 - np.power(beta, cls_num_list)
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * len(cls_num_list)
    per_cls_weights = torch.from_numpy(weights).float()
    return per_cls_weights

"""
训练增强工具模块

实现以下功能：
1. 梯度累计 (Gradient Accumulation)
2. Confidence Penalty 正则化
3. Focal Loss
4. Mixup 数据增强
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance

    Args:
        alpha (float): Weighting factor in range (0, 1) to balance positive/negative examples.
            Default: 0.25
        gamma (float): Focusing parameter for modulating loss. Higher gamma puts more focus
            on hard, misclassified examples. Default: 2.0
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', 'sum'.
            Default: 'mean'
    """

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) where C is the number of classes
            targets: (N,) where each value is in range [0, C-1]

        Returns:
            Loss tensor
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        probs = F.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ConfidencePenalty(nn.Module):
    """Confidence Penalty for preventing overconfidence

    Encourages the model to maintain higher entropy, preventing overconfidence
    on incorrect predictions.

    Args:
        num_classes (int): Number of classes in the classification task
        weight (float): Weight for the confidence penalty term. Default: 0.1
    """

    def __init__(self, num_classes: int, weight: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.max_entropy = np.log(num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) logits from the model

        Returns:
            Confidence penalty loss
        """
        probs = F.softmax(inputs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

        penalty = entropy - self.max_entropy
        return self.weight * torch.mean(penalty)


class Mixup:
    """Mixup data augmentation

    Implements mixup augmentation from "mixup: Beyond Empirical Risk Minimization"
    (https://arxiv.org/abs/1710.09412)

    Args:
        alpha (float): Alpha parameter for Beta distribution. Default: 0.4
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def __call__(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to a batch of images and labels.

        Args:
            images: (N, C, H, W) batch of images
            labels: (N,) batch of labels

        Returns:
            mixed_images: (N, C, H, W) mixed images
            labels_a: (N,) original labels
            labels_b: (N,) shuffled labels for mixing
            lam: mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        mixed_images = lam * images + (1 - lam) * images[index, :]
        labels_a, labels_b = labels, labels[index]

        return mixed_images, labels_a, labels_b, lam

    def mixup_criterion(
        self,
        criterion: nn.Module,
        pred: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """
        Compute loss for mixed labels.

        Args:
            criterion: Loss function (e.g., CrossEntropyLoss, FocalLoss)
            pred: (N, C) model predictions
            labels_a: (N,) first set of labels
            labels_b: (N,) second set of labels
            lam: mixing coefficient

        Returns:
            Mixed loss
        """
        return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


class CombinedLoss(nn.Module):
    """Combined loss with multiple components

    Combines cross-entropy/focal loss with confidence penalty.

    Args:
        use_focal (bool): Whether to use focal loss. Default: False
        focal_alpha (float): Alpha parameter for focal loss. Default: 0.25
        focal_gamma (float): Gamma parameter for focal loss. Default: 2.0
        confidence_penalty_weight (float): Weight for confidence penalty. Default: 0.0
        num_classes (int): Number of classes. Required if confidence_penalty_weight > 0
    """

    def __init__(
        self,
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        confidence_penalty_weight: float = 0.0,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.use_focal = use_focal

        if use_focal:
            self.base_loss = FocalLoss(
                alpha=focal_alpha, gamma=focal_gamma, reduction="mean"
            )
        else:
            self.base_loss = nn.CrossEntropyLoss(reduction="mean")

        if confidence_penalty_weight > 0:
            if num_classes is None:
                raise ValueError(
                    "num_classes must be provided when using confidence penalty"
                )
            self.confidence_penalty = ConfidencePenalty(
                num_classes=num_classes, weight=confidence_penalty_weight
            )
        else:
            self.confidence_penalty = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) logits from the model
            targets: (N,) ground truth labels

        Returns:
            Combined loss
        """
        loss = self.base_loss(inputs, targets)

        if self.confidence_penalty is not None:
            cp_loss = self.confidence_penalty(inputs)
            loss = loss + cp_loss

        return loss

    def compute_with_mixup(
        self,
        inputs: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """
        Compute loss with mixup augmentation.

        Args:
            inputs: (N, C) logits from the model
            labels_a: (N,) first set of labels
            labels_b: (N,) second set of labels
            lam: mixing coefficient

        Returns:
            Combined loss with mixup
        """
        if self.use_focal:
            loss_a = self.base_loss(inputs, labels_a)
            loss_b = self.base_loss(inputs, labels_b)
            base_loss = lam * loss_a + (1 - lam) * loss_b
        else:
            base_loss = lam * F.cross_entropy(inputs, labels_a) + (
                1 - lam
            ) * F.cross_entropy(inputs, labels_b)

        loss = base_loss

        if self.confidence_penalty is not None:
            cp_loss = self.confidence_penalty(inputs)
            loss = loss + cp_loss

        return loss


def apply_gradient_accumulation(
    loss: torch.Tensor,
    accumulation_steps: int,
    optimizer: torch.optim.Optimizer,
    batch_idx: int,
) -> bool:
    """
    Apply gradient accumulation.

    Args:
        loss: Loss tensor
        accumulation_steps: Number of steps to accumulate gradients
        optimizer: PyTorch optimizer
        batch_idx: Current batch index (0-based)

    Returns:
        True if optimizer step was performed, False otherwise
    """
    normalized_loss = loss / accumulation_steps
    normalized_loss.backward()

    should_step = (batch_idx + 1) % accumulation_steps == 0

    if should_step:
        optimizer.step()
        optimizer.zero_grad()

    return should_step

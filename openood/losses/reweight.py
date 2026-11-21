import torch
import torch.nn.functional as F

from .sce import soft_cross_entropy


def rew_ce(
    logits: torch.Tensor, labels: torch.Tensor, sample_weights: torch.Tensor
) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, reduction="none")
    return (losses * sample_weights.type_as(losses)).mean()


def rew_sce(
    logits: torch.Tensor, soft_labels: torch.Tensor, sample_weights: torch.Tensor
) -> torch.Tensor:
    losses = soft_cross_entropy(logits, soft_labels, reduce=False)
    return torch.mean(losses * sample_weights.type_as(losses))

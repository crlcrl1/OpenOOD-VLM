from collections.abc import Sequence
from typing import Literal

import torch


def loss_function(
    a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
) -> torch.Tensor | Literal[0]:

    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(
            1
            - cos_loss(
                a[item].view(a[item].shape[0], -1), b[item].view(b[item].shape[0], -1)
            )
        )
    return loss

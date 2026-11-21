from typing import cast

import torch
import torch.nn.functional as F
from torch import nn


class T2FNormNet(nn.Module):
    tau: torch.Tensor

    def __init__(self, backbone: nn.Module, tau: float, num_classes: int) -> None:
        super(T2FNormNet, self).__init__()

        self.register_buffer("tau", torch.tensor(tau))
        self.backbone = backbone
        if hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Identity()

        try:
            feature_size = cast(int, backbone.feature_size)
        except AttributeError:
            feature_size = cast(int, backbone.module.feature_size)

        self.new_fc = nn.Linear(feature_size, num_classes)

    def forward(
        self, x: torch.Tensor, return_feature: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        penultimate_features = self.backbone(x)
        if self.training:
            features = F.normalize(penultimate_features, dim=-1) / self.tau.item()
        else:
            features = penultimate_features / self.tau.item()

        logits_cls = self.new_fc(features)
        if return_feature:
            return logits_cls, penultimate_features
        else:
            return logits_cls

    def intermediate_forward(self, x: torch.Tensor) -> torch.Tensor:
        penultimate_features = self.backbone(x).squeeze()
        return penultimate_features

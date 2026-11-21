from typing import cast

import torch.nn.functional as F
from torch import Tensor, nn


class SimClrNet(nn.Module):
    def __init__(self, backbone: nn.Module, out_dim: int = 128) -> None:
        super(SimClrNet, self).__init__()

        self.backbone = backbone
        feature_dim = cast(int, backbone.feature_size)
        self.simclr_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, out_dim),
        )

    def forward(
        self, x: Tensor, return_feature: bool = False, return_feature_list: bool = False
    ) -> tuple[Tensor, list[Tensor]]:
        _, feature = self.backbone.forward(x, return_feature=True)

        return _, [F.normalize(self.simclr_head(feature), dim=-1)]

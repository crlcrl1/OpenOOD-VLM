from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
from torch import Tensor, nn
from torchvision.models.swin_transformer import SwinTransformer


class Swin_T(SwinTransformer):
    def __init__(
        self,
        patch_size: list[int] = [4, 4],
        embed_dim: int = 96,
        depths: Sequence[int] = [2, 2, 6, 2],
        num_heads: Sequence[int] = [3, 6, 12, 24],
        window_size: Sequence[int] = [7, 7],
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
    ) -> None:
        super(Swin_T, self).__init__(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            stochastic_depth_prob=stochastic_depth_prob,
            num_classes=num_classes,
        )
        self.feature_size = embed_dim * 2 ** (len(depths) - 1)

    @overload
    def forward(self, x: Tensor, return_feature: Literal[False] = False) -> Tensor: ...

    @overload
    def forward(
        self, x: Tensor, return_feature: Literal[True]
    ) -> tuple[Tensor, Tensor]: ...

    def forward(
        self, x: Tensor, return_feature: bool = False
    ) -> tuple[Tensor, Tensor] | Tensor:
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)

        if return_feature:
            return self.head(x), x
        else:
            return self.head(x)

    def forward_threshold(self, x: Tensor, threshold: float) -> Tensor:
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        feature = x.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.head(feature)

        return logits_cls

    def get_fc(self) -> tuple[np.ndarray, np.ndarray]:
        fc = self.head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self) -> nn.Linear:
        return self.head

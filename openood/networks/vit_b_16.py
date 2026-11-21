from typing import Literal, cast, overload

import numpy as np
import torch
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer


class ViT_B_16(VisionTransformer):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 768,
        mlp_dim: int = 3072,
        num_classes: int = 1000,
    ) -> None:
        super(ViT_B_16, self).__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_classes=num_classes,
        )
        self.feature_size = hidden_dim

    @overload
    def forward(
        self, x: torch.Tensor, return_feature: Literal[False] = False
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self, x: torch.Tensor, return_feature: Literal[True]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(
        self, x: torch.Tensor, return_feature: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        if return_feature:
            return self.heads(x), x
        else:
            return self.heads(x)

    def forward_threshold(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        feature = x.clip(max=threshold)
        logits_cls = self.heads(feature)

        return logits_cls

    def get_fc(self) -> tuple[np.ndarray, np.ndarray]:
        fc = cast(nn.Linear, self.heads[0])
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self) -> nn.Linear:
        return cast(nn.Linear, self.heads[0])

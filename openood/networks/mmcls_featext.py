from typing import cast

import numpy as np
from mmpretrain.apis import get_model
from mmpretrain.models import ImageClassifier
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from torch import Tensor, nn

from openood.utils.config import Config


@MODELS.register_module()
class ImageClassifierWithReturnFeature(ImageClassifier):
    def forward(
        self,
        inputs: Tensor,
        data_samples: list[DataSample] | None = None,
        mode: str = "tensor",
        *,
        return_feature: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor] | dict[str, Tensor] | list[DataSample]:
        if return_feature and mode == "tensor":
            features = self.extract_feat(inputs, stage="pre_logits")
            return self.get_fc_layer()(features), features
        return super().forward(inputs, data_samples=data_samples, mode=mode)

    def get_fc(self) -> tuple[np.ndarray, np.ndarray]:
        layer = self.get_fc_layer()
        return layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy()

    def get_fc_layer(self) -> nn.Linear:
        return cast(nn.Linear, self.head.layers.head)


def build_mmpretrain_vit(network_config: Config) -> ImageClassifierWithReturnFeature:
    return get_model(
        network_config.model,
        pretrained=network_config.checkpoint if network_config.pretrained else False,
        type=ImageClassifierWithReturnFeature,
        head={"num_classes": network_config.num_classes},
    )

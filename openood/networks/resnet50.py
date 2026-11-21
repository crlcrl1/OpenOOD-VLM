from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


class ResNet50(ResNet):
    def __init__(
        self,
        block: type[BasicBlock | Bottleneck] = Bottleneck,
        layers: Sequence[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
    ) -> None:
        super(ResNet50, self).__init__(
            block=block, layers=layers, num_classes=num_classes
        )
        self.feature_size = 2048

    @overload
    def forward(
        self,
        x: Tensor,
        return_feature: Literal[False] = False,
        return_feature_list: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        x: Tensor,
        return_feature: Literal[True],
        return_feature_list: bool = False,
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        return_feature: Literal[False] = False,
        *,
        return_feature_list: Literal[True],
    ) -> tuple[Tensor, list[Tensor]]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        return_feature: Literal[False],
        return_feature_list: Literal[True],
        /,
    ) -> tuple[Tensor, list[Tensor]]: ...

    def forward(
        self, x: Tensor, return_feature: bool = False, return_feature_list: bool = False
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]] | Tensor:
        feature1 = self.relu(self.bn1(self.conv1(x)))
        feature1 = self.maxpool(feature1)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        logits_cls = self.fc(feature)

        feature_list = [feature1, feature2, feature3, feature4, feature5]
        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            return logits_cls, feature_list
        else:
            return logits_cls

    def forward_threshold(self, x: Tensor, threshold: float) -> Tensor:
        feature1 = self.relu(self.bn1(self.conv1(x)))
        feature1 = self.maxpool(feature1)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.fc(feature)

        return logits_cls

    def intermediate_forward(self, x: Tensor, layer_index: int) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        if layer_index == 1:
            return out

        out = self.layer2(out)
        if layer_index == 2:
            return out

        out = self.layer3(out)
        if layer_index == 3:
            return out

        out = self.layer4(out)
        if layer_index == 4:
            return out

        raise ValueError

    def get_fc(self) -> tuple[np.ndarray, np.ndarray]:
        fc = self.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self) -> nn.Linear:
        return self.fc

from typing import cast

from torch import Tensor, nn


class MCDNet(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        super(MCDNet, self).__init__()

        self.backbone = backbone
        try:
            feature_size = cast(int, backbone.feature_size)
        except AttributeError:
            feature_size = cast(int, backbone.module.feature_size)

        self.fc1 = nn.Linear(feature_size, num_classes)
        self.fc2 = nn.Linear(feature_size, num_classes)

    # test conf
    def forward(
        self, x: Tensor, return_double: bool = False
    ) -> tuple[Tensor, Tensor] | Tensor:

        _, feature = self.backbone(x, return_feature=True)

        logits1 = self.fc1(feature)
        logits2 = self.fc2(feature)

        if return_double:
            return logits1, logits2
        else:
            return logits1

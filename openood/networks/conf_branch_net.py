from typing import cast

from torch import Tensor, nn


class ConfBranchNet(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        super(ConfBranchNet, self).__init__()

        self.backbone = backbone
        if hasattr(self.backbone, "fc"):
            # remove fc otherwise ddp will
            # report unused params
            self.backbone.fc = nn.Identity()

        try:
            feature_size = cast(int, backbone.feature_size)
        except AttributeError:
            feature_size = cast(int, backbone.module.feature_size)

        self.fc = nn.Linear(feature_size, num_classes)
        self.confidence = nn.Linear(feature_size, 1)

    # test conf
    def forward(
        self, x: Tensor, return_confidence: bool = False
    ) -> tuple[Tensor, Tensor] | Tensor:

        _, feature = self.backbone(x, return_feature=True)

        pred = self.fc(feature)
        confidence = self.confidence(feature)

        if return_confidence:
            return pred, confidence
        else:
            return pred

from typing import cast

from torch import Tensor, nn


class RotNet(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        super(RotNet, self).__init__()

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
        self.rot_fc = nn.Linear(feature_size, 4)

    def forward(
        self, x: Tensor, return_rot_logits: bool = False
    ) -> tuple[Tensor, Tensor] | Tensor:
        _, feature = self.backbone(x, return_feature=True)

        logits = self.fc(feature)
        rot_logits = self.rot_fc(feature)

        if return_rot_logits:
            return logits, rot_logits
        else:
            return logits

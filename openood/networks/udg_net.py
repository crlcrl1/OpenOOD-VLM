from typing import cast

from torch import Tensor, nn


class UDGNet(nn.Module):
    def __init__(
        self, backbone: nn.Module, num_classes: int, num_clusters: int
    ) -> None:
        super(UDGNet, self).__init__()
        self.backbone = backbone
        if hasattr(self.backbone, "fc"):
            # remove fc otherwise ddp will
            # report unused params
            self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(cast(int, backbone.feature_size), num_classes)
        self.fc_aux = nn.Linear(cast(int, backbone.feature_size), num_clusters)

    def forward(
        self, x: Tensor, return_feature: bool = False, return_aux: bool = False
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor] | Tensor:
        _, feature = self.backbone(x, return_feature=True)
        logits_cls = self.fc(feature)
        logits_aux = self.fc_aux(feature)

        if return_aux:
            if return_feature:
                return logits_cls, logits_aux, feature
            else:
                return logits_cls, logits_aux
        else:
            if return_feature:
                return logits_cls, feature
            else:
                return logits_cls

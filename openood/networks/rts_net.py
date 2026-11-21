from torch import Tensor, nn


class RTSNet(nn.Module):
    def __init__(
        self, backbone: nn.Module, feature_size: int, num_classes: int, dof: int = 16
    ) -> None:
        """
        dof: degree of freedom of variance
        """
        super(RTSNet, self).__init__()
        self.backbone = backbone
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.dof = dof
        self.logvar_rts = nn.Sequential(
            nn.Linear(feature_size, self.dof),
            nn.BatchNorm1d(self.dof),
        )

    def forward(
        self, x: Tensor, return_var: bool = False
    ) -> tuple[Tensor, Tensor] | Tensor:
        logits_cls, feature = self.backbone(x, return_feature=True)
        if return_var:
            logvar = self.logvar_rts(feature)
            variance = logvar.exp()
            return logits_cls, variance
        else:
            return logits_cls

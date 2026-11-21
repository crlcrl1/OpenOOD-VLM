import torch.nn.functional as F
from torch import Tensor, nn


class DropoutNet(nn.Module):
    def __init__(self, backbone: nn.Module, dropout_p: float) -> None:
        super(DropoutNet, self).__init__()
        self.backbone = backbone
        self.dropout_p = dropout_p

    def forward(self, x: Tensor, use_dropout: bool = True) -> Tensor:
        if use_dropout:
            return self.forward_with_dropout(x)
        else:
            return self.backbone(x)

    def forward_with_dropout(self, x: Tensor) -> Tensor:
        _, feature = self.backbone(x, return_feature=True)
        feature = F.dropout2d(feature, self.dropout_p, training=True)
        logits_cls = self.backbone.fc(feature)

        return logits_cls

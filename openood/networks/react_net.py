import numpy as np
from torch import Tensor, nn


class ReactNet(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super(ReactNet, self).__init__()
        self.backbone = backbone

    def forward(
        self, x: Tensor, return_feature: bool = False, return_feature_list: bool = False
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]]:
        try:
            return self.backbone(x, return_feature, return_feature_list)
        except TypeError:
            return self.backbone(x, return_feature)

    def forward_threshold(self, x: Tensor, threshold: float) -> Tensor:
        _, feature = self.backbone(x, return_feature=True)
        feature = feature.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.backbone.get_fc_layer()(feature)
        return logits_cls

    def get_fc(self) -> tuple[np.ndarray, np.ndarray]:
        fc = self.backbone.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

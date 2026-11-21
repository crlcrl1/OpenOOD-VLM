import numpy as np
import torch
from torch import nn


class ASHNet(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super(ASHNet, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        x: torch.Tensor,
        return_feature: bool = False,
        return_feature_list: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, list[torch.Tensor]]
    ):
        try:
            return self.backbone(x, return_feature, return_feature_list)
        except TypeError:
            return self.backbone(x, return_feature)

    def forward_threshold(self, x: torch.Tensor, percentile: float) -> torch.Tensor:
        _, feature = self.backbone(x, return_feature=True)
        feature = ash_b(feature.view(feature.size(0), -1, 1, 1), percentile)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.backbone.get_fc_layer()(feature)
        return logits_cls

    def get_fc(self) -> tuple[np.ndarray, np.ndarray]:
        fc = self.backbone.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()


def ash_b(x: torch.Tensor, percentile: float = 65) -> torch.Tensor:
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    t.zero_().scatter_(dim=1, index=i, src=fill)
    return x


def ash_p(x: torch.Tensor, percentile: float = 65) -> torch.Tensor:
    assert x.dim() == 4
    assert 0 <= percentile <= 100

    b, c, h, w = x.shape

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    return x


def ash_s(x: torch.Tensor, percentile: float = 65) -> torch.Tensor:
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])

    return x


def ash_rand(
    x: torch.Tensor, percentile: float = 65, r1: int = 0, r2: int = 10
) -> torch.Tensor:
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    v = v.uniform_(r1, r2)
    t.zero_().scatter_(dim=1, index=i, src=v)
    return x

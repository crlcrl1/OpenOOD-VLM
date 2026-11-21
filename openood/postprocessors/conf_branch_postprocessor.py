import torch
from torch import nn

from openood.utils.config import Config

from .base_postprocessor import BasePostprocessor


class ConfBranchPostprocessor(BasePostprocessor):
    def __init__(self, config: Config) -> None:
        super(ConfBranchPostprocessor, self).__init__(config)
        self.config = config

    @torch.no_grad()
    def postprocess(
        self, net: nn.Module, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, conf = net(data, return_confidence=True)
        conf = torch.sigmoid(conf)
        _, pred = torch.max(output, dim=1)
        return pred, conf

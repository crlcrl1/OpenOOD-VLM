from collections.abc import Sequence

import torch
from torch import nn

from openood.utils.config import Config

from .base_postprocessor import BasePostprocessor


class ASHPostprocessor(BasePostprocessor):
    def __init__(self, config: Config) -> None:
        super(ASHPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.percentile = self.args.percentile
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    @torch.no_grad()
    def postprocess(
        self, net: nn.Module, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = net.forward_threshold(data, self.percentile)
        _, pred = torch.max(output, dim=1)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred, energyconf

    def set_hyperparam(self, hyperparam: Sequence[float]) -> None:
        self.percentile = hyperparam[0]

    def get_hyperparam(self) -> float:
        return self.percentile

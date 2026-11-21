from collections.abc import Sequence

import torch
from torch import nn

from openood.utils.config import Config

from .base_postprocessor import BasePostprocessor


class MCMPostprocessor(BasePostprocessor):
    def __init__(self, config: Config) -> None:
        super(MCMPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    @torch.no_grad()
    def postprocess(
        self, net: nn.Module, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = net(data)
        score = torch.softmax(output / self.tau, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def set_hyperparam(self, hyperparam: Sequence[float]) -> None:
        self.tau = hyperparam[0]

    def get_hyperparam(self) -> float:
        return self.tau

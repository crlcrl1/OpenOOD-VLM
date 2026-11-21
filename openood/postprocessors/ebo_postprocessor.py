from collections.abc import Sequence

import torch
from torch import nn

from openood.utils.config import Config

from .base_postprocessor import BasePostprocessor


class EBOPostprocessor(BasePostprocessor):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.temperature = self.args.temperature
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    @torch.no_grad()
    def postprocess(
        self, net: nn.Module, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = net(data)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        conf = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
        return pred, conf

    def set_hyperparam(self, hyperparam: Sequence[float]) -> None:
        self.temperature = hyperparam[0]

    def get_hyperparam(self) -> float:
        return self.temperature

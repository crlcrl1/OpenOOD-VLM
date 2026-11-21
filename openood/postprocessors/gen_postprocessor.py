from collections.abc import Sequence

import torch
from torch import nn

from openood.utils.config import Config

from .base_postprocessor import BasePostprocessor


class GENPostprocessor(BasePostprocessor):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.gamma = self.args.gamma
        self.M = self.args.M
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    @torch.no_grad()
    def postprocess(
        self, net: nn.Module, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = net(data)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        conf = self.generalized_entropy(score, self.gamma, self.M)
        return pred, conf

    def set_hyperparam(self, hyperparam: Sequence[float]) -> None:
        self.gamma = hyperparam[0]
        self.M = hyperparam[1]

    def get_hyperparam(self) -> list[float]:
        return [self.gamma, self.M]

    def generalized_entropy(
        self, softmax_id_val: torch.Tensor, gamma: float = 0.1, M: int = 100
    ) -> torch.Tensor:
        probs = softmax_id_val
        probs_sorted = torch.sort(probs, dim=1)[0][:, -M:]
        scores = torch.sum(probs_sorted**gamma * (1 - probs_sorted) ** (gamma), dim=1)

        return -scores

import torch
from torch import nn

from openood._typing import LoaderDict, OODLoaders
from openood.trainers.dsvdd_trainer import init_center_c
from openood.utils.config import Config

from .base_postprocessor import BasePostprocessor


class DSVDDPostprocessor(BasePostprocessor):
    def __init__(self, config: Config) -> None:
        super(DSVDDPostprocessor, self).__init__(config)
        self.hyperpara = {}

    def setup(
        self, net: nn.Module, id_loader_dict: LoaderDict, ood_loader_dict: OODLoaders
    ) -> None:
        if self.config.c == "None" and self.config.network.name != "dcae":
            self.c = init_center_c(id_loader_dict["train"], net)
        else:
            self.c = self.config.c

    @torch.no_grad()
    def postprocess(
        self, net: nn.Module, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = net(data)
        if self.config.network.name != "dcae":
            conf = torch.sum(
                (outputs - self.c) ** 2, dim=tuple(range(1, outputs.dim()))
            )

        # this is for pre-training the dcae network from the original paper
        elif self.config.network.name == "dcae":
            conf = torch.sum((outputs - data) ** 2, dim=tuple(range(1, outputs.dim())))
        else:
            raise NotImplementedError

        return -1 * torch.ones(data.shape[0]), conf

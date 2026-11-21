from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from openood._typing import LoaderDict, OODLoaders
from openood.utils.config import Config

from .base_postprocessor import BasePostprocessor


class ReactPostprocessor(BasePostprocessor):
    def __init__(self, config: Config) -> None:
        super(ReactPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.percentile = self.args.percentile
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(
        self, net: nn.Module, id_loader_dict: LoaderDict, ood_loader_dict: OODLoaders
    ) -> None:
        if not self.setup_flag:
            activation_log = []
            net.eval()
            with torch.no_grad():
                for batch in tqdm(
                    id_loader_dict["val"], desc="Setup: ", position=0, leave=True
                ):
                    data = batch["data"].cuda()
                    data = data.float()

                    _, feature = net(data, return_feature=True)
                    activation_log.append(feature.data.cpu().numpy())

            self.activation_log = np.concatenate(activation_log, axis=0)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(
        self, net: nn.Module, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = net.forward_threshold(data, self.threshold)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred, energyconf

    def set_hyperparam(self, hyperparam: Sequence[float]) -> None:
        self.percentile = hyperparam[0]
        self.threshold = np.percentile(self.activation_log.flatten(), self.percentile)
        print(
            "Threshold at percentile {:2d} over id data is: {}".format(
                self.percentile, self.threshold
            )
        )

    def get_hyperparam(self) -> float:
        return self.percentile

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from openood._typing import LoaderDict, OODLoaders
from openood.utils.config import Config

from .base_postprocessor import BasePostprocessor

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


class DICEPostprocessor(BasePostprocessor):
    def __init__(self, config: Config) -> None:
        super(DICEPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.p = self.args.p
        self.mean_act = None
        self.masked_w = None
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
                    id_loader_dict["train"], desc="Setup: ", position=0, leave=True
                ):
                    data = batch["data"].cuda()
                    data = data.float()

                    _, feature = net(data, return_feature=True)
                    activation_log.append(feature.data.cpu().numpy())

            activation_log = np.concatenate(activation_log, axis=0)
            self.mean_act = activation_log.mean(0)
            self.setup_flag = True
        else:
            pass

    def calculate_mask(self, w: torch.Tensor) -> None:
        contrib = self.mean_act[None, :] * w.data.squeeze().cpu().numpy()
        self.thresh = np.percentile(contrib, self.p)
        mask = torch.Tensor((contrib > self.thresh)).cuda()
        self.masked_w = w * mask

    @torch.no_grad()
    def postprocess(
        self, net: nn.Module, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fc_weight, fc_bias = net.get_fc()
        if self.masked_w is None:
            self.calculate_mask(torch.from_numpy(fc_weight).cuda())
        _, feature = net(data, return_feature=True)
        vote = feature[:, None, :] * self.masked_w
        output = vote.sum(2) + torch.from_numpy(fc_bias).cuda()
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred, energyconf

    def set_hyperparam(self, hyperparam: Sequence[float]) -> None:
        self.p = hyperparam[0]

    def get_hyperparam(self) -> float:
        return self.p

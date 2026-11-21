from copy import deepcopy

import torch
from torch import nn
from tqdm import tqdm

from openood._typing import LoaderDict, OODLoaders
from openood.utils.config import Config

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


def distance(
    penultimate: torch.Tensor, target: torch.Tensor, metric: str = "inner_product"
) -> torch.Tensor:
    if metric == "inner_product":
        return torch.sum(torch.mul(penultimate, target), dim=1)
    elif metric == "euclidean":
        return -torch.sqrt(torch.sum((penultimate - target) ** 2, dim=1))
    elif metric == "cosine":
        return torch.cosine_similarity(penultimate, target, dim=1)
    else:
        raise ValueError("Unknown metric: {}".format(metric))


class SHEPostprocessor(BasePostprocessor):
    def __init__(self, config: Config) -> None:
        super(SHEPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.activation_log = None
        self.setup_flag = False

    def setup(
        self, net: nn.Module, id_loader_dict: LoaderDict, ood_loader_dict: OODLoaders
    ) -> None:
        if not self.setup_flag:
            net.eval()

            all_activation_log = []
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for batch in tqdm(
                    id_loader_dict["train"], desc="Eval: ", position=0, leave=True
                ):
                    data = batch["data"].cuda()
                    labels = batch["label"]
                    all_labels.append(deepcopy(labels))

                    logits, features = net(data, return_feature=True)
                    all_activation_log.append(features.cpu())
                    all_preds.append(logits.argmax(1).cpu())

            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            all_activation_log = torch.cat(all_activation_log)

            self.activation_log = []
            for i in range(self.num_classes):
                mask = torch.logical_and(all_labels == i, all_preds == i)
                class_correct_activations = all_activation_log[mask]
                self.activation_log.append(
                    class_correct_activations.mean(0, keepdim=True)
                )

            self.activation_log = torch.cat(self.activation_log).cuda()
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(
        self, net: nn.Module, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, feature = net(data, return_feature=True)
        pred = output.argmax(1)
        conf = distance(feature, self.activation_log[pred], self.args.metric)
        return pred, conf

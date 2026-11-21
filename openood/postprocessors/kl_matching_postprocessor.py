import numpy as np
import scipy
import torch
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances_argmin_min
from torch import nn
from tqdm import tqdm

from openood._typing import LoaderDict, OODLoaders
from openood.utils.config import Config

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


class KLMatchingPostprocessor(BasePostprocessor):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.setup_flag = False

    def kl(self, p: np.ndarray, q: np.ndarray) -> float:
        return scipy.stats.entropy(p, q)

    def setup(
        self, net: nn.Module, id_loader_dict: LoaderDict, ood_loader_dict: OODLoaders
    ) -> None:
        if not self.setup_flag:
            net.eval()

            print("Extracting id validation softmax posterior distributions")
            all_softmax = []
            preds = []
            with torch.no_grad():
                for batch in tqdm(
                    id_loader_dict["val"], desc="Setup: ", position=0, leave=True
                ):
                    data = batch["data"].cuda()
                    logits = net(data)
                    all_softmax.append(F.softmax(logits, 1).cpu())
                    preds.append(logits.argmax(1).cpu())

            all_softmax = torch.cat(all_softmax)
            preds = torch.cat(preds)

            self.mean_softmax_val = []
            for i in tqdm(range(self.num_classes)):
                # if there are no validation samples
                # for this category
                if torch.sum(preds.eq(i).float()) == 0:
                    temp = np.zeros((self.num_classes,))
                    temp[i] = 1
                    self.mean_softmax_val.append(temp)
                else:
                    self.mean_softmax_val.append(
                        all_softmax[preds.eq(i)].mean(0).numpy()
                    )

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(
        self, net: nn.Module, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = net(data)
        preds = logits.argmax(1)
        softmax = F.softmax(logits, 1).cpu().numpy()
        scores = -pairwise_distances_argmin_min(
            softmax, np.array(self.mean_softmax_val), metric=self.kl
        )[1]
        return preds, torch.from_numpy(scores)

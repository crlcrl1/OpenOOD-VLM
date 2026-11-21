import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from openood._typing import LoaderDict, Metrics, Network, OODLoaders
from openood.postprocessors.base_postprocessor import BasePostprocessor
from openood.utils import Config


class ADEvaluator:
    def __init__(self, config: Config) -> None:
        self.config = config

    def eval_ood(
        self,
        net: Network,
        id_data_loader: LoaderDict,
        ood_data_loaders: OODLoaders,
        postprocessor: BasePostprocessor,
        epoch_idx: int = -1,
    ) -> Metrics:
        with torch.no_grad():
            if type(net) is dict:
                for subnet in net.values():
                    subnet.eval()
            else:
                net.eval()
            auroc = self.get_auroc(
                net, id_data_loader["test"], ood_data_loaders["val"], postprocessor
            )
            metrics = {
                "epoch_idx": epoch_idx,
                "image_auroc": auroc,
            }
            return metrics

    def report(self, test_metrics: Metrics) -> None:

        print(
            "Complete Evaluation:\n"
            "{}\n"
            "==============================\n"
            "AUC Image: {:.2f} \n"
            "==============================".format(
                self.config.dataset.name, 100.0 * test_metrics["image_auroc"]
            ),
            flush=True,
        )
        print("Completed!", flush=True)

    def get_auroc(
        self,
        net: Network,
        id_data_loader: DataLoader,
        ood_data_loader: DataLoader,
        postprocessor: BasePostprocessor,
    ) -> float:
        _, id_conf, id_gt = postprocessor.inference(net, id_data_loader)
        _, ood_conf, ood_gt = postprocessor.inference(net, ood_data_loader)
        ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood

        conf = np.concatenate([id_conf, ood_conf])
        label = np.concatenate([id_gt, ood_gt])

        ind_indicator = np.zeros_like(label)
        ind_indicator[label != -1] = 1

        fpr, tpr, _ = roc_curve(ind_indicator, conf)

        auroc = auc(fpr, tpr)

        return auroc

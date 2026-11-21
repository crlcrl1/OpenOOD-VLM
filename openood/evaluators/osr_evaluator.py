from __future__ import annotations

from torch import nn

from openood._typing import LoaderDict, OODLoaders
from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .ood_evaluator import OODEvaluator


class OSREvaluator(OODEvaluator):
    def __init__(self, config: Config) -> None:
        super(OSREvaluator, self).__init__(config)

    def eval_ood(
        self,
        net: nn.Module,
        id_data_loader: LoaderDict,
        ood_data_loaders: OODLoaders,
        postprocessor: BasePostprocessor,
    ) -> None:
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()

        # load training in-distribution data
        assert "test" in id_data_loader, "id_data_loaders should have the key: test!"
        dataset_name = self.config.dataset.name
        print(f"Performing inference on {dataset_name} dataset...", flush=True)
        id_pred, id_conf, id_gt = postprocessor.inference(net, id_data_loader["test"])
        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        # load nearood data and compute ood metrics
        self._eval_ood(
            net,
            [id_pred, id_conf, id_gt],
            ood_data_loaders,
            postprocessor,
            ood_split="osr",
        )

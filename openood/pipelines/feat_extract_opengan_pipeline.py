from typing import cast

from torch import nn

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.evaluators.base_evaluator import BaseEvaluator
from openood.networks import get_network
from openood.utils import setup_logger
from openood.utils.config import Config


class FeatExtractOpenGANPipeline:
    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self) -> None:
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        id_loader_dict = get_dataloader(self.config)
        ood_loader_dict = get_ood_dataloader(self.config)
        assert "train" in id_loader_dict
        assert "val" in id_loader_dict
        assert "val" in ood_loader_dict

        # init network
        net = cast(nn.Module, get_network(self.config.network))

        # init evaluator
        evaluator = cast(BaseEvaluator, get_evaluator(self.config))

        # sanity check on id val accuracy
        print("\nStart evaluation on ID val data...", flush=True)
        test_metrics = evaluator.eval_acc(net, id_loader_dict["val"])
        print(
            "\nComplete Evaluation, accuracy {:.2f}%".format(100 * test_metrics["acc"]),
            flush=True,
        )

        # start extracting features
        print("\nStart Feature Extraction...", flush=True)
        print("\t ID training data...")
        evaluator.extract(net, id_loader_dict["train"], "id_train")

        print("\t ID val data...")
        evaluator.extract(net, id_loader_dict["val"], "id_val")

        print("\t OOD val data...")
        evaluator.extract(net, ood_loader_dict["val"], "ood_val")
        print("\nComplete Feature Extraction!")

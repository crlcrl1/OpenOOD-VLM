from typing import cast

from torch import nn

from openood.datasets import get_dataloader
from openood.evaluators import get_evaluator
from openood.evaluators.base_evaluator import BaseEvaluator
from openood.networks import get_network
from openood.utils import setup_logger
from openood.utils.config import Config


class TestAccPipeline:
    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self) -> None:
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        loader_dict = get_dataloader(self.config)
        test_loader = loader_dict["test"]

        # init network
        net = cast(nn.Module, get_network(self.config.network))

        # init evaluator
        evaluator = cast(BaseEvaluator, get_evaluator(self.config))

        # start calculating accuracy
        print("\nStart evaluation...", flush=True)
        test_metrics = evaluator.eval_acc(net, test_loader)
        print(
            "\nComplete Evaluation, accuracy {:.2f}%".format(100 * test_metrics["acc"]),
            flush=True,
        )

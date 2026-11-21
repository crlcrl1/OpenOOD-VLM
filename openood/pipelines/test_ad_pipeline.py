from typing import cast

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators.ad_evaluator import ADEvaluator
from openood.evaluators.utils import get_evaluator
from openood.networks.utils import get_network
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger
from openood.utils.config import Config


class TestAdPipeline:
    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self) -> None:
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        id_loader_dict = get_dataloader(self.config)
        ood_loader_dict = get_ood_dataloader(self.config)

        # init network
        net = get_network(self.config.network)

        # init evaluator
        evaluator = cast(ADEvaluator, get_evaluator(self.config))

        postprocessor = get_postprocessor(self.config)
        # setup for distance-based methods
        postprocessor.setup(net, id_loader_dict, ood_loader_dict)

        print("Start testing...", flush=True)
        test_metrics = evaluator.eval_ood(
            net, id_loader_dict, ood_loader_dict, postprocessor
        )
        evaluator.report(test_metrics)

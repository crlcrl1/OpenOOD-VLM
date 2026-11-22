"""Run the CIFAR-100 versus CIFAR-10 row of TANL Table A10(b) from the repo root."""

import argparse
import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from openood._typing import ScoreArrays
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators.fsood_evaluator_clip import OODEvaluatorClipTTA
from openood.evaluators.metrics import compute_all_metrics, fpr_recall
from openood.networks import get_network
from openood.networks.clip_fixed_ood_prompt import get_selected_ood_text_list
from openood.postprocessors.activated_neg_postprocessor import ActivatedNegPostprocessor
from openood.utils.config import Config, merge_configs

EXPERIMENT = Path(__file__).resolve().parent
CONFIGS = (
    "configs/datasets/cifar100/cifar100_clip.yml",
    "configs/networks/fixed_clip.yml",
    "configs/pipelines/test/test_fsood.yml",
    "configs/preprocessors/base_preprocessor.yml",
    "configs/postprocessors/tanl.yml",
)
SEEDS = (0, 1, 2)


class ScoreRecorder(ActivatedNegPostprocessor):
    """Retain the unmodified outputs of the repository's shuffled evaluation."""

    scores: ScoreArrays

    def inference(
        self, net: nn.Module, data_loader: DataLoader, progress: bool = True
    ) -> ScoreArrays:
        self.scores = super().inference(net, data_loader, progress)
        return self.scores


@dataclass
class Result:
    seed: int
    id_count: int
    ood_count: int
    fpr95_ood_positive_pct: float
    auroc_pct: float
    id_accuracy_pct: float
    fpr95_id_positive_pct: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_result(seed: int, scores: ScoreArrays) -> Result:
    pred, conf, label = scores
    id_count = int(np.count_nonzero(label >= 0))
    ood_count = int(np.count_nonzero(label == -1))
    if (id_count, ood_count) != (10000, 10000) or not np.isfinite(conf).all():
        raise ValueError("Unexpected evaluation sample count or non-finite confidence")
    np.savez_compressed(
        EXPERIMENT / f"scores_seed{seed}.npz", pred=pred, conf=conf, label=label
    )
    fpr, auroc, _, _, accuracy = compute_all_metrics(conf, label, pred)
    id_positive_fpr, _ = fpr_recall(conf, label, 0.95)
    return Result(
        seed,
        id_count,
        ood_count,
        100 * fpr,
        100 * auroc,
        100 * accuracy,
        100 * id_positive_fpr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-upstream", action="store_true")
    args = parser.parse_args()
    config = merge_configs(
        *(Config(path) for path in CONFIGS), Config(str(EXPERIMENT / "protocol.yml"))
    )
    config.parse_refs()
    id_loader = get_dataloader(config)["test"]
    ood_loaders = get_ood_dataloader(config)
    # Building the WordNet cache loads a second CLIP model and consumes RNG.
    # Finish it before seeding construction of the noise images for all runs.
    get_selected_ood_text_list(config.dataset.name, config.network.backbone.ood_number)
    set_seed(0)
    net = get_network(config.network).eval()
    print(
        "ID classes:",
        net.n_cls,
        "candidate words:",
        net.text_features_all.shape[1] - net.n_cls,
        flush=True,
    )
    evaluator = OODEvaluatorClipTTA(config)
    results = []
    for seed in SEEDS:
        set_seed(seed)
        postprocessor = ScoreRecorder(config)
        evaluator._eval_ood(net, id_loader, ood_loaders, postprocessor, "nearood")
        result = save_result(seed, postprocessor.scores)
        print(result, flush=True)
        results.append(asdict(result))
        with (EXPERIMENT / "metrics.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    if args.check_upstream:
        from upstream_check import check_upstream

        set_seed(SEEDS[0])
        check_upstream(net, config, id_loader, ood_loaders, EXPERIMENT)


if __name__ == "__main__":
    main()

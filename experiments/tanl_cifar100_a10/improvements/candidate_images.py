"""Verify the joint score on real CIFAR PNGs and evaluate it on DTD JPEGs."""

import numpy as np
import torch
from ablation import write_rows
from capture import (
    HERE,
    CaptureQuantiles,
    load_image_network,
    method_config,
    uniform_bank,
)
from run import SEEDS, set_seed

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators.fsood_evaluator_clip import OODEvaluatorClipTTA
from openood.evaluators.metrics import compute_all_metrics
from openood.utils.config import Config, merge_configs


@torch.no_grad()
def main():
    config = merge_configs(
        method_config(),
        Config(str(HERE / "positive.yml")),
        Config(str(HERE / "joint_class.yml")),
    )
    config.ood_dataset.split_names = ["nearood", "farood"]
    config.ood_dataset.nearood.datasets = ["cifar10"]
    config.ood_dataset.farood.datasets = ["texture"]
    net = load_image_network(config, uniform_bank())
    id_loader, ood_loaders = get_dataloader(config)["test"], get_ood_dataloader(config)
    evaluator = OODEvaluatorClipTTA(config)
    rows = []
    for dataset, split, seeds in (
        ("cifar10", "nearood", (0,)),
        ("texture", "farood", SEEDS),
    ):
        for seed in seeds:
            class_prob = []

            def record(module, args, output, target=class_prob):
                features, _, scale = output
                # Independently recompute on the original, fixed text matrix. The
                # integrated scorer uses logits from the final ranked text matrix.
                logits = (scale * features @ module.text_features)[:, : module.n_cls]
                target.append(logits.float().softmax(1).amax(1).cpu())

            set_seed(seed)
            real = CaptureQuantiles(config)
            with net.register_forward_hook(record):
                evaluator._eval_ood(net, id_loader, ood_loaders, real, split)
            pred, joint, label = real.scores
            prob = torch.cat(class_prob).numpy()
            with np.load(
                HERE / "scores" / f"{dataset}_uniform_positive_{seed}.npz"
            ) as baseline:
                if not np.array_equal(label, baseline["label"]) or not np.array_equal(
                    pred, baseline["pred"]
                ):
                    raise ValueError(
                        f"Sample order or classifier changed for {dataset}, seed {seed}"
                    )
                original = baseline["conf"].astype(np.float32)
            if not np.array_equal(joint, original * prob):
                raise ValueError(
                    f"Integrated and independent joint scores differ for {dataset}, seed {seed}"
                )
            for method, score in (
                ("positive", original),
                ("top_class_only", prob),
                ("joint_top_class", joint),
            ):
                fpr, auc, _, _, acc = compute_all_metrics(score, label, pred)
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "fpr95": 100 * fpr,
                        "auroc": 100 * auc,
                        "id_accuracy": 100 * acc,
                        "images": len(label),
                    }
                )
            print(
                f"Verified {dataset} seed {seed}: {len(label)} joint scores", flush=True
            )
    write_rows(HERE / "candidate_images.csv", rows)


if __name__ == "__main__":
    main()

"""Evaluate the frozen positive-only choice and its uniform-template control on DTD."""

import numpy as np
import torch
from ablation import replay, summarize, write_rows
from capture import (
    HERE,
    CaptureQuantiles,
    load_image_network,
    method_config,
    prepare_ood_split,
    uniform_bank,
)
from diagnose import FeatureBank
from run import SEEDS, set_seed

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators.fsood_evaluator_clip import OODEvaluatorClipTTA
from openood.evaluators.metrics import compute_all_metrics
from openood.utils.config import Config, merge_configs


@torch.no_grad()
def main():
    prepare_ood_split("texture", 5640)
    config = merge_configs(method_config(), Config(str(HERE / "positive.yml")))
    config.ood_dataset.split_names = ["farood"]
    config.ood_dataset.farood.datasets = ["texture"]
    id_loader, ood = get_dataloader(config)["test"], get_ood_dataloader(config)
    evaluator = OODEvaluatorClipTTA(config)
    bank = uniform_bank()
    raw, cached = load_image_network(config, bank), FeatureBank(bank, 1000)
    rows = []
    for seed in SEEDS:
        features = []

        def record(module, args, output, target=features):
            target.append(output[0].cpu())

        set_seed(seed)
        with raw.register_forward_hook(record):
            real = CaptureQuantiles(config)
            evaluator._eval_ood(raw, id_loader, ood, real, "farood")
        raw_pred, raw_conf, labels = real.scores
        if (
            np.count_nonzero(labels >= 0) != 10000
            or np.count_nonzero(labels == -1) != 5640
        ):
            raise ValueError("Unexpected DTD or CIFAR-100 split")
        stream = {"images": torch.cat(features), "rng": real.cpu_rng}
        for method in ("baseline", "positive"):
            # A fresh config per replay keeps the independent raw configuration fixed.
            scores, _ = replay(method_config(), cached, method, stream)
            pred, conf = scores["pred"], scores["conf"]
            if not np.array_equal(raw_pred, pred) or (
                method == "positive" and not np.array_equal(raw_conf, conf)
            ):
                raise ValueError(
                    f"Raw/replay prediction or score mismatch for {method}"
                )
            fpr, auc, _, _, acc = compute_all_metrics(conf, labels, pred)
            np.savez_compressed(
                HERE / "scores" / f"texture_uniform_{method}_{seed}.npz",
                label=labels,
                **scores,
            )
            rows.append(
                {
                    "dataset": "texture",
                    "template": "uniform",
                    "method": method,
                    "seed": seed,
                    "fpr95": 100 * fpr,
                    "auroc": 100 * auc,
                    "id_accuracy": 100 * acc,
                    "raw_scores_matched": len(conf) if method == "positive" else 0,
                }
            )
            print(
                f"DTD seed {seed}, {method}: FPR95 {100 * fpr:.2f}, AUROC {100 * auc:.2f}",
                flush=True,
            )
    write_rows(HERE / "texture_seeds.csv", rows)
    write_rows(HERE / "texture_summary.csv", summarize(rows))


if __name__ == "__main__":
    main()

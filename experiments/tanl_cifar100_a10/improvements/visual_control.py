"""Independently reconstruct the two real-image FIFOs and score visual evidence alone."""

import numpy as np
import torch
from ablation import DATASETS, TEMPLATES, summarize, write_rows
from capture import HERE, method_config
from capture_features import CACHE
from run import SEEDS
from scipy.special import logit, logsumexp

from openood.evaluators.metrics import compute_all_metrics


def visual_odds(images, gates, scale, capacity, batch_size):
    # A separate CPU implementation uses only the already-recorded admission
    # decisions and image vectors. No class labels enter either FIFO.
    positive, negative = images[:0], images[:0]
    odds = np.zeros(len(images))
    for offset in range(0, len(images), batch_size):
        step = slice(offset, offset + batch_size)
        batch = images[step]
        if len(positive) and len(negative):
            pos_density = logsumexp(scale * (batch @ positive.T), axis=1) - np.log(
                len(positive)
            )
            neg_density = logsumexp(scale * (batch @ negative.T), axis=1) - np.log(
                len(negative)
            )
            odds[step] = pos_density - neg_density
        positive = np.concatenate((positive, batch[gates[step] == 1]))[-capacity:]
        negative = np.concatenate((negative, batch[gates[step] == -1]))[-capacity:]
    return odds


def main():
    config = method_config()
    scale = float(torch.load(CACHE / "bank.pt", map_location="cpu")["scale"])
    rows = []
    for dataset in DATASETS:
        prefix = "" if dataset == "cifar10" else f"{dataset}_"
        for seed in SEEDS:
            stream = torch.load(CACHE / f"{prefix}stream{seed}.pt", map_location="cpu")
            images = stream["images"].numpy().astype(np.float64)
            for template in TEMPLATES:
                for method, fusion in (
                    ("baseline", "visual_otsu"),
                    ("quantiles", "visual_quantiles"),
                ):
                    run = HERE / "scores" / f"{dataset}_{template}_{method}_{seed}.npz"
                    fused = (
                        HERE / "scores" / f"{dataset}_{template}_{fusion}_{seed}.npz"
                    )
                    with np.load(run) as text, np.load(fused) as combined:
                        if not all(
                            np.array_equal(text[k], combined[k])
                            for k in ("pred", "label", "gate")
                        ):
                            raise ValueError(
                                f"Visual scoring changed the classifier, stream or queue in {fused}"
                            )
                        odds = visual_odds(
                            images,
                            text["gate"],
                            scale,
                            config.postprocessor.postprocessor_args.memleng,
                            config.dataset.test.batch_size,
                        )
                        # Match the GPU clamp in float32 before evaluating in float64.
                        p = np.clip(text["conf"].astype(np.float32), 1e-7, 1 - 1e-7)
                        error = float(
                            np.max(
                                np.abs(
                                    odds
                                    + logit(p.astype(np.float64))
                                    - combined["conf"]
                                )
                            )
                        )
                        if error > 2e-4:
                            raise ValueError(
                                f"Visual log odds differ by {error} in {fused}"
                            )
                        fpr, auc, _, _, acc = compute_all_metrics(
                            odds, text["label"], text["pred"]
                        )
                    rows.append(
                        {
                            "dataset": dataset,
                            "template": template,
                            "method": fusion + "_only",
                            "seed": seed,
                            "fpr95": 100 * fpr,
                            "auroc": 100 * auc,
                            "id_accuracy": 100 * acc,
                            "max_abs_fused_error": error,
                        }
                    )
    write_rows(HERE / "visual_only_seeds.csv", rows)
    write_rows(HERE / "visual_only_summary.csv", summarize(rows))
    print(
        f"Verified {len(rows)} pairs of visual/text scores; max error {max(row['max_abs_fused_error'] for row in rows):.3g}"
    )


if __name__ == "__main__":
    main()

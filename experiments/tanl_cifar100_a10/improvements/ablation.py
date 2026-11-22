"""Compare both TANL changes on paired frozen streams and check raw-image references."""

import argparse
import csv
from collections import defaultdict
from unittest.mock import patch

import numpy as np
import torch
from capture import HERE, method_config
from capture_features import CACHE
from diagnose import FeatureBank
from run import EXPERIMENT, SEEDS

from openood.evaluators.metrics import compute_all_metrics
from openood.postprocessors import get_postprocessor

METHODS = ("baseline", "quantiles", "diversity", "combined")
VISUAL_METHODS = ("visual_otsu", "visual_quantiles")
SINGLE_SIDES = ("positive", "negative")
TEMPLATES = ("released", "uniform")
DATASETS = ("cifar10", "tin")


def replay(config, net, method, stream):
    args = config.postprocessor.postprocessor_args
    args.admission_tail = (
        0.20
        if method in ("quantiles", "combined", "visual_quantiles", *SINGLE_SIDES)
        else None
    )
    args.admission_side = method if method in SINGLE_SIDES else "both"
    args.diversity_correlation = 0.90 if method in ("diversity", "combined") else None
    args.visual_cache = method in VISUAL_METHODS
    config.postprocessor.name = "actneg" if method == "baseline" else "adaptive_tanl"
    post = get_postprocessor(config)
    post.reset_memory()
    gates, replacements, preds, confs = [], [], [], []
    admission_masks, rank = post._admission_masks, post._rank_batch_negatives

    def record_admission(scores):
        positive, negative = admission_masks(scores)
        gates.append((positive.to(torch.int8) - negative.to(torch.int8)).cpu())
        return positive, negative

    def record_rank(scores, activations, count):
        indices = rank(scores, activations, count)
        if args.diversity_correlation is not None:
            original = scores.sort(descending=True).indices[:count]
            replacements.append((~torch.isin(indices, original)).sum().item())
        return indices

    torch.set_rng_state(stream["rng"])
    with (
        patch.object(post, "_admission_masks", record_admission),
        patch.object(post, "_rank_batch_negatives", record_rank),
    ):
        for images in stream["images"].split(config.dataset.test.batch_size):
            pred, conf = post.postprocess(net, images.cuda())
            preds.append(pred.cpu())
            confs.append(conf.cpu())
    return {
        "pred": torch.cat(preds).numpy(),
        "conf": torch.cat(confs).numpy(),
        "gate": torch.cat(gates).numpy(),
    }, float(np.mean(replacements)) if replacements else 0.0


def evaluate(dataset, template, method, seed, stream, outputs, replacements):
    label = stream["labels"].numpy()
    pred, conf, gate = (outputs[k] for k in ("pred", "conf", "gate"))
    is_id = label >= 0
    if (
        is_id.sum() != 10000
        or (~is_id).sum() != {"tin": 6526, "cifar10": 10000}[dataset]
    ):
        raise ValueError(f"Unexpected {dataset} split")
    if not np.isfinite(conf).all() or len(gate) != len(label):
        raise ValueError("Invalid confidence or admission mask")
    # Check all classifiers and sample orders. Check scores as well when the
    # corresponding method has an independent run through the real image loader.
    if template == "uniform" and method == "visual_quantiles" and seed == 0:
        ref = CACHE / f"{dataset}_raw_visual_quantiles{seed}.npz"
        compare_score = True
    elif dataset == "tin" or (
        template == "uniform" and method == "quantiles" and seed == 0
    ):
        ref = CACHE / f"{dataset}_raw_quantiles{seed}.npz"
        compare_score = template == "uniform" and method == "quantiles"
    else:
        ref = EXPERIMENT / f"scores_seed{seed}.npz"
        compare_score = template == "released" and method == "baseline"
    with np.load(ref) as reference:
        if not (
            np.array_equal(pred, reference["pred"])
            and np.array_equal(label, reference["label"])
        ):
            raise ValueError(f"Classifier or image order differs from {ref}")
        if compare_score and not np.array_equal(conf, reference["conf"]):
            raise ValueError(f"Scores differ from {ref}")
    fpr, auc, _, _, acc = compute_all_metrics(conf, label, pred)
    return {
        "dataset": dataset,
        "template": template,
        "method": method,
        "seed": seed,
        "fpr95": 100 * fpr,
        "auroc": 100 * auc,
        "id_accuracy": 100 * acc,
        "positive_count": int((gate == 1).sum()),
        "positive_true_id": int(((gate == 1) & is_id).sum()),
        "negative_count": int((gate == -1).sum()),
        "negative_true_ood": int(((gate == -1) & ~is_id).sum()),
        "replaced_per_batch": replacements,
        "raw_scores_matched": len(conf) if compare_score else 0,
    }


def write_rows(path, rows):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    by_case = defaultdict(list)
    for row in rows:
        by_case[(row["dataset"], row["template"], row["method"])].append(row)
    summary = []
    for (dataset, template, method), group in by_case.items():
        item = {
            "dataset": dataset,
            "template": template,
            "method": method,
            "seeds": len(group),
        }
        for metric in ("fpr95", "auroc", "id_accuracy"):
            values = [r[metric] for r in group]
            item[f"{metric}_mean"] = np.mean(values)
            item[f"{metric}_sd"] = np.std(values, ddof=1)
        summary.append(item)
    return summary


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, nargs="+", default=DATASETS)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--visual", action="store_true", help="run the visual cache follow-up"
    )
    group.add_argument(
        "--sides", action="store_true", help="change one queue gate at a time"
    )
    options = parser.parse_args()
    config = method_config()
    bank = torch.load(CACHE / "bank.pt", map_location="cpu")
    alternatives = {
        "released": bank,
        "uniform": {
            **bank,
            "texts": torch.load(CACHE / "uniform_texts.pt", map_location="cpu"),
        },
    }
    score_dir = HERE / "scores"
    score_dir.mkdir(exist_ok=True)
    methods, suffix = METHODS, ""
    if options.visual:
        methods, suffix = VISUAL_METHODS, "_visual"
    if options.sides:
        methods, suffix = SINGLE_SIDES, "_sides"
    for dataset in options.dataset:
        rows = []
        for template in TEMPLATES:
            net = FeatureBank(
                alternatives[template], config.postprocessor.postprocessor_args.beta
            )
            for seed in SEEDS:
                prefix = "" if dataset == "cifar10" else f"{dataset}_"
                stream = torch.load(
                    CACHE / f"{prefix}stream{seed}.pt", map_location="cpu"
                )
                for method in methods:
                    outputs, replacements = replay(config, net, method, stream)
                    row = evaluate(
                        dataset, template, method, seed, stream, outputs, replacements
                    )
                    np.savez_compressed(
                        score_dir / f"{dataset}_{template}_{method}_{seed}.npz",
                        label=stream["labels"].numpy(),
                        **outputs,
                    )
                    rows.append(row)
                    print(
                        f"{dataset} {template} {method} seed={seed}: FPR95 {row['fpr95']:.2f}, AUROC {row['auroc']:.2f}",
                        flush=True,
                    )
        stem = dataset + suffix
        write_rows(HERE / f"{stem}_seeds.csv", rows)
        write_rows(HERE / f"{stem}_summary.csv", summarize(rows))


if __name__ == "__main__":
    main()

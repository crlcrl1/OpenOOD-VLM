"""Compare seed 0 to the upstream postprocessor on the same frozen network."""

import importlib.util
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from openood._typing import OODLoaders
from openood.evaluators import fsood_evaluator_clip
from openood.evaluators.metrics import compute_all_metrics
from openood.utils.config import Config

UPSTREAM = "c6fef2f5ed890df2f59f9c6c3f9ae14fb5a72ab9"
SOURCE = "openood/postprocessors/activated_neg_postprocessor.py"


def load_upstream(directory: Path) -> ModuleType:
    path = directory / "activated_neg_postprocessor.py"
    path.write_bytes(subprocess.check_output(["git", "show", f"{UPSTREAM}:{SOURCE}"]))
    spec = importlib.util.spec_from_file_location(
        "openood.postprocessors.a10_ref", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load upstream TANL from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_upstream(
    net: nn.Module,
    config: Config,
    id_loader: DataLoader,
    ood_loaders: OODLoaders,
    experiment: Path,
) -> None:
    with TemporaryDirectory() as directory:
        upstream = load_upstream(Path(directory))
    batch_size = id_loader.batch_size
    gap = config.postprocessor.postprocessor_args.gap
    positive_batches = []
    negative_batches = []
    thresholds = []
    threshold_fn = upstream.find_best_threshold

    def record_threshold(history: torch.Tensor) -> float:
        threshold = threshold_fn(history)
        batch = history[-batch_size:]
        positive_batches.append(
            (batch > threshold + gap * (1 - threshold)).cpu().numpy()
        )
        negative_batches.append((batch < threshold - gap * threshold).cpu().numpy())
        thresholds.append(float(threshold))
        return threshold

    # Observe the detector and evaluator at their existing interfaces. Neither
    # receives additional inputs, and the observer returns the original results.
    with (
        patch.object(upstream, "find_best_threshold", record_threshold),
        patch.object(
            fsood_evaluator_clip, "compute_all_metrics", wraps=compute_all_metrics
        ) as metrics,
    ):
        evaluator = fsood_evaluator_clip.OODEvaluatorClipTTA(config)
        evaluator._eval_ood(
            net,
            id_loader,
            ood_loaders,
            upstream.ActivatedNegPostprocessor(config),
            "nearood",
        )
    conf, labels, pred = metrics.call_args.args
    with np.load(experiment / "scores_seed0.npz") as saved:
        aligned = np.array_equal(labels, saved["label"])
        pred_agree = np.array_equal(pred, saved["pred"])
        abs_diff = np.abs(conf.astype(np.float32) - saved["conf"].astype(np.float32))
    if not aligned or len(labels) != len(thresholds) * batch_size:
        raise ValueError("Upstream replay did not align with the recorded stream")
    positive = np.concatenate(positive_batches)
    negative = np.concatenate(negative_batches)
    fpr, auroc, _, _, _ = compute_all_metrics(conf, labels, pred)
    report = {
        "upstream_commit": UPSTREAM,
        "samples": len(labels),
        "batch_size": batch_size,
        "labels_identical": aligned,
        "predictions_identical": pred_agree,
        "unequal_confidences": int(np.count_nonzero(abs_diff)),
        "max_abs_confidence_difference": float(abs_diff.max()),
        "upstream_fpr95_pct": float(100 * fpr),
        "upstream_auroc_pct": float(100 * auroc),
        "median_queue_threshold": float(np.median(thresholds)),
        "pseudo_id_enqueued": int(positive.sum()),
        "true_id_among_pseudo_id": int(np.count_nonzero(positive & (labels >= 0))),
        "pseudo_ood_enqueued": int(negative.sum()),
        "true_ood_among_pseudo_ood": int(np.count_nonzero(negative & (labels == -1))),
    }
    (experiment / "upstream_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)
    # The CIFAR launcher lists these four fixed thresholds. In the published
    # postprocessor the dynamic estimate replaces the configured value.
    fixed = []
    for value in (0.01, 0.1, 0.3, 0.5):
        torch.manual_seed(0)
        with (
            patch.object(upstream, "find_best_threshold", return_value=value),
            patch.object(
                fsood_evaluator_clip, "compute_all_metrics", wraps=compute_all_metrics
            ) as fixed_metrics,
        ):
            evaluator._eval_ood(
                net,
                id_loader,
                ood_loaders,
                upstream.ActivatedNegPostprocessor(config),
                "nearood",
            )
        fixed_conf, fixed_labels, fixed_pred = fixed_metrics.call_args.args
        if not np.array_equal(labels, fixed_labels) or not np.array_equal(
            pred, fixed_pred
        ):
            raise ValueError("Fixed-threshold audit used a different test stream")
        fixed_fpr, fixed_auc, _, _, _ = compute_all_metrics(fixed_conf, labels, pred)
        fixed.append(
            {
                "threshold": value,
                "fpr95_pct": 100 * float(fixed_fpr),
                "auroc_pct": 100 * float(fixed_auc),
            }
        )
        print("Fixed-threshold diagnostic", fixed[-1], flush=True)
    (experiment / "fixed_thresholds.json").write_text(
        json.dumps(fixed, indent=2) + "\n"
    )

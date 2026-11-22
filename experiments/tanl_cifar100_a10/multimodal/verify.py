"""Verify final-score identities and compare native raw-image runs with frozen replays."""

import argparse
from unittest.mock import patch

import numpy as np
import torch
from ablation import write_rows
from capture import load_image_network, uniform_bank
from evaluate import scores_path
from prepare_views import FEATURES, HERE, study_config
from run import set_seed

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import fsood_evaluator_clip
from openood.evaluators.fsood_evaluator_clip import OODEvaluatorClipTTA
from openood.networks import get_network
from openood.networks.clip_fixed_ood_prompt import get_selected_ood_text_list
from openood.postprocessors import get_postprocessor
from openood.utils.config import Config, merge_configs

RAW_PAIRS = {"cifar10": "nearood", "svhn": "farood", "tin": "nearood"}
CHECK_DATASETS = {
    "combined": ("cifar10", "svhn"),
    "selected": ("cifar10", "svhn"),
    "dual_view": ("tin",),
}


def verify_equations() -> None:
    config = study_config()
    classes, negatives = 6, 11
    rng = torch.Generator().manual_seed(101)
    logits = torch.randn(19, classes + negatives, dtype=torch.float64, generator=rng)
    args = config.postprocessor.postprocessor_args
    args.beta = negatives
    for step in (0, 1, 4):
        args.gamma = step
        ends = [negatives] if step == 0 else range(step, negatives + step, step)
        id_weights = logits[:, :classes].exp()
        neg_weights = logits[:, classes:].exp()
        total = id_weights.sum(1)
        for k in (1, 3, classes):
            top = id_weights.topk(k, dim=1).values.sum(1)
            for mode in ("joint", "compete"):
                args.id_evidence_topk, args.id_evidence_mode = k, mode
                post = get_postprocessor(config)
                score = post._final_confidence(logits, classes)
                denom = total if mode == "joint" else top
                direct = torch.stack(
                    [top / (denom + neg_weights[:, :end].sum(1)) for end in ends]
                ).mean(0)
                torch.testing.assert_close(score, direct, atol=2e-14, rtol=2e-14)
                if k == classes:
                    assert torch.equal(
                        score, post._score_prefixes(logits, classes, negatives, step)
                    )
    del args["id_evidence_topk"]
    args.id_evidence_mode, args.top_class_score = "joint", True
    legacy = get_postprocessor(config)
    for data in (logits.half(), logits):
        expected = legacy._score_prefixes(data, classes, negatives, step).float()
        expected *= data[:, :classes].float().softmax(dim=1).amax(dim=1)
        actual = legacy._final_confidence(data, classes)
        assert actual.dtype == expected.dtype and torch.equal(actual, expected)
    print("Verified both score equations, complete-class endpoints and legacy top-1.")


@torch.no_grad()
def capture_images(config: Config, case: str, prompt: str, head: str) -> None:
    id_loader, loaders = get_dataloader(config)["test"], get_ood_dataloader(config)
    if case == "combined":
        # Build the text bank from scratch independently, with the original noise seed.
        get_selected_ood_text_list("cifar100", 1000)
        set_seed(0)
        net = get_network(config.network).eval()
        if not torch.equal(
            net.text_features_all.cpu(), torch.load(FEATURES / f"{prompt}.pt")
        ):
            raise ValueError("Native network and cached ensemble text vectors differ")
    else:
        bank = uniform_bank()
        if prompt != "nice":
            bank["texts"] = torch.load(FEATURES / f"{prompt}.pt")
        net = load_image_network(config, bank)
    evaluator = OODEvaluatorClipTTA(config)
    for name in CHECK_DATASETS[case]:
        split = RAW_PAIRS[name]
        set_seed(0)
        post = get_postprocessor(config)
        selected = {split: {name: loaders[split][name]}}
        with patch.object(
            fsood_evaluator_clip,
            "compute_all_metrics",
            wraps=fsood_evaluator_clip.compute_all_metrics,
        ) as metrics:
            evaluator._eval_ood(net, id_loader, selected, post, split)
        conf, label, pred = metrics.call_args.args
        path = FEATURES / f"raw_{name}_{case}.npz"
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, pred=pred, label=label, **{head: conf})
        temp.replace(path)
        print(f"Captured {len(label)} native scores on {name}", flush=True)


def compare_images(case: str, prompt: str, view: str, head: str) -> None:
    checks = []
    for name in CHECK_DATASETS[case]:
        path = scores_path(name, 0, prompt, view)
        with (
            np.load(path) as saved,
            np.load(FEATURES / f"raw_{name}_{case}.npz") as native,
        ):
            for key in ("label", "pred", head):
                np.testing.assert_array_equal(native[key], saved[key])
            count = len(native["label"])
        checks.append(
            {
                "dataset": name,
                "seed": 0,
                "prompt": prompt,
                "view": view,
                "head": head,
                "exact_samples": count,
            }
        )
        print(f"Verified {count} raw-image scores on {name}", flush=True)
    write_rows(HERE / f"{case}_raw_checks.csv", checks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CHECK_DATASETS, default="combined")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="rebuild CLIP and score both raw-image streams",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="compare native and replayed per-image outputs",
    )
    options = parser.parse_args()
    verify_equations()
    if options.raw or options.compare:
        config = merge_configs(
            study_config(), Config(str(HERE / f"{options.case}.yml"))
        )
        prompt, view = (
            config.network.backbone.text_prompt,
            config.network.backbone.image_view,
        )
        args = config.postprocessor.postprocessor_args
        k = args.get("id_evidence_topk")
        head = "all" if k is None else f"{args.id_evidence_mode}{k}"
        if options.raw:
            capture_images(config, options.case, prompt, head)
        if options.compare:
            compare_images(options.case, prompt, view, head)

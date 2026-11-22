"""Paired image, text and final-score ablations on frozen CLIP features."""

import argparse
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
import yaml
from ablation import write_rows
from capture import uniform_bank
from diagnose import FeatureBank
from prepare_views import FEATURES, HERE, OOD_COUNTS, PREVIOUS, study_config
from run import SEEDS
from torch import nn

from openood.evaluators.metrics import compute_all_metrics
from openood.networks.clip_fixed_ood_prompt import average_image_views
from openood.postprocessors.activated_neg_postprocessor import (
    competing_id_score,
    topk_id_mass,
)
from openood.postprocessors.adaptive_tanl_postprocessor import AdaptiveTANLPostprocessor
from openood.utils.config import Config

VIEWS_BY_PROMPT = {
    "nice": ("original", "flip", "mean"),
    "vanilla": ("original",),
    "nice_vanilla": ("original", "mean"),
}
HEADS = ("all", "joint1", "joint3", "compete1", "compete3")


def scores_path(dataset: str, seed: int, prompt: str, view: str) -> Path:
    return FEATURES / f"{dataset}_{seed}_{prompt}_{view}.npz"


class PairedScores(AdaptiveTANLPostprocessor):
    """Only the last readout varies across heads. All five share one queue trajectory."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.extra_scores = defaultdict(list)
        self.gates = []

    def _admission_masks(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos, neg = super()._admission_masks(scores)
        self.gates.append(pos.to(torch.int8) - neg.to(torch.int8))
        return pos, neg

    def _final_confidence(self, output: torch.Tensor, class_num: int) -> torch.Tensor:
        full = super()._final_confidence(output, class_num)
        for k in (1, 3):
            joint = full * topk_id_mass(output[:, :class_num], k)
            compete = competing_id_score(output, class_num, k, self.beta, self.gamma)
            self.extra_scores[f"joint{k}"].append(joint)
            self.extra_scores[f"compete{k}"].append(compete)
        return full


@torch.no_grad()
def replay(
    config: Config, net: nn.Module, stream: dict[str, torch.Tensor], view: str
) -> dict[str, np.ndarray]:
    post = PairedScores(config)
    post.reset_memory()
    torch.set_rng_state(stream["rng"])
    pred, full = [], []
    batch_size = config.dataset.test.batch_size
    features = stream["flip"] if view == "flip" else stream["images"]
    for index in range(0, len(features), batch_size):
        batch = slice(index, index + batch_size)
        images = features[batch].cuda()
        if view == "mean":
            images = average_image_views(images, stream["flip"][batch].cuda())
        classes, conf = post.postprocess(net, images)
        pred.append(classes)
        full.append(conf)
    outputs = {
        key: torch.cat(parts).cpu().numpy() for key, parts in post.extra_scores.items()
    }
    outputs.update(
        all=torch.cat(full).cpu().numpy(),
        pred=torch.cat(pred).cpu().numpy(),
        gate=torch.cat(post.gates).cpu().numpy(),
        label=stream["labels"].numpy(),
    )
    if not all(np.isfinite(outputs[key]).all() for key in HEADS):
        raise ValueError("Nonfinite final confidence")
    return outputs


def check_baseline(dataset: str, seed: int, outputs: dict[str, np.ndarray]) -> None:
    path = PREVIOUS / "scores" / f"{dataset}_uniform_positive_{seed}.npz"
    with np.load(path) as ref:
        for key in ("pred", "gate", "label", "conf"):
            np.testing.assert_array_equal(
                outputs["all" if key == "conf" else key], ref[key]
            )


def evaluate(
    datasets: Sequence[str], seeds: Sequence[int], prompts: Sequence[str]
) -> None:
    config = study_config()
    bank = uniform_bank()
    count = config.postprocessor.postprocessor_args.beta
    for prompt in prompts:
        views = VIEWS_BY_PROMPT[prompt]
        texts = (
            bank["texts"] if prompt == "nice" else torch.load(FEATURES / f"{prompt}.pt")
        )
        net = FeatureBank({**bank, "texts": texts}, count)
        for dataset in datasets:
            for seed in seeds:
                stream = torch.load(
                    FEATURES / f"{dataset}_{seed}_views.pt", map_location="cpu"
                )
                for view in views:
                    outputs = replay(config, net, stream, view)
                    if dataset != "svhn" and (prompt, view) == ("nice", "original"):
                        check_baseline(dataset, seed, outputs)
                    path = scores_path(dataset, seed, prompt, view)
                    temp = path.with_suffix(".tmp.npz")
                    np.savez_compressed(temp, **outputs)
                    temp.replace(path)
                print(f"Scored {dataset}, seed {seed}, prompt {prompt}", flush=True)


def summarize(*, select_known: bool) -> None:
    rows, grouped = [], defaultdict(list)
    datasets = ("cifar10", "tin", "texture") if select_known else OOD_COUNTS
    for dataset in datasets:
        for prompt, view in (
            (p, v) for p, views in VIEWS_BY_PROMPT.items() for v in views
        ):
            for seed in SEEDS:
                with np.load(scores_path(dataset, seed, prompt, view)) as data:
                    for head in HEADS:
                        fpr, auc, _, _, acc = compute_all_metrics(
                            data[head], data["label"], data["pred"]
                        )
                        row = {
                            "dataset": dataset,
                            "prompt": prompt,
                            "view": view,
                            "head": head,
                            "seed": seed,
                            "fpr95": 100 * fpr,
                            "auroc": 100 * auc,
                            "id_accuracy": 100 * acc,
                        }
                        rows.append(row)
                        grouped[dataset, prompt, view, head].append(row)
    summary = []
    for (dataset, prompt, view, head), values in grouped.items():
        row = {
            "dataset": dataset,
            "prompt": prompt,
            "view": view,
            "head": head,
            "seeds": len(values),
        }
        for key in ("fpr95", "auroc", "id_accuracy"):
            samples = [v[key] for v in values]
            row[f"{key}_mean"] = mean(samples)
            row[f"{key}_sd"] = stdev(samples)
        summary.append(row)
    if select_known:
        by_method = defaultdict(list)
        for row in summary:
            by_method[row["prompt"], row["view"], row["head"]].append(row["fpr95_mean"])
        prompt, view, head = min(
            by_method, key=lambda method: (mean(by_method[method]), method)
        )
        mode = "compete" if head.startswith("compete") else "joint"
        k = None if head == "all" else int(head.removeprefix(mode))
        chosen = {
            "network": {"backbone": {"text_prompt": prompt, "image_view": view}},
            "postprocessor": {
                "postprocessor_args": {"id_evidence_topk": k, "id_evidence_mode": mode}
            },
        }
        (HERE / "selected.yml").write_text(yaml.safe_dump(chosen, sort_keys=False))
        print(
            f"Selected {prompt}/{view}/{head}; known mean FPR95 {mean(by_method[prompt, view, head]):.4f}"
        )
    else:
        write_rows(HERE / "seeds.csv", rows)
        write_rows(HERE / "summary.csv", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=OOD_COUNTS, default=list(OOD_COUNTS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument(
        "--prompts", nargs="+", choices=VIEWS_BY_PROMPT, default=list(VIEWS_BY_PROMPT)
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--summarize", action="store_true", help="read all completed score files"
    )
    actions.add_argument(
        "--select-known",
        action="store_true",
        help="select by mean FPR95 on CIFAR-10, TIN and DTD only",
    )
    options = parser.parse_args()
    if options.summarize or options.select_known:
        summarize(select_known=options.select_known)
    else:
        evaluate(options.datasets, options.seeds, options.prompts)

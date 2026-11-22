"""Replay controlled TANL interventions on the same frozen, verified CIFAR streams."""

import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from capture_features import CACHE, diagnostic_config
from run import EXPERIMENT, SEEDS
from torch import nn

from openood.evaluators.metrics import compute_all_metrics, fpr_recall
from openood.networks.clip import clip
from openood.networks.clip_fixed_ood_prompt import (
    get_class_names,
    get_selected_ood_text_list,
    load_clip_to_cpu,
)
from openood.postprocessors import activated_neg_postprocessor as tanl

CASES = (
    "released",
    "unweighted_variance",
    "image_history",
    "zero_gap",
    "image_history_zero_gap",
    "paper_initialization",
    "paper_initialization_unweighted",
    "past_true_queues",
    "past_true_id",
    "past_true_ood",
    "static_gate",
    "final_score_updates",
    "imagenet_launcher",
    "uniform_prompt",
    "mixed_no_period",
    "uniform_no_period",
    "paper_text_and_equations",
    "strip_brackets",
    "float32_initial_activations",
    "labels_10",
    "labels_100",
    "labels_500",
)


class FeatureBank(nn.Module):
    def __init__(self, bank: dict, count: int):
        super().__init__()
        self.n_cls = bank["id_classes"]
        self.logit_scale = bank["scale"].cuda()
        self.noise_image_features = bank["noise"].cuda()
        self.words = bank["words"]
        self.text_features_all = bank["texts"].cuda()
        if count != bank["selected"] - self.n_cls:
            groups = get_selected_ood_text_list("cifar100", count)
            words = [word for group in groups for word in group]
            positions = {word: i + self.n_cls for i, word in enumerate(self.words)}
            indices = [*range(self.n_cls), *(positions[word] for word in words)]
            self.text_features_all = self.text_features_all[:, indices].contiguous()
            self.words = words
        self.text_features = self.text_features_all[
            :, : self.n_cls + count
        ].contiguous()

    def forward(self, images, return_feat=True):
        return images, self.text_features.T, self.logit_scale


def gaussian_features() -> torch.Tensor:
    model = load_clip_to_cpu("ViT-B/16").cuda().eval()
    rng = torch.Generator().manual_seed(0)
    noise = torch.randn((300, 3, 224, 224), generator=rng)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    features = torch.cat(
        [model.encode_image(batch.cuda()) for batch in noise.split(15)]
    )
    return features / features.norm(dim=-1, keepdim=True)


def prompt_bank(bank, case, model):
    """Replace only the named prompts, preserving class and candidate order."""
    corpus = torch.load(
        "data/txtfiles_output/wordnet_cifar100_cossim_dedup.pth", map_location="cpu"
    )
    adjectives = set(corpus["text_list_adj"])
    texts = bank["texts"].clone()
    class_names = get_class_names("cifar100")
    for i, word in enumerate([*class_names, *bank["words"]]):
        adj = i >= bank["id_classes"] and word in adjectives
        if case == "mixed_no_period":
            prompt = f"This is a {word} photo" if adj else f"The nice {word}"
        elif case == "uniform_no_period" and adj:
            # The base is mixed_no_period, so the nouns and ID texts already match.
            prompt = f"The nice {word}"
        elif case == "uniform_prompt" and adj:
            prompt = f"The nice {word}."
        elif case == "strip_brackets" and ("[" in word or "]" in word):
            cleaned = word.replace("[", "").replace("]", "")
            prompt = (
                f"This is a {cleaned} photo."
                if word in adjectives
                else f"The nice {cleaned}."
            )
        else:
            continue
        vector = model.encode_text(clip.tokenize(prompt).cuda())
        vector /= vector.norm(dim=-1, keepdim=True)
        vector /= vector.norm(dim=-1, keepdim=True)
        texts[:, i] = vector[0].cpu()
    return {**bank, "texts": texts}


def unweighted_variance(history, threshold):
    above = history >= threshold
    high, low = history[above], history[~above]
    if high.numel() == 0 or low.numel() == 0:
        return history.new_tensor(float("inf"))
    return high.var(unbiased=False) + low.var(unbiased=False)


def quantiles(values):
    return np.quantile(values, [0, 0.05, 0.5, 0.95, 1]).tolist()


@torch.no_grad()
def replay(case, seed, bank, stream, reference, gaussian):
    config = diagnostic_config()
    args = config.postprocessor.postprocessor_args
    if case.startswith("labels_"):
        args.beta = int(case.split("_")[1])
    if "zero_gap" in case:
        args.gap = 0.0
    if case == "imagenet_launcher":
        args.alpha = 0.0
        args.gap = 0.5
    combined = case == "paper_text_and_equations"
    paper_init = case.startswith("paper_initialization") or combined
    net = FeatureBank(bank, args.beta)
    if paper_init:
        net.noise_image_features = gaussian
    post = tanl.ActivatedNegPostprocessor(config)
    if case == "static_gate":
        post.mute_mutual_enhancement = True
    final_updates = case == "final_score_updates" or combined
    change_updates = case.startswith("past_true") or final_updates
    batch_size = config.dataset.test.batch_size
    labels = stream["labels"].numpy()
    torch.set_rng_state(stream["rng"])
    thresholds, gate_scores, confidences, predictions = [], [], [], []
    snapshots = {}
    original_threshold = tanl.find_best_threshold
    original_update = tanl.update_queue
    original_softmax = torch.softmax

    def softmax(values, *args, **kwargs):
        if (
            case == "float32_initial_activations"
            and post.reset
            and values.dtype == torch.float16
        ):
            values = values.float()
        return original_softmax(values, *args, **kwargs)

    variance = (
        unweighted_variance
        if "unweighted" in case or combined
        else tanl.compute_os_variance_tensor
    )
    init_order = torch.randperm(300, generator=torch.Generator().manual_seed(0))

    def init_queue(to_add, queue, max_len=1000, init=False):
        if paper_init and init and len(queue) == net.n_cls:
            queue = queue.repeat((3, 1))[init_order]
        return original_update(to_add, queue, max_len, init)

    def score_threshold(history):
        step = len(thresholds)
        if paper_init and step == 0:
            ids = history[: net.n_cls].repeat(3)[init_order]
            history = torch.cat((ids, history[net.n_cls :]))
            post.scoretensor = history
        seen = (step + 1) * batch_size
        observed = history[-seen:]
        threshold = original_threshold(observed if "image_history" in case else history)
        batch = history[-batch_size:].cpu().numpy()
        thresholds.append(threshold)
        gate_scores.append(batch)
        if change_updates:
            # These references predate all updates from the current batch.
            before[:] = [
                post.score_from_pos,
                post.score_from_neg,
                post.cached_pos_features,
                post.cached_neg_features,
            ]
        if step in (0, 9, 49, 98, 99):
            snapshot = {"threshold": threshold, "batch_scores": quantiles(batch)}
            if case == "released":
                snapshot["threshold_without_initial_scores"] = original_threshold(
                    observed
                )
                if step in (0, 49):
                    for label, queue in (
                        ("positive", post.score_from_pos),
                        ("negative", post.score_from_neg),
                    ):
                        snapshot[label + "_zero_fraction"] = (
                            (queue == 0).float().mean().item()
                        )
                        snapshot[label + "_mean_candidate_mass"] = (
                            queue.sum(dim=1).mean().item()
                        )
                ranking = (
                    post.score_from_neg.mean(0) - post.score_from_pos.mean(0)
                ).argsort(descending=True)
                snapshot["top_negative_words_before_batch"] = [
                    net.words[i] for i in ranking[:10].tolist()
                ]
            if step == 0:
                prior = history[:-batch_size].cpu().numpy()
                n_pos = len(prior) - len(net.noise_image_features)
                snapshot["text_prior"] = quantiles(prior[:n_pos])
                snapshot["noise_prior"] = quantiles(prior[n_pos:])
                snapshot["prior_counts"] = [n_pos, len(prior) - n_pos]
            snapshots[str(step + 1)] = snapshot
        return threshold

    before = []
    with (
        patch.object(tanl, "find_best_threshold", score_threshold),
        patch.object(tanl, "compute_os_variance_tensor", variance),
        patch.object(tanl, "update_queue", init_queue),
        patch.object(torch, "softmax", softmax),
    ):
        for offset, images in enumerate(stream["images"].split(batch_size)):
            images = images.cuda()
            pred, conf = post.postprocess(net, images)
            predictions.append(pred.cpu())
            confidences.append(conf.cpu())
            if change_updates:
                logits = net.logit_scale * images @ net.text_features_all
                acts = logits.float().softmax(dim=1)[:, net.n_cls :]
                if final_updates:
                    ids = conf > post.thres + post.gap * (1 - post.thres)
                    oods = conf < post.thres * (1 - post.gap)
                    post.scoretensor[-len(conf) :] = conf.to(post.scoretensor.dtype)
                else:
                    # Score first. Only future batches can see these labels.
                    start = offset * batch_size
                    ids = stream["labels"][start : start + len(images)].cuda() >= 0
                    oods = ~ids
                if case != "past_true_ood":
                    post.score_from_pos = original_update(
                        acts[ids], before[0], post.queue_len
                    )
                    post.cached_pos_features = original_update(
                        images[ids], before[2], post.queue_len
                    )
                if case != "past_true_id":
                    post.score_from_neg = original_update(
                        acts[oods], before[1], post.queue_len
                    )
                    post.cached_neg_features = original_update(
                        images[oods], before[3], post.queue_len
                    )
    conf = torch.cat(confidences).numpy()
    pred = torch.cat(predictions).numpy()
    if not np.array_equal(labels, reference["label"]):
        raise ValueError("The controlled replay changed the sample alignment")
    if case.endswith("no_period") or combined:
        expected = torch.cat(
            [
                (net.logit_scale * b.cuda() @ net.text_features)[:, : net.n_cls]
                .argmax(1)
                .cpu()
                for b in stream["images"].split(batch_size)
            ]
        )
    else:
        expected = reference["pred"]
    if not np.array_equal(pred, expected):
        raise ValueError(
            "The controlled replay changed the sample alignment or ID classifier"
        )
    if case == "released" and not np.array_equal(conf, reference["conf"]):
        raise ValueError(
            f"The released replay changed scores by up to {np.max(np.abs(conf - reference['conf']))}"
        )
    fpr, auroc, _, _, acc = compute_all_metrics(conf, labels, pred)
    id_positive_fpr, _ = fpr_recall(conf, labels, 0.95)
    gate = np.concatenate(gate_scores)
    t = np.repeat(thresholds, batch_size)
    positive = gate > t + args.gap * (1 - t)
    negative = gate < t - args.gap * t
    return {
        "seed": seed,
        "fpr95": 100 * fpr,
        "auroc": 100 * auroc,
        "fpr95_id_positive": 100 * id_positive_fpr,
        "id_acc": 100 * acc,
        "scale": float(net.logit_scale),
        "threshold_median": float(np.median(thresholds)),
        "threshold_range": [min(thresholds), max(thresholds)],
        "gate_id": quantiles(gate[labels >= 0]),
        "gate_ood": quantiles(gate[labels < 0]),
        "positive_gate": int(positive.sum()),
        "true_id_positive_gate": int(np.count_nonzero(positive & (labels >= 0))),
        "negative_gate": int(negative.sum()),
        "true_ood_negative_gate": int(np.count_nonzero(negative & (labels < 0))),
        "snapshots": snapshots,
    }


def summarize(output: Path) -> None:
    paths = [output / f"seed{seed}.json" for seed in SEEDS]
    if not all(path.exists() for path in paths):
        return
    runs = [json.loads(path.read_text()) for path in paths]
    fields = ("fpr95", "auroc", "fpr95_id_positive", "id_acc")
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                *(f"{field}_{stat}_pct" for field in fields for stat in ("mean", "sd")),
            ]
        )
        for case in CASES:
            if all(set(fields).issubset(run.get(case, {})) for run in runs):
                values = np.array(
                    [[run[case][field] for field in fields] for run in runs]
                )
                summary = np.column_stack(
                    (values.mean(axis=0), values.std(axis=0, ddof=1))
                )
                writer.writerow([case, *summary.flatten()])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=SEEDS)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    options = parser.parse_args()
    bank = torch.load(CACHE / "bank.pt", map_location="cpu")
    alternates = {}
    prompt_cases = set(options.cases) & {
        "uniform_prompt",
        "strip_brackets",
        "mixed_no_period",
        "uniform_no_period",
    }
    if "paper_text_and_equations" in options.cases:
        prompt_cases.add("uniform_no_period")
    if prompt_cases:
        model = load_clip_to_cpu("ViT-B/16").cuda().eval()
        if "uniform_no_period" in prompt_cases:
            prompt_cases.add("mixed_no_period")
        for case in sorted(prompt_cases):
            base = (
                alternates["mixed_no_period"] if case == "uniform_no_period" else bank
            )
            alternates[case] = prompt_bank(base, case, model)
        del model
        if "paper_text_and_equations" in options.cases:
            alternates["paper_text_and_equations"] = alternates["uniform_no_period"]
    gaussian = (
        gaussian_features()
        if any(c.startswith("paper_") for c in options.cases)
        else None
    )
    output = EXPERIMENT / "diagnostics"
    output.mkdir(exist_ok=True)
    for seed in options.seeds:
        stream = torch.load(CACHE / f"stream{seed}.pt", map_location="cpu")
        path = output / f"seed{seed}.json"
        results = json.loads(path.read_text()) if path.exists() else {}
        with np.load(EXPERIMENT / f"scores_seed{seed}.npz") as reference:
            for case in options.cases:
                result = replay(
                    case, seed, alternates.get(case, bank), stream, reference, gaussian
                )
                results[case] = result
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(results, indent=2) + "\n")
                tmp.replace(path)
                print(
                    seed,
                    case,
                    "FPR95",
                    result["fpr95"],
                    "AUROC",
                    result["auroc"],
                    "gate",
                    result["positive_gate"],
                    result["negative_gate"],
                    flush=True,
                )
    summarize(output)


if __name__ == "__main__":
    main()

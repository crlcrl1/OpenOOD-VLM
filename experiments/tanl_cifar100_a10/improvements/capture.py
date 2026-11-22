"""Run the quantile gate on raw images and save independent references for replays."""

import argparse
import json
import shutil
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch
from capture_features import CACHE, CaptureStream, diagnostic_config
from prepare_inputs import DATA, DOWNLOADS, fetch_inputs
from run import SEEDS, set_seed

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators.fsood_evaluator_clip import OODEvaluatorClipTTA
from openood.evaluators.metrics import compute_all_metrics
from openood.networks import clip_fixed_ood_prompt, get_network
from openood.postprocessors.adaptive_tanl_postprocessor import AdaptiveTANLPostprocessor
from openood.utils.config import Config, merge_configs

HERE = Path(__file__).resolve().parent


def method_config():
    return merge_configs(diagnostic_config(), Config(str(HERE / "quantile.yml")))


def prepare_ood_split(dataset, count):
    fetch_inputs(HERE / f"{dataset}_sources.tsv")
    source_list = DOWNLOADS / f"test_{dataset}.txt"
    members = []
    for row in source_list.read_text().splitlines():
        name, label = row.split()
        if not name.startswith(dataset + "/") or label != "-1":
            raise ValueError(f"Unexpected {dataset} sample {row}")
        members.append(name.removeprefix(dataset + "/"))
    if len(set(members)) != count or len(members) != count:
        raise ValueError(f"Unexpected {dataset} test split")
    with ZipFile(DOWNLOADS / f"{dataset}.zip") as archive:
        for member in members:
            archive.extract(member, DATA / "images_classic" / dataset)
    shutil.copyfile(
        source_list, DATA / "benchmark_imglist" / "cifar100" / source_list.name
    )


def uniform_bank():
    bank = torch.load(CACHE / "bank.pt", map_location="cpu")
    bank["texts"] = torch.load(CACHE / "uniform_texts.pt", map_location="cpu")
    return bank


def load_image_network(config, bank):
    """Build the real frozen image encoder using the previously verified text vectors."""
    set_seed(0)
    selected = bank["selected"]
    text = bank["texts"].cuda()
    with patch.object(
        clip_fixed_ood_prompt,
        "get_text_features_neg",
        return_value=(text[:, :selected].contiguous(), text[:, selected:].contiguous()),
    ):
        net = get_network(config.network).eval()
    if not torch.equal(net.noise_image_features.cpu(), bank["noise"]):
        raise ValueError("The raw run has different initialization")
    return net


class CaptureQuantiles(CaptureStream, AdaptiveTANLPostprocessor):
    """Use the existing evaluation/feature recorder with the new postprocessor."""


def check_positive_images(net, config):
    config = merge_configs(config, Config(str(HERE / "positive.yml")))
    config.ood_dataset.nearood.datasets = ["cifar10", "tin"]
    id_loader = get_dataloader(config)["test"]
    ood = get_ood_dataloader(config)["nearood"]
    evaluator = OODEvaluatorClipTTA(config)
    checks = []
    for dataset in config.ood_dataset.nearood.datasets:
        set_seed(0)
        post = CaptureQuantiles(config)
        evaluator._eval_ood(net, id_loader, {"nearood": {dataset: ood[dataset]}}, post)
        pred, conf, label = post.scores
        with np.load(HERE / "scores" / f"{dataset}_uniform_positive_0.npz") as saved:
            for key, values in (("pred", pred), ("conf", conf), ("label", label)):
                if not np.array_equal(values, saved[key]):
                    raise ValueError(
                        f"Positive gate raw/replay {key} mismatch on {dataset}"
                    )
        fpr, auc, _, _, _ = compute_all_metrics(conf, label, pred)
        checks.append(
            {
                "dataset": dataset,
                "seed": 0,
                "exact_score_matches": len(label),
                "fpr95": 100 * fpr,
                "auroc": 100 * auc,
            }
        )
    (HERE / "positive_raw_check.json").write_text(json.dumps(checks, indent=2) + "\n")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visual",
        action="store_true",
        help="check visual scoring on seed 0 of each pair",
    )
    options = parser.parse_args()
    prepare_ood_split("tin", 6526)
    config = method_config()
    if options.visual:
        config = merge_configs(config, Config(str(HERE / "visual.yml")))
    config.ood_dataset.nearood.datasets = ["cifar10", "tin"]
    id_loader = get_dataloader(config)["test"]
    ood_loaders = get_ood_dataloader(config)["nearood"]
    set_seed(0)
    net = get_network(config.network).eval()
    base = torch.load(CACHE / "bank.pt", map_location="cpu")
    if not torch.equal(net.noise_image_features.cpu(), base["noise"]):
        raise ValueError("Initial noise differs from the reproduced baseline")
    uniform = net.text_features_all.cpu()
    if not torch.equal(uniform[:, : net.n_cls], base["texts"][:, : net.n_cls]):
        raise ValueError("Uniform template changed the ID classifier")
    if options.visual:
        if not torch.equal(
            uniform, torch.load(CACHE / "uniform_texts.pt", map_location="cpu")
        ):
            raise ValueError("Text bank changed across raw runs")
    else:
        torch.save(uniform, CACHE / "uniform_texts.pt")
    evaluator = OODEvaluatorClipTTA(config)
    for dataset, seeds in (
        ("cifar10", (0,)),
        ("tin", (0,) if options.visual else SEEDS),
    ):
        for seed in seeds:
            images = []

            def record(module, args, output, target=images):
                target.append(output[0].detach().cpu())

            set_seed(seed)
            post = CaptureQuantiles(config)
            with net.register_forward_hook(record):
                evaluator._eval_ood(
                    net, id_loader, {"nearood": {dataset: ood_loaders[dataset]}}, post
                )
            pred, conf, label = post.scores
            stream = {
                "images": torch.cat(images),
                "labels": torch.from_numpy(label),
                "rng": post.cpu_rng,
            }
            prefix = "" if dataset == "cifar10" else f"{dataset}_"
            stream_path = CACHE / f"{prefix}stream{seed}.pt"
            if dataset == "cifar10" or options.visual:
                prior = torch.load(stream_path, map_location="cpu")
                if not all(torch.equal(stream[key], prior[key]) for key in stream):
                    raise ValueError(
                        f"{dataset} stream, image features or RNG differ from baseline"
                    )
            else:
                torch.save(stream, stream_path)
            method = "visual_quantiles" if options.visual else "quantiles"
            np.savez_compressed(
                CACHE / f"{dataset}_raw_{method}{seed}.npz",
                pred=pred,
                conf=conf,
                label=label,
            )
            print(
                f"Saved {dataset}, seed {seed}, {len(label)} raw image scores",
                flush=True,
            )


if __name__ == "__main__":
    main()

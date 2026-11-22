"""Capture both image views in the exact batches used by the OpenOOD evaluator."""

import argparse
from pathlib import Path

import numpy as np
import torch
from capture import HERE as PREVIOUS
from capture import load_image_network, method_config, prepare_ood_split, uniform_bank
from capture_features import CACHE
from run import SEEDS, set_seed
from torch.utils.data import ConcatDataset, DataLoader

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.utils.config import Config, merge_configs

HERE = Path(__file__).resolve().parent
FEATURES = CACHE / "multimodal"
OOD_COUNTS = {"cifar10": 10000, "tin": 6526, "texture": 5640, "svhn": 26032}


def study_config() -> Config:
    config = merge_configs(method_config(), Config(str(PREVIOUS / "positive.yml")))
    config.ood_dataset.split_names = ["nearood", "farood"]
    config.ood_dataset.nearood.datasets = ["cifar10", "tin"]
    config.ood_dataset.farood.datasets = ["texture", "svhn"]
    return config


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=OOD_COUNTS, default=list(OOD_COUNTS)
    )
    options = parser.parse_args()
    for name in ("tin", "texture", "svhn"):
        prepare_ood_split(name, OOD_COUNTS[name])
    config = study_config()
    id_loader = get_dataloader(config)["test"]
    loaders = get_ood_dataloader(config)
    net = load_image_network(config, uniform_bank())
    FEATURES.mkdir(exist_ok=True)
    for name in options.datasets:
        split = "nearood" if name in config.ood_dataset.nearood.datasets else "farood"
        mixed = ConcatDataset([loaders[split][name].dataset, id_loader.dataset])
        for seed in SEEDS:
            path = FEATURES / f"{name}_{seed}_views.pt"
            if path.exists():
                continue
            set_seed(seed)
            loader = DataLoader(
                mixed,
                batch_size=id_loader.batch_size,
                num_workers=id_loader.num_workers,
                shuffle=True,
            )
            original, mirrored, labels = [], [], []
            for batch in loader:
                if not original:
                    rng = torch.get_rng_state()
                x = batch["data"].cuda()
                original.append(net.encode_view(x).cpu())
                mirrored.append(net.encode_view(x.flip((-1,))).cpu())
                labels.append(batch["label"])
            stream = {
                "images": torch.cat(original),
                "flip": torch.cat(mirrored),
                "labels": torch.cat(labels),
                "rng": rng,
            }
            y = stream["labels"]
            if (int((y >= 0).sum()), int((y == -1).sum())) != (10000, OOD_COUNTS[name]):
                raise ValueError(f"Unexpected sample count on {name}")
            if name in ("cifar10", "tin"):
                prefix = "" if name == "cifar10" else "tin_"
                ref = torch.load(CACHE / f"{prefix}stream{seed}.pt", map_location="cpu")
                if any(not torch.equal(stream[key], ref[key]) for key in ref):
                    raise ValueError(
                        f"Original CLIP features, order or RNG differ: {name}, {seed}"
                    )
            elif name == "texture":
                with np.load(
                    PREVIOUS / "scores" / f"texture_uniform_positive_{seed}.npz"
                ) as ref:
                    np.testing.assert_array_equal(y.numpy(), ref["label"])
            temp = path.with_suffix(".tmp")
            torch.save(stream, temp)
            temp.replace(path)
            print(
                f"Saved {name}, seed {seed}, {len(y)} paired raw-image features",
                flush=True,
            )


if __name__ == "__main__":
    main()

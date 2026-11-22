"""Capture the exact public-run CLIP features for paired diagnostic replays."""

from pathlib import Path

import numpy as np
import torch
from run import CONFIGS, EXPERIMENT, SEEDS, ScoreRecorder, set_seed
from torch import nn

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators.fsood_evaluator_clip import OODEvaluatorClipTTA
from openood.networks import get_network
from openood.networks.clip_fixed_ood_prompt import get_selected_ood_text_list
from openood.utils.config import Config, merge_configs

CACHE = Path("data/tanl_a10_diagnostic_features")


def diagnostic_config() -> Config:
    config = merge_configs(
        *(Config(path) for path in CONFIGS), Config(str(EXPERIMENT / "protocol.yml"))
    )
    config.parse_refs()
    return config


class CaptureStream(ScoreRecorder):
    def postprocess(self, net: nn.Module, data: torch.Tensor):
        if self.reset:
            self.cpu_rng = torch.get_rng_state()
        return super().postprocess(net, data)


@torch.no_grad()
def main() -> None:
    CACHE.mkdir(exist_ok=True)
    config = diagnostic_config()
    id_loader = get_dataloader(config)["test"]
    ood_loaders = get_ood_dataloader(config)
    words = get_selected_ood_text_list(
        config.dataset.name, config.network.backbone.ood_number
    )
    set_seed(0)
    net = get_network(config.network).eval()
    torch.save(
        {
            "texts": net.text_features_all.cpu(),
            "selected": net.text_features.shape[1],
            "noise": net.noise_image_features.cpu(),
            "scale": net.logit_scale.cpu(),
            "id_classes": net.n_cls,
            "words": [word for group in words for word in group],
        },
        CACHE / "bank.pt",
    )
    evaluator = OODEvaluatorClipTTA(config)
    for seed in SEEDS:
        images = []

        def capture_image(module, args, output, target=images):
            target.append(output[0].detach().cpu())

        set_seed(seed)
        recorder = CaptureStream(config)
        with net.register_forward_hook(capture_image):
            evaluator._eval_ood(net, id_loader, ood_loaders, recorder, "nearood")
        pred, conf, label = recorder.scores
        with np.load(EXPERIMENT / f"scores_seed{seed}.npz") as saved:
            for key, value in (("pred", pred), ("conf", conf), ("label", label)):
                if not np.array_equal(value, saved[key]):
                    raise ValueError(f"Capture changed {key} at seed {seed}")
        torch.save(
            {
                "images": torch.cat(images),
                "labels": torch.from_numpy(label),
                "rng": recorder.cpu_rng,
            },
            CACHE / f"stream{seed}.pt",
        )
        print(f"Captured {len(label)} verified samples at seed {seed}", flush=True)


if __name__ == "__main__":
    main()

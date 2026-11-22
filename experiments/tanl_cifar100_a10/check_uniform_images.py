"""Independently run the single adjective-template change on the original PNGs."""

import json
from unittest.mock import patch

import numpy as np
import torch
from capture_features import CACHE, diagnostic_config
from diagnose import FeatureBank, prompt_bank
from run import EXPERIMENT, SEEDS, ScoreRecorder, set_seed

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators.fsood_evaluator_clip import OODEvaluatorClipTTA
from openood.evaluators.metrics import compute_all_metrics
from openood.networks import get_network
from openood.networks.clip import clip
from openood.postprocessors.activated_neg_postprocessor import ActivatedNegPostprocessor


@torch.no_grad()
def main():
    config = diagnostic_config()
    id_loader = get_dataloader(config)["test"]
    ood_loaders = get_ood_dataloader(config)
    tokenize = clip.tokenize
    replacements = 0

    def uniform_tokenizer(texts, *args, **kwargs):
        nonlocal replacements
        rewritten = []
        for text in [texts] if isinstance(texts, str) else texts:
            if text.startswith("This is a ") and text.endswith(" photo."):
                text = (
                    "The nice "
                    + text.removeprefix("This is a ").removesuffix(" photo.")
                    + "."
                )
                replacements += 1
            rewritten.append(text)
        return tokenize(rewritten, *args, **kwargs)

    set_seed(0)
    with patch.object(clip, "tokenize", uniform_tokenizer):
        net = get_network(config.network).eval()
    base = torch.load(CACHE / "bank.pt", map_location="cpu")
    alternative = prompt_bank(base, "uniform_prompt", net.model)
    if not torch.equal(net.text_features_all.cpu(), alternative["texts"]):
        raise ValueError("The fresh text encoder and replay text bank differ")
    if not torch.equal(net.noise_image_features.cpu(), base["noise"]):
        raise ValueError("The initial noise changed")
    cached = FeatureBank(alternative, config.postprocessor.postprocessor_args.beta)
    evaluator = OODEvaluatorClipTTA(config)
    report = {
        "rewritten_prompts": replacements,
        "all_text_vectors_identical": True,
        "runs": [],
    }
    for seed in SEEDS:
        set_seed(seed)
        real = ScoreRecorder(config)
        evaluator._eval_ood(net, id_loader, ood_loaders, real, "nearood")
        pred, conf, labels = real.scores
        stream = torch.load(CACHE / f"stream{seed}.pt", map_location="cpu")
        torch.set_rng_state(stream["rng"])
        post = ActivatedNegPostprocessor(config)
        scores = [
            post.postprocess(cached, b.cuda())[1].cpu()
            for b in stream["images"].split(id_loader.batch_size)
        ]
        if not np.array_equal(conf, torch.cat(scores).numpy()) or not np.array_equal(
            labels, stream["labels"].numpy()
        ):
            raise ValueError(
                f"The real-image and replay scores or sample order differ at seed {seed}"
            )
        fpr, auc, _, _, _ = compute_all_metrics(conf, labels, pred)
        report["runs"].append(
            {
                "seed": seed,
                "exact_score_matches": len(conf),
                "fpr95": 100 * fpr,
                "auroc": 100 * auc,
            }
        )
    path = EXPERIMENT / "diagnostics" / "uniform_image_run.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()

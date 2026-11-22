"""Check the prefix-sum formulation of Eq. 15 and time it on the reproduced workload."""

import json
from statistics import median
from time import perf_counter

import numpy as np
import torch
from capture import (
    HERE,
    check_positive_images,
    load_image_network,
    method_config,
    uniform_bank,
)
from capture_features import CACHE
from diagnose import FeatureBank
from run import SEEDS, ScoreRecorder, set_seed

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators.fsood_evaluator_clip import OODEvaluatorClipTTA
from openood.evaluators.metrics import compute_all_metrics
from openood.postprocessors.activated_neg_postprocessor import (
    ActivatedNegPostprocessor,
    activation_aware_score,
    prefix_activation_aware_score,
)


def timed(fn, *args):
    torch.cuda.synchronize()
    start = perf_counter()
    value = fn(*args)
    torch.cuda.synchronize()
    return value, perf_counter() - start


def fixed_logits(logits):
    rows = []
    for dtype in (torch.float16, torch.float32):
        x = logits.to(dtype)
        reference = activation_aware_score(x, 100, 1000, 1)
        fast = prefix_activation_aware_score(x, 100, 1000, 1)
        for step in (1, 7, 1003):
            expected = activation_aware_score(x.double(), 100, 1000, step)
            actual = prefix_activation_aware_score(x.double(), 100, 1000, step)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        times = {"loop": [], "prefix": []}
        scorers = [
            ("loop", activation_aware_score),
            ("prefix", prefix_activation_aware_score),
        ]
        for trial in range(15):
            for name, scorer in scorers if trial % 2 == 0 else reversed(scorers):
                _, sec = timed(scorer, x, 100, 1000, 1)
                times[name].append(1000 * sec)
        rows.append(
            {
                "dtype": str(dtype),
                "max_abs_error": float((reference - fast).abs().max()),
                "loop_median_ms": median(times["loop"]),
                "prefix_median_ms": median(times["prefix"]),
                "times_ms": times,
            }
        )
    return rows


def score_stream(post, net, images):
    preds, confs = [], []
    for batch in images.split(post.config.dataset.test.batch_size):
        pred, conf = post.postprocess(net, batch)
        preds.append(pred)
        confs.append(conf)
    return torch.cat(preds), torch.cat(confs)


@torch.no_grad()
def main():
    config = method_config()
    config.postprocessor.name = "actneg"
    bank = uniform_bank()
    frozen = FeatureBank(bank, config.postprocessor.postprocessor_args.beta)
    first = torch.load(CACHE / "stream0.pt", map_location="cpu")["images"][:200].cuda()
    logits = frozen.logit_scale * first @ frozen.text_features
    report = {
        "gpu": torch.cuda.get_device_name(),
        "fixed_logits": fixed_logits(logits),
        "stream": [],
        "raw": [],
    }

    seed0 = {}
    for seed in SEEDS:
        stream = torch.load(CACHE / f"stream{seed}.pt", map_location="cpu")
        images, labels = stream["images"].cuda(), stream["labels"].numpy()
        with np.load(HERE / "scores" / f"cifar10_uniform_baseline_{seed}.npz") as ref:
            for fast in (False, True) if seed % 2 == 0 else (True, False):
                config.postprocessor.postprocessor_args.prefix_score = fast
                torch.set_rng_state(stream["rng"])
                post = ActivatedNegPostprocessor(config)
                (pred, conf), sec = timed(score_stream, post, frozen, images)
                pred, conf = pred.cpu().numpy(), conf.cpu().numpy()
                if not np.array_equal(pred, ref["pred"]) or not np.array_equal(
                    labels, ref["label"]
                ):
                    raise ValueError("Classifier or sample order changed")
                if not fast and not np.array_equal(conf, ref["conf"]):
                    raise ValueError("Legacy scoring changed")
                fpr, auc, _, _, _ = compute_all_metrics(conf, labels, pred)
                err = np.abs(conf.astype(float) - ref["conf"])
                report["stream"].append(
                    {
                        "seed": seed,
                        "prefix": fast,
                        "seconds": sec,
                        "fpr95": 100 * fpr,
                        "auroc": 100 * auc,
                        "max_abs_error": float(err.max()),
                        "mean_abs_error": float(err.mean()),
                        "equal_scores": int((err == 0).sum()),
                    }
                )
                if seed == 0:
                    seed0[fast] = (pred, conf, labels)

    # Use the real CLIP, PNG loader and evaluator. Skip only repeated text encoding
    # in model setup, outside the timed interval, using the previously verified bank.
    net = load_image_network(config, bank)
    id_loader, ood = get_dataloader(config)["test"], get_ood_dataloader(config)
    evaluator = OODEvaluatorClipTTA(config)
    # Warm the image encoder before measuring. The ABBA order balances run position.
    net(torch.zeros(1, 3, 224, 224, device="cuda"), return_feat=True)
    for fast in (False, True, True, False):
        config.postprocessor.postprocessor_args.prefix_score = fast
        set_seed(0)
        post = ScoreRecorder(config)
        _, sec = timed(evaluator._eval_ood, net, id_loader, ood, post, "nearood")
        if not all(
            np.array_equal(a, b) for a, b in zip(seed0[fast], post.scores, strict=True)
        ):
            raise ValueError("PNG and cached-feature scores or sample order differ")
        report["raw"].append(
            {
                "prefix": fast,
                "seconds": sec,
                "exact_replay_matches": len(post.scores[1]),
            }
        )
    (HERE / "prefix_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Prefix benchmark complete", flush=True)
    check_positive_images(net, method_config())


if __name__ == "__main__":
    main()

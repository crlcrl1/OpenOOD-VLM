"""Screen two scoring changes on the frozen positive-gate TANL streams."""

import numpy as np
import torch
from ablation import DATASETS, summarize, write_rows
from capture import HERE, method_config, uniform_bank
from capture_features import CACHE
from diagnose import FeatureBank
from run import SEEDS

from openood.evaluators.metrics import compute_all_metrics
from openood.postprocessors.adaptive_tanl_postprocessor import AdaptiveTANLPostprocessor
from openood.utils.config import Config, merge_configs


class StabilityRanking(AdaptiveTANLPostprocessor):
    """Penalize one historical standard error at the final negative-label ranking."""

    def __init__(self, config):
        super().__init__(config)
        self.gates = []
        self.replacements = []

    def _admission_masks(self, scores):
        positive, negative = super()._admission_masks(scores)
        self.gates.append((positive.to(torch.int8) - negative.to(torch.int8)).cpu())
        return positive, negative

    def _rank_batch_negatives(self, scores, activations, count):
        pos = self.score_from_pos
        neg = self.score_from_neg
        # Both FIFOs start with at least 15 entries in this protocol. They still
        # include the original text/noise proxies during warm-up.
        variance = pos.var(0, correction=1) / len(pos) + neg.var(0, correction=1) / len(
            neg
        )
        selected = super()._rank_batch_negatives(
            scores - variance.sqrt(), activations, count
        )
        original = super()._rank_batch_negatives(scores, activations, count)
        self.replacements.append((~torch.isin(selected, original)).sum().item())
        return selected


@torch.no_grad()
def main():
    config = merge_configs(method_config(), Config(str(HERE / "positive.yml")))
    net = FeatureBank(uniform_bank(), config.postprocessor.postprocessor_args.beta)
    rows = []
    for dataset in DATASETS:
        prefix = "" if dataset == "cifar10" else f"{dataset}_"
        for seed in SEEDS:
            stream = torch.load(CACHE / f"{prefix}stream{seed}.pt", map_location="cpu")
            torch.set_rng_state(stream["rng"])
            post = StabilityRanking(config)
            predictions, class_prob, stable_conf = [], [], []
            for images in stream["images"].split(config.dataset.test.batch_size):
                images = images.cuda()
                pred, score = post.postprocess(net, images)
                logits = (net.logit_scale * images @ net.text_features)[
                    :, : net.n_cls
                ].float()
                probability, top_class = logits.softmax(1).max(1)
                if not torch.equal(top_class, pred):
                    raise ValueError(
                        "ID probability and TANL use different predictions"
                    )
                predictions.append(pred.cpu())
                class_prob.append(probability.cpu())
                stable_conf.append(score.cpu())
            pred = torch.cat(predictions).numpy()
            p_id = torch.cat(class_prob).numpy()
            with np.load(
                HERE / "scores" / f"{dataset}_uniform_positive_{seed}.npz"
            ) as ref:
                label = stream["labels"].numpy()
                if not all(
                    np.array_equal(a, ref[k])
                    for k, a in (
                        ("label", label),
                        ("pred", pred),
                        ("gate", torch.cat(post.gates).numpy()),
                    )
                ):
                    raise ValueError(
                        "The sample order, classifier or FIFO admissions changed"
                    )
                baseline = ref["conf"].astype(np.float32)
            scores = {
                "positive": baseline,
                "stable_labels": torch.cat(stable_conf).numpy(),
                "top_class_only": p_id,
                "joint_top_class": baseline * p_id,
            }
            for method, score in scores.items():
                if not np.isfinite(score).all():
                    raise ValueError(f"Non-finite scores in {method}")
                fpr, auc, _, _, acc = compute_all_metrics(score, label, pred)
                rows.append(
                    {
                        "dataset": dataset,
                        "template": "uniform",
                        "method": method,
                        "seed": seed,
                        "fpr95": 100 * fpr,
                        "auroc": 100 * auc,
                        "id_accuracy": 100 * acc,
                        "checked_gates": len(label),
                        "replaced_per_batch": float(np.mean(post.replacements))
                        if method == "stable_labels"
                        else 0.0,
                    }
                )
            print(f"Checked {dataset} seed {seed}", flush=True)
    write_rows(HERE / "candidate_scores_seeds.csv", rows)
    write_rows(HERE / "candidate_scores_summary.csv", summarize(rows))


if __name__ == "__main__":
    main()

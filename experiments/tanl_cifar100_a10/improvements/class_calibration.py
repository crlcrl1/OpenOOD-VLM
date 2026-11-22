"""Screen past-only calibration by predicted ID class on recorded positive-gate streams."""

import numpy as np
from ablation import DATASETS, summarize, write_rows
from capture import HERE, method_config
from run import SEEDS
from scipy.special import logit

from openood.evaluators.metrics import compute_all_metrics


def calibrate(conf, pred, gates, class_count, batch_size, history_size):
    odds = logit(np.clip(conf.astype(np.float64), 1e-7, 1 - 1e-7))
    result = odds.copy()
    history = np.empty(0, dtype=np.int64)
    # One class's expected share of a full, balanced history supplies the prior weight.
    prior_count = history_size / class_count
    for start in range(0, len(odds), batch_size):
        batch = np.arange(start, min(start + batch_size, len(odds)))
        if len(history):
            global_median = np.median(odds[history])
            for cls in np.unique(pred[batch]):
                members = history[pred[history] == cls]
                if len(members):
                    offset = np.median(odds[members]) - global_median
                    weight = len(members) / (len(members) + prior_count)
                    result[batch[pred[batch] == cls]] -= weight * offset
        history = np.concatenate((history, batch[gates[batch] == 1]))[-history_size:]
    return result


def main():
    config = method_config()
    rows = []
    for dataset in (*DATASETS, "texture"):
        for seed in SEEDS:
            path = HERE / "scores" / f"{dataset}_uniform_positive_{seed}.npz"
            with np.load(path) as run:
                score = calibrate(
                    run["conf"],
                    run["pred"],
                    run["gate"],
                    100,
                    config.dataset.test.batch_size,
                    config.postprocessor.postprocessor_args.admission_window,
                )
                fpr, auc, _, _, acc = compute_all_metrics(
                    score, run["label"], run["pred"]
                )
            rows.append(
                {
                    "dataset": dataset,
                    "template": "uniform",
                    "method": "class_calibration",
                    "seed": seed,
                    "fpr95": 100 * fpr,
                    "auroc": 100 * auc,
                    "id_accuracy": 100 * acc,
                }
            )
    write_rows(HERE / "class_calibration_seeds.csv", rows)
    write_rows(HERE / "class_calibration_summary.csv", summarize(rows))


if __name__ == "__main__":
    main()

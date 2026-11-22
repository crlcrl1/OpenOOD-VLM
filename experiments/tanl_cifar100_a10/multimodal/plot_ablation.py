"""Draw the change from the positive-queue baseline in percentage points."""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
ROWS = [
    ("nice", "original", "joint1", "Top-1 联合概率"),
    ("nice", "original", "joint3", "Top-3 联合概率"),
    ("nice", "original", "compete1", "Top-1 类竞争"),
    ("nice", "original", "compete3", "Top-3 类竞争"),
    ("nice_vanilla", "original", "all", "双文本模板"),
    ("nice", "mean", "all", "双图像视角"),
    ("nice_vanilla", "mean", "all", "双模板＋双视角"),
    ("nice_vanilla", "mean", "compete3", "双模板＋双视角＋前三类竞争"),
]
DATASETS = {"cifar10": "CIFAR-10", "tin": "Tiny IN", "texture": "DTD", "svhn": "SVHN*"}


def main() -> None:
    with (HERE / "summary.csv").open() as source:
        scores = {
            (r["dataset"], r["prompt"], r["view"], r["head"]): r
            for r in csv.DictReader(source)
        }
    colors = LinearSegmentedColormap.from_list(
        "difference", ["#a03b1b", "#fffaf4", "#087267"]
    )
    plt.rcParams.update(
        {"font.family": ["Noto Sans CJK SC", "DejaVu Sans"], "font.size": 10}
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), layout="constrained")
    for ax, (metric, direction, title) in zip(
        axes,
        (("fpr95_mean", -1, "FPR95 降幅"), ("auroc_mean", 1, "AUROC 增幅")),
        strict=True,
    ):
        delta = np.array(
            [
                [
                    direction
                    * (
                        float(scores[d, p, v, h][metric])
                        - float(scores[d, "nice", "original", "all"][metric])
                    )
                    for d in DATASETS
                ]
                for p, v, h, _ in ROWS
            ]
        )
        limit = max(np.abs(delta).max(), 0.1)
        ax.imshow(delta, cmap=colors, aspect="auto", vmin=-limit, vmax=limit)
        for (i, j), value in np.ndenumerate(delta):
            ax.text(
                j,
                i,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > limit * 0.65 else "#202729",
            )
        ax.set_xticks(range(len(DATASETS)), DATASETS.values())
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([label for *_, label in ROWS] if ax is axes[0] else [])
        ax.tick_params(length=0, pad=5)
        ax.axvline(2.5, color="#656b6a", linewidth=1, linestyle=":")
        ax.set_title(title + "（百分点）", pad=12)
        ax.spines[:].set_visible(False)
    fig.supxlabel("相对正队列基线，正数表示改善。* SVHN 为预留数据集。", fontsize=10)
    fig.savefig(HERE / "ablations.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

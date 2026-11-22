"""Plot the paired uniform-template TANL and positive-gate ROC curves."""

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import roc_curve

matplotlib.use("Agg")
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    plt.rcParams.update(
        {"font.family": ["Noto Sans CJK SC", "DejaVu Sans"], "font.size": 10}
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(9.0, 4.1), sharex=True, sharey=True, layout="constrained"
    )
    grid = np.linspace(0, 1, 1001)
    for ax, (dataset, title) in zip(
        axes, (("cifar10", "CIFAR-10"), ("tin", "Tiny ImageNet")), strict=True
    ):
        ax.plot([0, 1], [0, 1], color="#bdc2c8", linewidth=0.8, zorder=0)
        ax.axhline(0.95, color="#636b75", linewidth=0.9, linestyle=":")
        for method, label, color, style in (
            ("baseline", "统一模板 TANL", "#52677c", "--"),
            ("positive", "只改正队列", "#bd4c24", "-"),
        ):
            curves, points = [], []
            for seed in (0, 1, 2):
                with np.load(
                    HERE / "scores" / f"{dataset}_uniform_{method}_{seed}.npz"
                ) as saved:
                    fpr, tpr, _ = roc_curve(saved["label"] == -1, -saved["conf"])
                points.append(fpr[np.flatnonzero(tpr >= 0.95)[0]])
                curves.append(np.interp(grid, fpr, tpr))
            ax.plot(
                grid,
                np.mean(curves, axis=0),
                style,
                color=color,
                linewidth=2,
                label=label,
            )
            ax.plot(
                np.mean(points),
                0.95,
                marker="o",
                markerfacecolor="white",
                markeredgecolor=color,
                markersize=6,
            )
        ax.set(title=f"OOD · {title}", xlabel="ID 误报率", xlim=(0, 1), ylim=(0, 1.02))
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 0.95])
        ax.grid(alpha=0.13)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("OOD 检出率")
    axes[1].legend(loc="lower right", frameon=False)
    fig.savefig(HERE / "positive_roc.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

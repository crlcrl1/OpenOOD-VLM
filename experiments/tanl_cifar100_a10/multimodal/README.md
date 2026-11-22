# TANL 的类别证据、文本模板与图像视角消融

## 结果

**作业主实验可以用单模板双视角。** 把原图和水平翻转图的 CLIP 表示平均，保留 TANL 的全类评分。相对上一轮正队列基线，四组的平均 FPR95 均下降，按完整精度计算降幅为 0.05–0.96 个百分点。CIFAR-100 分类准确率从 63.21% 升至 63.98%。每张图需要两次视觉编码。[双视角配置](dual_view.yml)

下表每格为三个测试顺序的平均 FPR95 ↓ / AUROC ↑，单位为百分数。共运行 6 种图文表示和 5 种评分，得到 30 个配置、4 个 OOD 数据集、360 条逐种子结果。[完整均值与顺序间标准差](summary.csv) · [逐种子结果](seeds.csv)

| 配置 | CIFAR-10 | Tiny ImageNet | DTD 纹理 | SVHN 街景数字（预留） | CIFAR-100 分类准确率 |
|---|---:|---:|---:|---:|---:|
| 上一轮正队列基线 | 63.88 / 82.77 | 66.83 / 80.98 | 17.93 / 96.98 | 1.53 / 99.64 | 63.21 |
| 上一轮 Top-1 联合概率 | 54.06 / 83.99 | 65.90 / 79.61 | 33.20 / 94.47 | 1.20 / 99.67 | 63.21 |
| Top-3 联合概率 | 55.54 / 84.17 | 67.09 / 80.07 | 30.20 / 95.44 | 0.82 / 99.75 | 63.21 |
| Top-1 类竞争 | 55.19 / 84.21 | 66.02 / 79.77 | 29.14 / 95.40 | 1.09 / 99.70 | 63.21 |
| Top-3 类竞争 | 57.04 / 84.20 | 66.62 / 80.23 | 26.72 / 96.01 | 0.83 / 99.76 | 63.21 |
| 仅翻转图 | 63.50 / 82.85 | 66.78 / 81.00 | 16.94 / 97.01 | 1.79 / 99.54 | 63.25 |
| **仅双视角** | **63.19 / 82.93** | **66.51 / 81.06** | **16.98 / 97.05** | **1.48 / 99.63** | **63.98** |
| 仅词名模板 | 68.44 / 80.07 | 65.50 / 81.38 | 21.14 / 96.19 | 3.21 / 99.24 | 55.94 |
| 仅双模板 | 64.79 / 82.50 | 64.82 / 82.10 | 18.29 / 96.80 | 1.92 / 99.50 | 61.54 |
| 双模板＋双视角，按前三组选出 | 63.76 / 82.83 | 64.02 / 82.30 | 17.34 / 96.88 | 2.10 / 99.47 | 61.93 |
| 双模板＋双视角＋Top-3 竞争，预设组合 | 56.79 / 84.33 | 64.56 / 81.37 | 26.88 / 96.00 | 1.02 / 99.74 | 61.93 |
| 双模板＋双视角＋Top-1 联合概率 | 53.14 / 84.24 | 64.73 / 80.47 | 34.44 / 94.36 | 1.74 / 99.56 | 61.93 |
| 双模板＋双视角＋Top-3 联合概率 | 54.54 / 84.54 | 64.82 / 81.13 | 30.70 / 95.40 | 1.22 / 99.67 | 61.93 |

**跨数据集选择。** 按三个已用数据集的等权平均 FPR95，双模板＋双视角＋全类评分排第一，基线 49.55%，该组合 48.38%。Tiny ImageNet 的三个顺序分别降低 4.09、2.41、1.93 个百分点。CIFAR-10 的平均降幅只有 0.12 个百分点，其中两个顺序的 FPR95 略升。预留 SVHN 上，三个顺序均出现回退，平均 FPR95 从 1.53% 升到 2.10%，AUROC 从 99.64% 降到 99.47%。[按预定规则生成的配置](selected.yml)

**论文对应实验。** 在 CIFAR-10 的 30 个配置中，双模板＋双视角＋Top-1 联合概率的 FPR95 最低，为 53.14%，AUROC 为 84.24%。上一轮该行的最低 FPR95 为 54.06%，论文表 A10(b) 报告 56.32% / 80.47%。同一新配置在 DTD 的 FPR95 升到 34.44%。若关注 CIFAR-10 的 AUROC，改用 Top-3 联合概率可达到 84.54%，对应 FPR95 为 54.54%。这两项是对本轮配置的事后比较。[CIFAR-10 最低 FPR95 配置](a10_joint1.yml) · [论文与本地原版复现对照](../README.md)

![四种改动相对正队列基线的 FPR95 与 AUROC 变化](ablations.png)

## 实验设计

基线沿用统一提示 `The nice {}.` 和正队列的历史 80% 分位入队规则。冻结 CLIP ViT-B/16，每批 200 张图、队列长 300、负标签取 1,000 个，所有方法使用相同的 69,929 个候选负词和噪声初始化。每组独立混洗 3 次，种子为 0、1、2。每个测试流含全部 10,000 张 CIFAR-100 测试图，分别混入 CIFAR-10 的 10,000 张、Tiny ImageNet 的 6,526 张、DTD 的 5,640 张或 SVHN 的 26,032 张。FPR95 按仓库约定取 OOD 为正类，即检出 95% OOD 时误拒的 ID 比例。[上一轮实验](../improvements/README.md) · [SVHN 数据与清单校验值](../improvements/svhn_sources.tsv)

实验先固定以下方法和对照。对 CIFAR-10、Tiny ImageNet、DTD 三个已用数据集的 FPR95 取等权平均，以最低者作为候选，再单独报告其在此前未用于选方法的 SVHN 上的表现。另单列预设的“双模板、双视角、前三类竞争”组合，以及逐项消融。三个种子估计同一批图像对在线顺序的敏感度。

| 改动 | 具体做法 | 对照 |
|---|---|---|
| 前三类联合概率 | TANL 分数乘以 CIFAR-100 内部 softmax 的前三类概率之和 | 原始全类评分、上一轮前一类联合概率 |
| 前三类与负词竞争 | 每个激活负词前缀只与得分最高的三个 ID 类计算 softmax | 只有一个 ID 类参与竞争、全部 100 类参与 |
| 文本模板集成 | 对每个 ID 类和每个负词分别编码 `The nice {}.` 和 `{}.`，归一化、等权平均后再归一化 | 两个单模板分别运行 |
| 图像双视角 | 编码同一输入及其水平翻转，各自归一化后将特征等权相加并归一化 | 仅原图、仅翻转图 |

若所有 ID 类的指数质量为 $A$，前三类占 $T$，第 $m$ 个负词前缀占 $N_m$，原分数对 $A/(A+N_m)$ 取平均。联合评分对 $T/(A+N_m)$ 取平均，竞争评分则对 $T/(T+N_m)$ 取平均。两种评分均只替换最后一步，同一表示下共用每批的入队决定和负词排序。切换图像或文本表示时重新运行完整队列。图像双视角需要每张图做两次视觉前向，文本集成可离线编码。[评分实现](../../../openood/postprocessors/activated_neg_postprocessor.py) · [CLIP 模板集成](https://proceedings.mlr.press/v139/radford21a.html) · [测试时图像增强研究](https://arxiv.org/abs/2011.11156)

## 核验

三个既有数据集共 156,498 次基线预测与上一轮的标签顺序、预测类别、分数及入队标记逐项相等。对 360 条结果分别用秩和重算 AUROC，均与仓库一致。FPR95 沿用仓库默认的 ROC 折点。若按全部唯一阈值直接计数，有 4 条逐种子记录相差 0.01–0.05 个百分点，来源是共线 ROC 点的省略。[仓库指标代码](../../../openood/evaluators/metrics.py) · [逐种子结果](seeds.csv)

双模板＋双视角＋前三类竞争的正常入口从权重重新生成了全部 70,029 个文本向量，随后从 CIFAR-10、SVHN 原图运行。种子 0 的 20,000 和 36,032 个标签、预测和分数分别与特征回放逐项相等。按前三组选出的全类方案复用上述文本向量，也在这两个原图流上逐项相等。[预设组合核验](combined_raw_checks.csv) · [预选方案核验](selected_raw_checks.csv)

单模板双视角另用 Tiny ImageNet 与 CIFAR-100 的原图核验，种子 0 的 16,526 个预测、分数和标签与回放逐项相等。[双视角原图核验](dual_view_raw_checks.csv)

## 复现

从仓库根目录使用系统 Python 和 CUDA。沿用本机[上一轮](../improvements/README.md)缓存时，可从本次的 `prepare_texts.py` 开始。全新运行先执行代码块中的六条基线准备命令。本次的图像特征和逐图评分缓存在 `data/tanl_a10_diagnostic_features/multimodal/`。

```bash
base=experiments/tanl_cifar100_a10
study="$base/multimodal"
export PYTHONHASHSEED=0 PYTHONPATH=.:"$base":"$base/improvements"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

# 首次复现时准备基线
/usr/bin/python3 "$base/prepare_inputs.py"
/usr/bin/python3 "$base/run.py"
/usr/bin/python3 "$base/capture_features.py"
/usr/bin/python3 "$base/improvements/capture.py"
/usr/bin/python3 "$base/improvements/ablation.py" --sides
/usr/bin/python3 "$base/improvements/texture_holdout.py"

# 本次消融
/usr/bin/python3 "$study/prepare_texts.py"
/usr/bin/python3 "$study/prepare_views.py"
/usr/bin/python3 "$study/evaluate.py"
/usr/bin/python3 "$study/evaluate.py" --select-known
/usr/bin/python3 "$study/evaluate.py" --summarize
/usr/bin/python3 "$study/verify.py" --raw --compare
/usr/bin/python3 "$study/verify.py" --case selected --raw --compare
/usr/bin/python3 "$study/verify.py" --case dual_view --raw --compare
/usr/bin/python3 "$study/plot_ablation.py"
```

单独重跑某组可使用 `evaluate.py --datasets cifar10 --prompts nice --seeds 0`。在仓库评测入口，按顺序把 [quantile.yml](../improvements/quantile.yml)、[positive.yml](../improvements/positive.yml) 和 [combined.yml](combined.yml) 叠加到[复现协议](../protocol.yml)，即可使用上述预设组合。

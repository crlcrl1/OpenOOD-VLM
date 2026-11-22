# TANL 的正队列入队门槛

**在统一提示模板的 TANL 上，只替换正队列门槛，使两组近分布测试的平均 FPR95 从 67.91% 降到 65.36%，平均 AUROC 从 79.49% 升到 81.87%。** 收益主要来自 Tiny ImageNet。CIFAR-10 的三个测试顺序均有 AUROC 增益，FPR95 的变化分别为 −0.05、−0.39 和 +0.69 个百分点。[逐种子 CIFAR-10 结果](cifar10_sides_seeds.csv) · [逐种子 Tiny ImageNet 结果](tin_sides_seeds.csv)

## 对照与结果

CIFAR-100 作 ID，每组均包含全部 10,000 张 ID 测试图像。CIFAR-10 含 10,000 张 OOD 图像，Tiny ImageNet 含 6,526 张。使用同一 ViT-B/16、预处理、词序、初始化与混合流，batch 为 200。随机种子 0、1、2 只改变测试顺序和相应的初始化排列。每格为三个种子的 **FPR95 ↓ / AUROC ↑**，单位为百分数。FPR95 采用仓库的 OOD 正类口径，即检出至少 95% 的 OOD 时，误报为 OOD 的 ID 比例。[复现协议](../protocol.yml) · [OpenOOD 数据定义](https://github.com/Jingkang50/OpenOOD/blob/main/openood/evaluation_api/datasets.py) · [TIN 清单与校验值](tin_sources.tsv)

| 方法 | CIFAR-10 | Tiny ImageNet | 两组等权平均 |
|---|---:|---:|---:|
| 论文表 A10(b) | 56.32 / 80.47 | 59.16 / 89.65 | 57.74 / 85.06 |
| 本地原始模板 TANL | 72.32 / 72.94 | 77.62 / 73.00 | 74.97 / 72.97 |
| 本地统一模板 TANL | 63.80 / 82.24 | 72.02 / 76.74 | 67.91 / 79.49 |
| **统一模板 + 正队列分位数** | **63.88 / 82.77** | **66.83 / 80.98** | **65.36 / 81.87** |

新方法在 CIFAR-10 的 FPR95、AUROC 样本标准差分别为 0.16、0.07，在 Tiny ImageNet 分别为 0.63、0.09。这些数描述打乱顺序带来的波动。两组的 ID 分类准确率均为 63.21%。论文 CIFAR 启动配置缺失，因此方法增益以同模板、同数据的本地对照计算。[论文第 16 页](https://arxiv.org/pdf/2603.25250v1#page=16) · [复现差距排查](../diagnostics/README.md)

![统一模板下，CIFAR-10 与 Tiny ImageNet 上的基线、正队列改动及 95% OOD 检出率](positive_roc.png)

曲线在同一误报率网格上对三个顺序取均值，空心点标出三个实测 FPR95 的均值。CIFAR-10 的两条曲线有交叉，95% 检出率处几乎重合。[绘图脚本](plot_roc.py)

## 入队规则与消融

取 TANL 对当前图像给出的初步 ID 分数。高于最近 1,000 张真实图像分数的第 80 百分位时进入正队列。第一批用本批估计门槛，随后只用此前批次。负队列继续使用原来的 Otsu 门槛与间隔，并排除本批已经进入正队列的图像。两个 FIFO 容量仍为 300，历史权重 0.95，每批选 1,000 个负标签。测试真值只参与指标与队列纯度统计。[方法实现](../../../openood/postprocessors/adaptive_tanl_postprocessor.py) · [分位数配置](quantile.yml) · [正队列配置](positive.yml)

在 CIFAR-10 种子 0 上，统一模板基线累计有 309 张图进入正队列，其中 303 张是真 ID。改动后，4,071 张中有 3,586 张是真 ID。正队列获得了近 12 倍的真实 ID 图像，纯度也从 98.1% 降到 88.1%。负队列若同时改收低于历史第 20 百分位的图像，CIFAR-10 的两项指标都会变差。[基线入队记录](cifar10_seeds.csv) · [单侧门槛记录](cifar10_sides_seeds.csv)

| 在统一模板上改动 | CIFAR-10 | Tiny ImageNet |
|---|---:|---:|
| 只改负队列 | 70.86 / 79.16 | 75.22 / 78.60 |
| 正负两侧都改 | 68.91 / 78.69 | 74.61 / 78.97 |
| 只按视觉共激活去重 | 64.72 / 81.80 | 71.80 / 76.82 |
| 双侧分位数 + 去重 | 69.21 / 78.95 | 74.01 / 79.40 |
| 双侧分位数 + 视觉队列评分 | 78.68 / 74.38 | 83.15 / 80.54 |

去重在激活差排序前 2,000 个词中计算本批图像响应的相关系数，把与任一更高排名词相关性达到 0.90 的词后移。视觉评分则把历史正、负图像队列的对数核密度比加到 TANL 对数几率上，排除当前批和初始化用的文本、噪声。原始模板下，视觉评分在 TIN 得到 65.40 / 87.70，在 CIFAR-10 得到 87.17 / 75.78。视觉缓存已有 AdaNeg 等相关工作。[原始与统一模板消融](tin_summary.csv) · [CIFAR-10 视觉评分](cifar10_visual_summary.csv) · [TIN 视觉评分](tin_visual_summary.csv) · [纯视觉对照](visual_only_summary.csv) · [AdaNeg](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4462db5eee6823b2abad0d1f955e187a-Abstract-Conference.html)

## 独立验证

分位点、窗口和去重阈值在第一轮评测前固定。先比较两侧同时改动与去重，随后补做单侧消融并选定正队列方案。两个近分布数据集均参与了方案选择。冻结模板、门槛和评分规则后，使用 5,640 张 DTD 纹理作为新的 OOD 测试集，沿用相同的 CIFAR-100 ID 图像。[DTD 来源与校验值](texture_sources.tsv) · [独立验证脚本](texture_holdout.py)

| DTD 远分布测试 | FPR95 ↓ | AUROC ↑ |
|---|---:|---:|
| 论文表 A10(b) | 31.36 | 95.36 |
| 统一模板 TANL | 21.48 ± 0.19 | 96.65 ± 0.08 |
| **统一模板 + 正队列分位数** | **17.93 ± 0.09** | **96.98 ± 0.04** |

DTD 的三个顺序中，两项指标均改善。相对同模板对照，平均 FPR95 降 3.55 个百分点、AUROC 升 0.33 个百分点。统一模板的本地基线本身已经高于论文在 DTD 的表现。[逐种子 DTD 结果](texture_seeds.csv)

两个近分布数据集种子 0 的 36,526 个正队列方案分数、DTD 三个种子的 46,920 个分数，均与从原图经过 CLIP 和仓库评测器得到的结果逐个相等。[近分布原图核验](positive_raw_check.json) · [DTD 核验](texture_seeds.csv)

## 正队列方案上的其他改法

保持统一模板、正队列的入队规则、测试图像和种子 0、1、2，单独改变最终评分或选词排序。每格为 FPR95 ↓ / AUROC ↑。DTD 在本轮属于已知数据集。

| 方法 | CIFAR-10 | Tiny ImageNet | DTD |
|---|---:|---:|---:|
| 上一轮正队列方案 | 63.88 / 82.77 | 66.83 / 80.98 | 17.93 / 96.98 |
| 最可能 ID 类的联合分数 | **54.06 / 83.99** | 65.90 / 79.61 | 33.20 / 94.47 |
| 仅用最可能 ID 类的条件概率 | 74.78 / 73.82 | 77.44 / 68.54 | 83.99 / 71.32 |
| 按预测类别校准 | 65.28 / 82.64 | **65.68 / 81.28** | 18.21 / 96.93 |
| 负标签排序扣除一份历史标准误 | 65.13 / 82.38 | 69.86 / 79.26 | 未测 |

**最可能类的联合分数。** 设 `S` 是上一轮 TANL 对 ID 的最终分数，`p` 是冻结 CLIP 对 100 个 ID 类的 softmax 最大值，保持模型原有 logit 尺度。取 `S × p`，等价于在式（15）每个前缀中计算概率最高的 ID 类所占的质量，再对前缀平均（忽略浮点舍入）。当前批已经有这些 logits，无需再编码图像。CIFAR-10 的 FPR95 在三个顺序分别是 54.03、54.07、54.09；DTD 的 FPR95 在三个顺序均变差。独立计算的逐图乘积与原图入口的输出在 CIFAR-10 种子 0、DTD 全部三个种子逐项相等，共 66,920 个分数。[可叠加配置](joint_class.yml) · [近分布逐种子结果](candidate_scores_seeds.csv) · [原图核验与 DTD 对照](candidate_images.csv)

**预测类别校准。** 在最近 1,000 个已接纳的伪 ID 样本中，用各预测类相对于全体的 log-odds 中位数偏差修正当前分数。第 `c` 类有 `n` 个样本时偏差乘 `n/(n+10)`，无历史样本的类使用全局尺度。基于保存的逐图分数和入队记录，严格按批顺序回放。该改动在 Tiny ImageNet 的 FPR95、AUROC 分别改善 1.16、0.30 个百分点，在另外两组未带来改善。[校准实现](class_calibration.py) · [各组结果](class_calibration_summary.csv)

**负标签稳定性。** 最终选词时，从原激活差扣除 `sqrt(var_pos/n_pos + var_neg/n_neg)`，其中方差和样本数来自当时的两个 FIFO。两个近分布数据集的三个种子均与上一轮的入队掩码、预测类别和图像顺序逐项一致。该改动在两组的误报率都上升。[对照实现](candidate_scores.py) · [均值与顺序间标准差](candidate_scores_summary.csv)

## 式（15）的独立加速

原实现为每个负标签前缀重复计算 softmax。用一次 `logsumexp` 求全部 ID 类质量，再用 `logcumsumexp` 求各负标签前缀的质量，就能计算相同的数学表达式。该开关独立于入队规则。以下统一模板 TANL 测试使用 RTX 5060 Ti、CIFAR-10 OOD、20,000 张混合图像和 CUDA 同步计时。[实现](../../../openood/postprocessors/activated_neg_postprocessor.py) · [完整计时与误差](prefix_benchmark.json)

| 测量范围 | 原实现 | 前缀累加 |
|---|---:|---:|
| 单个 200 × 1,100 的 FP16 评分，15 次中位数 | 10.40 ms | 0.091 ms |
| 已有图像特征，完整后处理，3 个顺序均值 | 3.25 s | 1.31 s |
| 原图加载、编码、后处理和指标，种子 0 各跑两次 | 35.98、32.51 s | 30.90、30.92 s |

原图测试的平均耗时从 34.25 s 降到 30.91 s，约 1.11 倍加速，排除模型和词库构建时间。三个种子的 FPR95 均逐项相等，AUROC 的最大变化为 0.000042 个百分点。FP16 下逐样本分数的最大绝对差为 0.000488。设 `postprocessor.postprocessor_args.prefix_score: true` 可启用。[计时脚本](prefix_benchmark.py)

## 重跑

从仓库根目录使用系统 Python。已有本机缓存时，可直接在下面的环境变量下运行 `/usr/bin/python3 experiments/tanl_cifar100_a10/improvements/ablation.py --sides` 重跑正、负队列消融。全量复现顺序如下。每张图的预测、分数与入队标记在 [scores](scores/) 中。

```bash
export PYTHONHASHSEED=0 PYTHONPATH=.:experiments/tanl_cifar100_a10
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
base=experiments/tanl_cifar100_a10
/usr/bin/python3 "$base/prepare_inputs.py"
/usr/bin/python3 "$base/run.py"
/usr/bin/python3 "$base/capture_features.py"

exp="$base/improvements"
/usr/bin/python3 "$exp/capture.py"
/usr/bin/python3 "$exp/ablation.py"
/usr/bin/python3 "$exp/ablation.py" --sides
/usr/bin/python3 "$exp/capture.py" --visual
/usr/bin/python3 "$exp/ablation.py" --visual
/usr/bin/python3 "$exp/visual_control.py"
/usr/bin/python3 "$exp/texture_holdout.py"
/usr/bin/python3 "$exp/candidate_scores.py"
/usr/bin/python3 "$exp/candidate_images.py"
/usr/bin/python3 "$exp/class_calibration.py"
/usr/bin/python3 "$exp/prefix_benchmark.py"
/usr/bin/python3 "$exp/plot_roc.py"
```

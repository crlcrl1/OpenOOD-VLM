# CIFAR-100 对 CIFAR-10 的 TANL 复现差距

**单独统一形容词模板就使 AUROC 超过了论文值，正样本入队过少也限制了在线选词。** 以下实验固定 ViT-B/16、69,929 个候选词、每批 200 张、相同的三条测试流。百分数是种子 0、1、2 的均值，每个种子包含全部 20,000 张图。[22 组对照的均值与标准差](summary.csv) · [原复现与参数](../README.md) · [论文表 A10(b)](https://arxiv.org/pdf/2603.25250v1#page=16)

## 文本模板

论文第 4.1 节写作 `The nice <label>`。公开代码用 `The nice {}.` 编码 100 个 ID 类名和 58,476 个候选名词，另用 `This is a {} photo.` 编码 11,453 个候选形容词。NegLabel 原版也按词性使用两个模板，其配置选择第 85 号名词模板，两个模板都没有句点。本实验以词性和句点为两个因素，重新编码相应文本，保持图像、候选词集合和顺序、温度、队列规则、测试流不变。[TANL 第 4.1 节](https://arxiv.org/html/2603.25250v1#S4.SS1) · [TANL 实现](../../../openood/networks/clip_fixed_ood_prompt.py) · [NegLabel 源码](https://github.com/XueJiang16/NegLabel/blob/3253db684075b2db47844676eccdfda40d67a573/mmcls/models/classifiers/multi_modal.py#L49-L93)

| 名词和 ID 模板 | 形容词模板 | FPR95 ↓ | AUROC ↑ | ID 分类准确率 |
|---|---|---:|---:|---:|
| `The nice {}.` | `This is a {} photo.`（仓库） | 72.32 | 72.94 | 63.21 |
| `The nice {}.` | `The nice {}.` | 63.80 | 82.24 | 63.21 |
| `The nice {}` | `This is a {} photo` | 74.18 | 71.90 | 59.28 |
| `The nice {}` | `The nice {}` | 64.98 | 82.51 | 59.28 |
| 论文表 A10(b) |  | **56.32** | **80.47** | — |

只替换形容词模板、保留句点，FPR95 改善 8.52 个百分点，AUROC 改善 9.29 个百分点。FPR95 仍比论文高 7.48 个百分点。另从原始 PNG 和公开图文编码入口重跑这一项，三个种子各 20,000 个置信分数均与特征回放逐项完全一致。四个模板条件全部列入比较，均未在两项指标上同时重现论文值。[原始图像对照](uniform_image_run.json)

**FPR95 口径会改变差距的大小。** 上表遵循本仓库，以 OOD 为正类并取召回率首次达到 95% 的点。只把正类换为 ID，四行的 FPR95 依次变为 82.29%、59.51%、84.10%、56.14%，AUROC 不变。最后一行与论文 56.32% 很近，但其 AUROC 为 82.51%，仍高 2.04 个百分点。论文引用的 NegLabel 代码确实把 ID 标作正类，切点取最接近 95% 的位置。本段为了只考察正类方向，保留了本仓库的切点规则。公开材料没有给出表 A10(b) 的逐图分数，无法确认作者当时计算这一行的口径。[本仓库指标函数](../../../openood/evaluators/metrics.py) · [NegLabel 指标函数](https://github.com/XueJiang16/NegLabel/blob/3253db684075b2db47844676eccdfda40d67a573/mmcls/utils/ood_evaluate.py#L43-L70)

## 在线队列

上游代码把 ID 类名与自身匹配得到的分数和噪声图像的分数都放进阈值历史。种子 0 的 100 个类名初始分数中位数为 1.000，而整条真实流中的入队分数中位数，ID 为 0.0216、OOD 为 0.0093。在第 50 批，阈值为 0.49，同一段分数历史去掉初始化样本后重算为 0.12。按式（9）的间隔 0.2，当时要超过 0.592 才能进正队列，低于 0.392 就能进负队列。这两个门槛决定谁能更新队列，最终的 FPR95 由返回分数的 ROC 曲线计算。整轮仅有 19 张图进正队列，100 个 ID 类里至少 81 类没有真实图像进入正队列。另有 9,936 张真实 ID 进了负队列。用统一的带句点模板后，种子 0 的正队列收到 309 张图，其中 303 张为真 ID。[阈值与队列实现](../../../openood/postprocessors/activated_neg_postprocessor.py) · [逐步观测](seed0.json)

为区分两个队列的作用，每批先按当前模型正常评分，评分结束后才用该批的真实 ID/OOD 身份填充指定队列。另一队列继续按原规则更新。所有条件使用相同图像特征和顺序。这三个使用历史真值的结果只衡量队列供给的影响，论文的零样本测试无法得到这些标签。

| 评分后的队列干预 | FPR95 ↓ | AUROC ↑ |
|---|---:|---:|
| 两个队列都按公开代码更新 | 72.32 | 72.94 |
| 只用真实 ID 更新正队列 | 55.59 | 86.07 |
| 只用真实 OOD 更新负队列 | 68.17 | 76.49 |
| 两个队列都按真实身份更新 | 53.82 | 87.01 |

单独补充真实正样本的作用更大。这个对照同时改变入队样本的数量、纯度和覆盖范围，无法继续把增益归因于其中某一项。只用真实图像的历史分数估计阈值，FPR95 / AUROC 为 73.27 / 73.51。单纯去掉初始文本分数也不足以恢复两项指标。

## 其他核查

- 对两份独立发布的 CIFAR 测试集逐张匹配 RGB 像素与数值类别，CIFAR-10 和 CIFAR-100 均为 10,000/10,000。用单独安装的 OpenAI CLIP 与仓库比较，种子 0 第一批的 200 个预处理张量和图像特征完全相同，100 个 ID 类名及 128 个候选词的 token 与文本特征也完全相同。实际温度为 0.01。三个种子从冻结特征重放时，原方法的逐图得分均与原始图像运行完全一致。[输入核验](input_audit.json) · [原始 CIFAR 数据说明](https://www.cs.toronto.edu/~kriz/cifar.html)
- 公开实现按两组的样本比例加权组内方差，附录写为两组方差直接相加。单独改为后者，FPR95 / AUROC 为 71.83 / 72.90。公开实现用 100 个类文本和 15 张混合噪声图像初始化。按式（10）的一种实现，正队列把每个类名重复三遍并打乱，负队列改用 300 张高斯噪声图像并沿用代码的像素缩放，结果为 72.13 / 72.89。仅在批末改用最终分数更新队列及分数历史，结果为 72.30 / 72.94。[论文附录 A6.5](https://arxiv.org/pdf/2603.25250v1#page=13) · [对照实现](../diagnose.py)
- 词表有 51 个含方括号的候选词，其中 `[acuminate` 和 `[immiscible` 在本地多批次排名靠前。只去掉这些词的方括号，结果为 71.17 / 75.48。仅把初始化 softmax 改成 FP32，结果仍为 72.32 / 72.94。[种子 0](seed0.json) · [种子 1](seed1.json) · [种子 2](seed2.json)
- 将无句点的统一模板、上述式（10）初始化、附录的无权重方差、批末最终分数更新同时启用，FPR95 / AUROC 为 65.21 / 81.99。改以 ID 为正类时为 57.01 / 81.99。这是根据公开文字构造的一组配置。作者的 CIFAR 脚本引用缺失的 `aaneg.yml`，仓库也未提供与表 A10(b) 绑定的配置或预测文件，尚不能据此确定原始配置。[作者发布的 CIFAR 脚本](https://github.com/PolyU-VCLab/OpenOOD-VLM/blob/2a2b73cec8657afccb52c6e3e33a399577af5cb1/scripts/ood/TANL/cifar.sh#L226-L263)

## 重跑

先按原复现说明准备输入和逐样本基线，从仓库根目录用相同的系统 Python 运行。第一条命令在真实图像上核对三次原始输出并记录冻结特征。输入核验单独需要 PyArrow。大型特征和数据缓存位于 `data/tanl_a10_diagnostic_features/`。

```bash
export PYTHONHASHSEED=0 PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/usr/bin/python3 experiments/tanl_cifar100_a10/capture_features.py
/usr/bin/python3 experiments/tanl_cifar100_a10/verify_inputs.py
/usr/bin/python3 experiments/tanl_cifar100_a10/diagnose.py
/usr/bin/python3 experiments/tanl_cifar100_a10/check_uniform_images.py
```

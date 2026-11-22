# TANL 在 CIFAR-100 与 CIFAR-10 上的近分布 OOD 复现

**论文表 A10(b) 的 CIFAR-10 这一行未复现。** 用 CIFAR-100 作 ID、CIFAR-10 作近分布 OOD，每个随机种子评测同一组 20,000 张图像。下表均为百分数，标准差是三个测试顺序之间的样本标准差。[论文第 16 页](https://arxiv.org/pdf/2603.25250v1#page=16) · [完整精度结果](metrics.csv)

后续排查发现，单独统一形容词和名词的提示模板，三次平均 FPR95 从 72.32% 降到 63.80%，AUROC 从 72.94% 升到 82.24%。正、负队列的独立对照还定位到本地运行中正样本几乎无法入队的问题。[复现差距的对照实验](diagnostics/README.md)

在这套基线上，只调整正队列的分位数门槛，使 CIFAR-10 与 Tiny ImageNet 两组近分布实验的平均 FPR95 再降 2.55 个百分点，平均 AUROC 再升 2.38 个百分点。[改进实验、远分布验证与运行命令](improvements/README.md)

后续实验在相同正队列上比较了两种类别评分、文本模板集成和图像翻转集成，并用额外的 SVHN 检验跨数据集表现。[四种改法的消融与配置](multimodal/README.md)

| 结果 | FPR95 ↓ | AUROC ↑ |
|---|---:|---:|
| 论文 | 56.32 | 80.47 |
| 种子 0 | 72.68 | 72.91 |
| 种子 1 | 72.46 | 72.97 |
| 种子 2 | 71.83 | 72.95 |
| 复现均值 ± 标准差 | **72.32 ± 0.44** | **72.94 ± 0.03** |
| 复现均值减论文值（百分点） | **+16.00** | **−7.53** |

各轮实验的主要数值见[改进实验总结](TANL改进尝试与结果.md)。文中链接的逐图分数、指标明细和图表保存在本地，可按各节的重跑步骤生成。

## 实验设置

- 使用 OpenAI CLIP ViT-B/16 和仓库的测试预处理、混合流评测器、TANL 后处理及指标函数。输入为 224×224，按仓库配置每批 200 张。CIFAR-100 的 100 个类全部参与分类，ID 分类准确率在三个种子下均为 63.21%。[CIFAR 配置](../../configs/datasets/cifar100/cifar100_clip.yml) · [实验参数](protocol.yml)
- 使用 NegLabel 发布的 27 份 WordNet 词表，按仓库逻辑去重后有 69,929 个候选词。TANL 每批选择 1,000 个，队列容量为 300，历史权重为 0.95，置信间隔参数为 0.2，评分采用式（15）的逐前缀平均。自动阈值完全由实现估计。先建词库缓存，再用种子 0 初始化冻结模型和噪声图像。种子 0、1、2 各自打乱测试流、重建两个队列。[词库来源](https://github.com/XueJiang16/NegLabel/tree/3253db684075b2db47844676eccdfda40d67a573/txtfiles) · [运行入口](run.py)
- 图像来自 OpenOOD 发布的 PNG。CIFAR-100 配置要求完整测试集，标准清单的 9,000 张 test 和 1,000 张 val 提供原始类标签，公开 full 清单提供排列。已校验三者的路径集合、10,000 张图像与类标签的一一对应，以及 CIFAR-10 的 10,000 个 OOD 条目。此处合并的两部分均来自原始 CIFAR-100 测试集。重新准备输入后，两份最终清单、20,000 个图像文件、27 份词表逐字节一致。[固定版本的输入地址与 SHA-256](sources.tsv) · [公共清单副本](https://github.com/zxk1212/OODD/tree/edbb1a32e5fe81e443942156f0a9cafb0297d95b/data/benchmark_imglist)

## 差距核查

1. **指标口径。** 仓库将 OOD 作为正类，用负的 ID 置信度计算 ROC。从保存的逐图分数直接计数 FPR95，并用秩统计量独立计算 AUROC，三个种子均与仓库一致。改以 ID 为正类时，平均 FPR95 为 82.29%。[指标实现](../../openood/evaluators/metrics.py) · [逐图分数 0](scores_seed0.npz) · [1](scores_seed1.npz) · [2](scores_seed2.npz)
2. **本地与上游的实现。** 本地版本为 `9055703`。把后处理换成上游 `c6fef2f` 的原始文件，固定模型和种子 0，20,000 个预测和置信分数逐项相同，最大绝对分差为 0。首次从空词库缓存出发，三种子的均值也为 72.33% / 72.94%（FPR95 / AUROC）。[上游后处理](https://github.com/YBZh/OpenOOD-VLM/blob/c6fef2f5ed890df2f59f9c6c3f9ae14fb5a72ab9/openood/postprocessors/activated_neg_postprocessor.py) · [对照结果](upstream_check.json) · [首次运行](cold_cache/metrics.csv)
3. **伪标签队列。** 上游种子 0 的自动阈值中位数为 0.49。整条流累计只有 19 张图进入伪 ID 队列，全部是真 ID。19,933 张进入伪 OOD 队列，其中 9,997 张是真 OOD、9,936 张是真 ID，伪 OOD 纯度仅 50.15%。真实标签仅用于事后统计。TANL 依靠两队列的激活差选词，这次运行中的伪 OOD 队列几乎没有区分两种分布。论文没有给出该行实验的入队统计，因而无法据此确定作者结果与本地结果不同的原因。[观测代码](upstream_check.py) · [入队统计](upstream_check.json)
4. **配置与阈值。** CIFAR 启动脚本引用的 `aaneg.yml` 未随仓库发布。本次使用仓库中的 `tanl.yml`，按论文和 CIFAR 脚本覆盖参数。脚本枚举四个阈值，后处理会在每批数据上用自动估计值覆盖传入值。为检验这个差异，另用种子 0 在相同测试顺序上固定四个阈值。四组的 FPR95 均高于论文的 56.32%，AUROC 均低于论文的 80.47%。[CIFAR 启动脚本](../../scripts/ood/TANL/cifar.sh) · [阈值实现](../../openood/postprocessors/activated_neg_postprocessor.py) · [固定阈值结果](fixed_thresholds.json)

| 固定阈值（种子 0） | FPR95 ↓ | AUROC ↑ |
|---|---:|---:|
| 0.01 | 76.58 | 74.61 |
| 0.10 | 73.91 | 73.60 |
| 0.30 | 73.25 | 73.10 |
| 0.50 | 72.20 | 72.96 |

## 重跑

从仓库根目录运行。实测环境为 RTX 5060 Ti 8 GB、Python 3.14.10、PyTorch 2.15.0、CUDA 13.4、torchvision 0.29.2。`prepare_inputs.py` 校验下载文件，已有缓存可离线复用。`--check-upstream` 额外运行上游对照及脚本中列出的固定阈值诊断。[项目依赖](../../setup.py)

```bash
/usr/bin/python3 experiments/tanl_cifar100_a10/prepare_inputs.py
PYTHONHASHSEED=0 PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /usr/bin/python3 -u experiments/tanl_cifar100_a10/run.py --check-upstream
```

# KGE_central集中式TransE工程

## 工程定位

本目录恢复此前在FB15k-237上完成的集中式TransE强基线，并支持通过YAML选择FB15k-237或WN18RR。它不创建客户端、不做联邦划分、不读取MAT拓扑，也不依赖FedML参数加载器。

为避免重新设计造成数值漂移，本工程逐文件恢复`HFLSnF_KG_v3.tasks.kge`中已经得到完整测试MRR `0.386967`的算法内核，包括：

- TransE模型；
- 两数据集统一编号和数据完整性校验；
- CUDA向量化filtered负采样；
- 频率子采样权重；
- 头尾交替的双向自对抗目标；
- 精确filtered头尾排名评估；
- 以验证MRR选择并恢复最佳模型。

这些模块现在实际保存在`KGE_central`目录中，不再依赖V3包。`KGE_central`还自己负责YAML读取、CUDA强制检查、随机种子、结果目录、配置快照和检查点保存，因此运行集中式实验时不会进入任何联邦训练代码。

## 历史正式基线

历史结果保存在：

```text
HFLSnF_KG_v3/results/hflsnf_kg_v3_centralized_strong_transe_cuda_20260727_231830_874259
```

主要指标：

| 指标 | 数值 |
|---|---:|
| 最佳epoch | 380 |
| 完整验证MRR | 0.393167 |
| 完整测试MRR | 0.386967 |
| 测试Hits@1 | 0.294782 |
| 测试Hits@3 | 0.426561 |
| 测试Hits@10 | 0.565035 |
| 测试平均排名 | 172.847 |
| 测试三元组数 | 20,466 |
| 头尾查询总数 | 40,932 |

机器可读版本见`baseline_reference.json`。

## 正式CUDA训练

在工作区根目录`D:\1\1myworkcode`执行：

```powershell
python -m KGE_central.run
```

默认加载：

```text
KGE_central/configs/centralized_fb15k237_strong_transe_cuda.yaml
```

该配置强制要求CUDA，不允许静默降级到CPU。当前对齐配方为种子42、256维、L1距离、450个epoch、每3个epoch使用关系分层验证子集选模、批量1024、256个负样本、学习率`5e-5`、双向自对抗目标，以及训练结束后的完整验证和完整测试。

FB15k-237默认复用：

```text
HFLSnF_KG_v5/data/FB15k-237
```

该目录与此前使用的V3目录三份文件逐字节一致，不改变编号顺序或训练数据。历史正式结果保留。

## 数据集选择与路径

正式YAML不需要填写`data_dir`，仅修改`data_args.dataset`即可切换数据集：

| 配置名称 | 默认目录，相对于本工程 |
|---|---|
| `fb15k-237` | `../HFLSnF_KG_v5/data/FB15k-237` |
| `wn18rr` | `../HFLSnF_KG_v5/data/wn18rr` |

名称忽略大小写并去除两端空格，兼容`fb15k237`别名。两种数据均读取`train.txt`、`valid.txt`和`test.txt`，按照原始顺序统一编号。WN18RR的数字形式实体始终作为字符串读取，保留前导零；附带字典文件不参与编号。

显式填写`data_dir`时，该路径优先于默认目录。相对路径以`KGE_central`为基准，绝对路径直接使用。文件不存在时直接报错，不回退到其他目录；需要恢复自动选择时，删除或注释该字段。

正式配置的`run_name`使用`kge_central_{dataset}_strong_transe_cuda`，运行前用规范名称替换占位符。配置快照记录实际数据集、绝对数据路径和展开后的结果名称。已有不带占位符的自定义名称仍然兼容。

## WN18RR首轮集中式实验

在工作区根目录执行：

```powershell
python -m KGE_central.run --config KGE_central/configs/centralized_wn18rr_strong_transe_cuda.yaml
```

首轮固定种子42、256维L1距离TransE、450个epoch、Adam学习率`learning_rate=lr=5e-5`、批量大小1,024、每个正样本256个负样本、头尾交替双向自对抗目标、`gamma=9`和对抗温度1；不启用提前停止，不扫描学习率或运行其他种子。

每3个epoch按照验证MRR选模，共150次。验证上限为4,096，超过WN18RR实际数量，因此每次使用全部3,034条验证三元组。训练结束恢复最佳验证模型，再运行完整验证和完整测试；沿用现有负采样过滤及精确filtered头尾排名定义。

本地原始文件统计为训练86,835条、验证3,034条、测试3,134条，共40,943个实体和11种关系，各划分内部和划分之间均无重复三元组。384个实体未在训练集中出现，保持原始划分，不删除对应测试查询。完整测试应包含3,134条三元组及6,268次头尾查询。

这是一份沿用现有配方的参考基线，不代表已经对WN18RR调优。根据训练后的验证曲线再决定是否增加预算或调参，不能依据测试指标选择配方。

正式配置强制要求CUDA。本机CPU加载检查和小图训练只验证代码链路，不计入正式精度。不会自动修改训练配方或降级为CPU正式训练。

## 实验验证与后续顺序

使用已安装PyTorch的Python环境执行：

```powershell
python -m unittest discover -s KGE_central/tests -v
```

测试覆盖数据集名称和路径选择、名称占位符、文本数据完整性、前导零与评估独有实体、合成图谱与词汇小图训练闭环、最佳检查点恢复和CUDA限制。真实数据加载核对也纳入测试；未安装对应数据时会明确跳过相关检查。

先完成WN18RR集中式参考实验；随后再单独实施NSFNET搭配FB15k-237和WN18RR的三方法对照；最后扩展38节点搭配DB100K。后两阶段分别需要先确定语义映射和划分质量，以及模型规模与通信资源假设，本次没有修改这些模块。

## CPU冒烟

下面的命令使用内存中的微型知识图谱，只验证代码链路，不代表正式精度：

```powershell
python -m KGE_central.run `
  --config KGE_central/configs/smoke_synthetic_cpu.yaml
```

也可以打开`run_from_ide.py`直接运行。该文件默认使用CPU冒烟配置；在GPU服务器上可把`DEFAULT_CONFIG_NAME`改为正式配置文件名。

## 输出文件

每次运行都会创建唯一的时间戳目录，并保存：

- `config_snapshot.json`：实际运行配置；
- `dataset_summary.json`：实体、关系和三元组数量；
- `entity2id.json`与`relation2id.json`：全局编号映射；
- `metrics.csv`：逐epoch损失、耗时和验证指标；
- `summary.json`：最佳epoch、完整验证和完整测试指标；
- `model_best.pt`：已恢复最佳验证模型的检查点。

## 修改实验参数

所有模型、训练、评估和设备参数都在YAML中。建议复制正式配置后只改变一个变量，并修改`run_name`，这样不会把新实验误认为历史强基线。

正式复现实验不要修改以下关键字段：

- `random_seed: 42`
- `embedding_dim: 256`
- `distance_norm: 1`
- `epochs: 450`
- `eval_every: 3`
- `batch_size: 1024`
- `learning_rate: 0.00005`
- `negative_sample_count: 256`
- `local_objective: bidirectional_self_adversarial`
- `test_max_triples: 0`

其中`test_max_triples: 0`表示使用全部官方测试三元组。

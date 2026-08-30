# KGE_pykeen双口径集中式TransE工程

## 工程定位

本目录是`KGE_central`的隔离对照工程。原目录保持不变，本工程固定使用PyKEEN 1.10.1，从两个口径验证自研集中式TransE框架的正确性与可信度。

- `matched_recipe`：使用PyKEEN模型与NSSA参数语义，保留原强基线的严格全真事实负采样、整批头尾交替、频率权重、验证选模和canonical评估。
- `pykeen_native`：使用PyKEEN官方pipeline、sLCWA、BasicNegativeSampler、NSSALoss、EarlyStopper和RankBasedEvaluator。

原生模式不会伪装成严格等价。它只过滤训练事实，不使用频率子采样权重，也不按整批交替头尾方向。差异会写入每个结果目录的`comparison_contract.json`。

历史自研参考值保存在`baseline_reference.json`，完整测试MRR为`0.386967`。

## 固定依赖

普通依赖记录在`requirements.txt`，服务器CUDA依赖合同记录在`requirements-server-py38-cu117.txt`。PyTorch使用CUDA专用索引，应与普通依赖分开安装。

固定版本为Python 3.8.18、PyTorch 1.13.1+cu117、CUDA构建11.7、PyKEEN 1.10.1、NumPy 1.21.0、PyYAML 6.0和torch-max-mem 0.0.4。

## 正式运行

matched正式实验：

```powershell
python -m KGE_pykeen.run --config KGE_pykeen/configs/matched_fb15k237_seed42_cuda.yaml
```

native正式实验：

```powershell
python -m KGE_pykeen.run --config KGE_pykeen/configs/native_fb15k237_seed42_cuda.yaml
```

两份配置共同固定种子42、256维、L1距离、450个epoch、每3个epoch验证选模、批量1024、Adam学习率`5e-5`、256个负样本、gamma 9.0和对抗温度1.0，不创建逆关系。matched与native的选模时点均为epoch 3、6、9直至450，完整测试仅在训练结束并恢复最佳模型后执行。

## 结果产物

每次运行创建独立时间戳目录并保存：

- `config_snapshot.json`：实际运行配置；
- `dataset_summary.json`：数据规模；
- `entity2id.json`和`relation2id.json`：全局映射；
- `environment_snapshot.json`：解释器、PyTorch、CUDA、PyKEEN和NumPy版本；
- `comparison_contract.json`：本模式的等价项和公开差异；
- `metrics.csv`：逐epoch训练记录；
- `pykeen_metrics.json`：PyKEEN optimistic和realistic独立指标；
- `summary.json`：canonical最佳验证和完整测试摘要；
- `model_best.pt`：带规范化键名和PyKEEN原始状态的完整检查点；
- `pykeen_model_raw.pt`：仅包含PyKEEN原始状态。

canonical检查点继续提供`entity_embeddings.weight`和`relation_embeddings.weight`，可被复制后的方向评估代码读取。

## 生成三臂报告

准备一个原`KGE_central`结果目录、一个matched结果目录和一个native结果目录后执行：

```powershell
python -m KGE_pykeen.compare `
  --central <KGE_central结果目录> `
  --matched <matched结果目录> `
  --native <native结果目录> `
  --output-dir <报告输出目录>
```

该命令先校验共同超参数、数据摘要、完整映射和完整测试规模，并对matched与native的训练、验证、测试划分逐项校验SHA-256，再生成`comparison_report.json`与简体中文`comparison_report.md`。历史central结果没有保存三元组划分哈希，因此报告会明确标注：central只能追溯验证数据摘要和映射文件，不能仅凭旧结果目录声称逐条划分哈希已经验证。严格模式的张量级合同负责证明数学等价；正式MRR只如实报告差值，不设置“必须完全相等”的错误门槛。

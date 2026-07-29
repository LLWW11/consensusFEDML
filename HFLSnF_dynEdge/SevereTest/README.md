# SevereTest：MNIST 单标签极端非 IID 与分层联邦学习实验

本目录实现一套与根目录现有 MATLAB、历史 HFL 和 SnF 流程隔离的实验。数据仍保留 200 个真实客户端，每个客户端只有一个 MNIST 标签：

- 标签 0 分给客户端 `0、10、20、…、190`；
- 标签 1 分给客户端 `1、11、21、…、191`；
- 其余标签以此类推；
- 普通 FedAvg 配置每轮训练客户端 `0–9`；
- 新的分层 FedAvg 配置每轮训练客户端 `0–29`，并分成三个边缘组；
- 两种配置聚合后都向客户端 `0–199` 全量下发。

分层实验的固定拓扑如下：

```text
客户端 0–9   ──边缘组 0──┐
客户端 10–19 ──边缘组 1──┼──云端──客户端 0–199
客户端 20–29 ──边缘组 2──┘
```

每个边缘组都恰好包含标签 0–9 各一个单标签客户端。边缘组内按客户端训练样本数执行 FedAvg；云端再按三个边缘组的训练样本总数聚合边缘模型。

YAML 中的 `partition_alpha: 0.1` 只用于记录原问题场景。实际划分由 `partition_strategy: "label_modulo_20"` 控制，不再调用狄利克雷分布，因此不会出现空客户端。

## 运行环境

30 客户端分层配置强制使用 `gpu_id: 0` 对应的第一张可见 CUDA
显卡。如果 CUDA 不可用、服务器安装的是 CPU 版 PyTorch，或者 GPU
编号越界，程序会直接报错，不会静默回退到 CPU。

在服务器运行前先检查：

```bash
python -c "import torch; print('cuda=', torch.cuda.is_available()); print('count=', torch.cuda.device_count()); print('torch_cuda=', torch.version.cuda)"
```

期望 `cuda=True` 且 `count` 至少为 1。训练日志中还应显示
`device = cuda:0` 和 `SevereTest 使用 GPU`。如果通过
`CUDA_VISIBLE_DEVICES` 限制了显卡，YAML 的 `gpu_id: 0` 表示限制后可见的
第一张显卡。

以下命令均从项目根目录执行。

## 单元测试

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  -m unittest -v SevereTest.tests.test_severe_experiment
```

## 30 客户端分层联邦学习 GPU 两轮冒烟测试

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  SevereTest\run_experiment.py `
  --cf SevereTest\configs\fedml_config_severe_hfl_30clients_smoke_2round.yaml
```

## 30 客户端分层联邦学习 GPU 正式 200 轮实验

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  SevereTest\run_experiment.py `
  --cf SevereTest\configs\fedml_config_severe_hfl_30clients_200round.yaml
```

原 10 客户端普通 FedAvg 配置仍保留为
`fedml_config_severe_first10_200round.yaml` 和
`fedml_config_severe_smoke_2round.yaml`，用于与分层实验进行同种子对照。

训练入口会在结束后自动运行结果分析。若只需重新分析最近一次已完成实验：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  SevereTest\analyze_result.py
```

也可以显式指定结果目录：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  SevereTest\analyze_result.py `
  --result-dir "result\SevereTest\<实验目录>"
```

## 每轮执行顺序

1. 客户端 0–29 从上一轮云模型开始，各训练一个本地 epoch。
2. 三十个本地模型对固定 100 张测试图片执行探针推理。
3. 边缘组 0、1、2 分别聚合组内十个客户端。
4. 三个边缘模型对同一批固定探针推理。
5. 云端按三个边缘组的训练样本总数聚合边缘模型。
6. 云模型对固定探针推理，并显式下发到全部 200 个客户端。
7. 在全部客户端测试分区上计算总体和逐类准确率。

有效共识对每张探针图片独立计算：

\[
A=1-\mathrm{JSD}_{norm},\qquad
C=1-\overline{H}_{norm},\qquad
S=A\times C
\]

最后才对 100 张图片求均值。这样可以避免把共同接近均匀分布误判成高质量共识。

## 输出目录

每次运行创建独立目录：

```text
result/SevereTest/<实验名_时间戳>/
```

主要文件包括：

- `experiment_metadata.json`：配置、三组边缘拓扑、训练范围、下发范围和探针口径；
- `partition_manifest.csv`：200 个客户端的标签及训练、测试样本数；
- `training_schedule.jsonl`：每轮训练客户端与下发客户端；
- `test_metrics.csv`：逐轮完整测试集指标；
- `class_test_metrics.csv`：逐轮 10 类测试指标；
- `probe_probabilities.npz`：30 个客户端、3 个边缘模型和云端的完整概率张量；
- `probe_epoch_summary.csv`：训练期逐轮共识摘要；
- `analysis/round_metrics.csv`：从 NPZ 重新计算并核对后的逐轮指标；
- `analysis/analysis_summary.json`：正式分析摘要；
- `analysis/figures/`：客户端共识、分层共识、准确率和逐类热力图。

本实验只运行一个随机种子，正式分析只能作为描述性证据，不能解释为统计显著性或因果关系。

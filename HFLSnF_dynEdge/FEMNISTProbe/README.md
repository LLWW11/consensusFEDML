# FEMNIST MAT 探针实验

本目录实现完整 FEMNIST 的 250 个狄利克雷逻辑客户端、37 个固定训练槽位、620 张类别均衡探针和 200 轮 MAT 拓扑显式循环的四方案机制实验。源 H5 中的书写者 ID 只用于读取文件，不参与客户端划分。旧 MNIST 入口未被替换。

## 实验边界

- 数据来自 `dataset/FEMNIST/fed_emnist_train.h5` 和 `fed_emnist_test.h5`。
- 完整 671585 张训练图片和 77483 张测试图片使用同一个 `62×250` 类别比例矩阵进行 Dirichlet non-IID 划分，默认 `partition_alpha=0.2`、`partition_seed=0`。
- 训练集和测试集分别确定性打乱类内样本，全部样本只分配一次，并拒绝任何空客户端。
- 37 个固定槽位严格对应逻辑客户端 `[123–128, 41–46, 0–5, 164–169, 82–87, 205–210, 129]`。
- 四方案使用相同逻辑客户端划分、固定槽位顺序、初始模型、探针、MAT 文件和训练随机种子。
- MAT 文件固定为 `matlab/result-U-6fixedge_epoch200_varAlpha_0p1_trainable.mat`，固定 `u=0.5`。
- 第 `global_epoch` 轮使用 MAT 第 `global_epoch % 200` 行，循环编号为 `global_epoch // 200`。
- MAT 每轮只提供有效组数 `k` 和参与人数 `n`。37 个固定槽位被切成 `k` 个连续候选段，每组先取 `n//k` 人，再把余数依次补给前面的组；不使用 MAT 的具体客户端身份。
- 聚合权重只使用各逻辑客户端的真实训练样本数。四份 YAML 的基线批大小为 20；一键快速脚本默认覆盖为 128。两种模式的尾批都不会丢弃。
- 本实验只有 seed 0，结果仅用于机制探索，不声明统计显著性。

四个正式配置位于 `configs/`：

- `femnist_hfl_snf_u05_5000.yaml`
- `femnist_hfl_no_snf_u05_5000.yaml`
- `femnist_fl_snf_u05_5000.yaml`
- `femnist_fl_no_snf_u05_5000.yaml`

## GPU 快速路径

训练器只保留一个共享 CNN、一个可复用无动量 SGD 优化器和本轮活跃客户端的扁平 FP32 参数矩阵，不创建 250 份内容相同的长期客户端模型。客户端到边缘、边缘到云的聚合均使用 GPU 张量的样本加权求和。

只物化 37 个固定候选的训练图片；候选训练数据和探针启动时常驻 GPU。完整测试集及逐图片逻辑客户端编号在预留至少 2 GB 显存后仍可容纳时常驻 GPU，否则保留在固定内存并在测试阶段异步分批传输。每个评估点按 250 个本地测试分区累计正确数、样本数和损失，再生成总体测试指标。

CUDA 模式启用 `channels_last`、TF32、固定形状 cuDNN benchmark，以及 PyTorch 1.13 的 `torch.cuda.amp`。模型参数、聚合、softmax、共识计算和 HDF5 均保持 FP32。AMP 是否用于正式实验由 200 轮配对校验自动决定。

## 一键运行四组正式实验

Windows PowerShell 或 VS Code 终端在仓库根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_femnist_250_four_experiments.ps1
```

脚本会依次运行 HFL+SnF、HFL-noSnF、FL+SnF 和 FL-noSnF 四组 5000 轮正式实验。四份配置都会在启动前接受检查，并且必须共同指向 `matlab/result-U-6fixedge_epoch200_varAlpha_0p1_trainable.mat`。脚本固定使用串行模式，避免四组逐轮日志在终端交错；每行输出也会同步保存到正式套件目录的 `logs/job_XX.log`。

默认使用 `D:\Anaconda3\Scripts\conda.exe`、`py37` 环境、0 号 GPU 和批大小 128。路径、环境名、GPU 编号或批大小不同时可显式覆盖：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_femnist_250_four_experiments.ps1 `
  -CondaExecutable "C:\Miniconda3\Scripts\conda.exe" `
  -CondaEnvironment "femnist-probe-py37-gpu" `
  -GpuId 1 `
  -BatchSize 128
```

当前 250 客户端划分下，HFL+SnF 平均每轮约有 97278 张图片。批大小 20 时平均需要约 4882 次优化步骤，批大小 128 时约为 781 次，优化步骤理论减少约 84%。批大小变化会改变本地 SGD 轨迹，因此快速模式四方案之间仍可公平比较，但不能把曲线直接视为旧 batch=20 实验的续跑；应重新开始四方案实验。

若必须延续已经运行的 batch=20 检查点，不要使用一键脚本默认的 128，而应保持原 YAML 的 batch=20，并通过单方案恢复入口继续。训练器兼容新版划分已有的 batch=20 v2 检查点，且会拒绝把该检查点加载到其他批大小。一次性客户端打乱优化仍会生效，因此恢复路径也能减少小索引 CUDA 内核开销。

正式套件仍执行原有 GPU 身份、AMP 门禁和性能门禁校验；首次在新 GPU 环境运行时，应先按下文完成 `calibrate` 与 `benchmark`。

## 服务器环境

在服务器仓库根目录创建 Python 3.7 环境：

```bash
conda env create -f FEMNISTProbe/server_environment_py37_gpu.yaml
conda activate femnist-probe-py37-gpu
python -m FEMNISTProbe.gpu_preflight --gpu_id 0
```

预检会验证 CUDA、cuDNN、FP32 卷积、AMP、TF32 和显存信息，并把结果写到 `result/FEMNISTProbe/gpu_preflight.json`。若服务器已经有兼容环境，可以只运行预检。

一次正式四方案实验必须固定在同一种 GPU 上。不要把部分方案放在 RTX 4090D、其余方案放在 RTX 4060 Laptop。

## 验收与正式运行

以下命令均在仓库根目录执行。

1. 运行 CPU 单元测试：

```bash
python -m unittest discover -s FEMNISTProbe/tests -v
```

2. 在目标 GPU 上执行四方案各 2 轮冒烟：

```bash
python -m FEMNISTProbe.run_suite --mode smoke --gpu_id 0
```

3. 用 HFL+SnF 完成 200 轮 FP32/AMP 串行配对校验：

```bash
python -m FEMNISTProbe.run_suite --mode calibrate --gpu_id 0
```

校验结果写入 `result/FEMNISTProbe/amp_gate.json`。最终完整测试准确率差异必须不超过 0.01，候选 S 和 Q 差异必须分别不超过 0.02，三条主要曲线的最低 Pearson 相关必须不低于 0.98，GradScaler 不得持续回退；AMP 的累计阶段耗时还必须小于 FP32。数值或性能任一门槛失败时，正式套件自动使用 FP32+TF32。

4. 在目标 GPU 上执行 100 轮快速路径/未优化参考路径对照：

```bash
python -m FEMNISTProbe.run_suite --mode benchmark --gpu_id 0
```

在 16 GB 及以上显存设备上，基准还会比较两个方案串行与双进程并行，只有双进程总吞吐达到串行的 1.5 倍且整卡峰值显存占用不超过 90% 时才推荐并行度 2。8 GB 档只执行快速路径对照并固定推荐并行度 1，不启动双进程性能测试。

5. 正式运行四组 5000 轮：

```bash
python -m FEMNISTProbe.run_suite --mode formal --gpu_id 0
```

在 RTX 4060 Laptop 上使用：

```bash
python -m FEMNISTProbe.run_suite --mode formal --gpu_id 0 --parallel 1
```

6. 完整性验收并生成中文报告：

```bash
python -m FEMNISTProbe.analyze_suite --suite_dir result/FEMNISTProbe/suite_formal_时间戳
```

分析器要求每组恰好有 5000 条拓扑记录、101 个探针点和 101 个完整测试点，并核对四组公共哈希、GPU 名称和精度模式完全相同。

## 中断恢复

每个评估点都会先落盘探针和测试结果，再原子替换 `checkpoint_latest.pt`。恢复时使用原配置，并指定检查点：

```bash
python -m FEMNISTProbe.run_experiment \
  --yaml_config_file FEMNISTProbe/configs/femnist_hfl_snf_u05_5000.yaml \
  --resume_checkpoint result/FEMNISTProbe/某次运行/checkpoint_latest.pt
```

恢复逻辑会按检查点坐标截断多余 CSV 和 JSONL 行，从 `next_epoch` 继续，避免重复或遗漏轮次。不得用另一方案、另一 MAT、另一候选清单或不同 AMP 模式的检查点恢复。旧版 37 书写者检查点与新版 250 客户端划分不兼容，加载时会被明确拒绝。

## 结果文件

每个方案目录包含：

- `shared_manifest.json`：Dirichlet 参数、250 端样本数、固定 37 槽位、620 张探针索引及公共哈希。
- `topology_schedule.jsonl`：每轮循环编号、MAT 行、k/n 平衡分组、活跃逻辑客户端和 250 端全量同步范围。
- `probe_probabilities.h5`：客户端、边缘和云端 FP32 概率，按时间点分块并使用 gzip。
- `probe_epoch_summary.csv`：A、C、S、正确 S、错误 S、覆盖率、组内/边缘/云共识和 Q。
- `test_metrics.csv`：250 个本地测试分区汇总得到的完整 77483 张测试图片损失与准确率。
- `stage_timing.csv`：训练、聚合、探针、测试和检查点累计耗时。
- `gpu_monitor.csv`：每 30 秒记录的 GPU 利用率、显存、温度、功率和时钟。
- `checkpoint_latest.pt`：云模型、AMP 缩放器、随机状态与恢复坐标。

HFL 的 HDF5 形状为客户端 `[101,37,620,62]`、边缘 `[101,6,620,62]`、云端 `[101,620,62]`。普通 FL 保留一个未激活的边缘槽位以维持统一读取接口。

客户端探针快照表示“本轮本地训练完成、聚合后全量同步发生之前”的状态：活跃槽位使用本轮本地模型，未活跃槽位使用该轮开始时已同步的云模型。调度日志中的 `synchronized_client_ids` 记录聚合后接收新云模型的全部 250 个逻辑客户端；下一轮所有活跃槽位都从该新云模型加载参数。

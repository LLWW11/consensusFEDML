# FEMNIST MAT 探针实验

本目录实现固定 37 名 FEMNIST 书写者、固定 620 张类别均衡探针和 200 轮 MAT 拓扑显式循环的四方案机制实验。旧 MNIST 入口未被替换，新增代码使用独立入口运行。

## 实验边界

- 数据来自 `dataset/FEMNIST/fed_emnist_train.h5` 和 `fed_emnist_test.h5`。
- seed 0 从 3400 名书写者中无放回固定选择 37 人，并要求其训练数据覆盖全部 62 类。
- 四方案使用相同书写者顺序、初始模型、探针、MAT 文件和训练随机种子。
- MAT 文件固定为 `matlab/result-U-6fixedge_epoch200_varAlpha_0p1_trainable.mat`，固定 `u=0.5`。
- 第 `global_epoch` 轮使用 MAT 第 `global_epoch % 200` 行，循环编号为 `global_epoch // 200`。
- 聚合权重只使用各书写者的真实训练样本数。本地批大小固定为 20，尾批不会丢弃。
- 本实验只有 seed 0，结果仅用于机制探索，不声明统计显著性。

四个正式配置位于 `configs/`：

- `femnist_hfl_snf_u05_5000.yaml`
- `femnist_hfl_no_snf_u05_5000.yaml`
- `femnist_fl_snf_u05_5000.yaml`
- `femnist_fl_no_snf_u05_5000.yaml`

## GPU 快速路径

训练器只保留一个共享 CNN、一个可复用无动量 SGD 优化器和本轮活跃客户端的扁平 FP32 参数矩阵，不创建 37 份长期客户端模型。客户端到边缘、边缘到云的聚合均使用 GPU 张量的样本加权求和。

候选训练数据和探针启动时常驻 GPU。完整测试集在预留至少 2 GB 显存后仍可容纳时常驻 GPU；否则保留在固定内存，并在测试阶段异步分批传输。24 GB 档使用 620 张探针批次和 4096 张测试批次，8 GB 档使用 256 与 1024。

CUDA 模式启用 `channels_last`、TF32、固定形状 cuDNN benchmark，以及 PyTorch 1.13 的 `torch.cuda.amp`。模型参数、聚合、softmax、共识计算和 HDF5 均保持 FP32。AMP 是否用于正式实验由 200 轮配对校验自动决定。

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

恢复逻辑会按检查点坐标截断多余 CSV 和 JSONL 行，从 `next_epoch` 继续，避免重复或遗漏轮次。不得用另一方案、另一 MAT、另一候选清单或不同 AMP 模式的检查点恢复。

## 结果文件

每个方案目录包含：

- `shared_manifest.json`：37 名候选、训练样本数、620 张探针索引及公共哈希。
- `topology_schedule.jsonl`：每轮循环编号、MAT 行、活跃槽位、真实书写者和边缘映射。
- `probe_probabilities.h5`：客户端、边缘和云端 FP32 概率，按时间点分块并使用 gzip。
- `probe_epoch_summary.csv`：A、C、S、正确 S、错误 S、覆盖率、组内/边缘/云共识和 Q。
- `test_metrics.csv`：完整 77483 张测试图片的损失与准确率。
- `stage_timing.csv`：训练、聚合、探针、测试和检查点累计耗时。
- `gpu_monitor.csv`：每 30 秒记录的 GPU 利用率、显存、温度、功率和时钟。
- `checkpoint_latest.pt`：云模型、AMP 缩放器、随机状态与恢复坐标。

HFL 的 HDF5 形状为客户端 `[101,37,620,62]`、边缘 `[101,6,620,62]`、云端 `[101,620,62]`。普通 FL 保留一个未激活的边缘槽位以维持统一读取接口。

客户端探针快照表示“本轮本地训练完成、聚合后全量同步发生之前”的状态：活跃槽位使用本轮本地模型，未活跃槽位使用该轮开始时已同步的云模型。调度日志中的 `synchronized_candidate_slots` 与 `synchronized_writer_ids` 则记录聚合后接收新云模型的全部 37 名候选。下一轮所有活跃客户端都从该新云模型加载参数。

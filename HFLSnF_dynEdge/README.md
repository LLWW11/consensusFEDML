# 动态拓扑联邦学习实验

本目录使用 `matlab/result-U-6fixedge_epoch200.mat` 中保存的候选槽位、边缘分组和拓扑信息，在 MNIST 上运行四组固定边缘对照实验：

- HFL-SnF-fixed
- HFL-noSnF-fixed
- FL-SnF
- FL-noSnF

MATLAB 中的 37 个物理客户端表示固定候选槽位，不直接等同于数据集中的真实客户端编号。实验启动时只从 200 个真实客户端中无放回抽取一次 37 个客户端，建立“MAT 槽位 → 真实客户端”的固定映射；后续所有 epoch 都复用这一映射，不再重新抽取 37 人。

## 每个 epoch 的执行流程

1. 实验初始化时，按 `random_seed` 从 200 个真实客户端中无放回抽取 37 人，并按抽取顺序固定对应 MAT 的 37 个槽位。
2. 每个 epoch 读取 MAT 对应行保存的候选槽位身份及边缘分组，再通过固定映射得到真实客户端；该过程不打乱候选列表，也不进行额外随机采样。
3. 仅 MAT 当前行启用的客户端基于各自持久本地模型完成一个本地 epoch；其他客户端本轮不训练。
4. 本地训练后，37个候选客户端分别对同一批固定100张测试图片执行一次批量推理；探针每类10张，全部epoch和四种方案使用相同索引与顺序。
5. HFL 先对活跃客户端执行组内样本数加权聚合，再执行云端聚合；FL 将活跃客户端模型直接交给云端聚合。
6. 边缘和云模型也分别对同一批100张图片执行批量推理；未启用边缘槽位保存为整块NaN。零参与 epoch 沿用上一有效云模型，仍形成一条完整探针记录。
7. 三层探针完整后写入内存缓冲，每10个epoch原子更新压缩NPZ；正常结束或异常退出时再保存最后一个完整epoch前缀。
8. 聚合完成后，将最新云模型下发给全部 200 个客户端；零参与 epoch 不产生新模型，但仍把上一有效云模型下发给全部客户端。
9. 下发完成后，200 个真实客户端分别加载自己的持久本地状态并评估自己的数据分区；全局准确率严格按“总正确数 ÷ 总样本数”计算。固定探针只用于监控，不替代完整测试集 `test_acc`。

所有 YAML 都固定使用 `model_distribution_scope: "all"`。MAT 当前 epoch 参与人数为 0 时，本轮不产生新的聚合模型，而是沿用上一有效云模型完成全量下发和逐客户端评估。

四份正式配置、根配置和两份冒烟配置均启用以下固定探针参数：

```yaml
probe_output_format: "npz"
probe_source: "test"
probe_samples_per_class: 10
probe_seed: 0
probe_inference_batch_size: 100
probe_checkpoint_interval: 10
probe_npz_file: "probe_probabilities.npz"
probe_summary_file: "probe_epoch_summary.csv"
```

`probe_seed` 只用于固定探针抽样，不改变训练随机状态。正式配置的推理批量等于100，因此每个客户端、边缘模型和云模型各用一次前向传播处理全部探针。

## 运行环境

本机训练环境为 Conda 的 `py37`。应通过 `conda run` 启动，以便正确加载 OpenSSL 和 FedML 依赖：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  torch_hierarchicalfl_mnist_lr_step_by_step_example.py --cf fedml_config.yaml
```

## 单独运行四组实验

```powershell
# HFL-SnF-fixed
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  torch_hierarchicalfl_mnist_lr_step_by_step_example.py --cf configs/fedml_config_hfl_snf_fixed_u05.yaml

# HFL-noSnF-fixed
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  torch_hierarchicalfl_mnist_lr_step_by_step_example.py --cf configs/fedml_config_hfl_no_snf_fixed_u05.yaml

# FL-SnF
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  torch_hierarchicalfl_mnist_lr_step_by_step_example.py --cf configs/fedml_config_fl_snf_u05.yaml

# FL-noSnF
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  torch_hierarchicalfl_mnist_lr_step_by_step_example.py --cf configs/fedml_config_fl_no_snf_u05.yaml
```

也可以依次运行全部四组 200 轮实验：

```powershell
.\run_fixed_u05_experiments.ps1
```

## 冒烟测试

HFL-SnF-fixed 的 3 轮冒烟配置：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  torch_hierarchicalfl_mnist_lr_step_by_step_example.py --cf configs/fedml_config_smoke_hfl_snf_fixed_u05.yaml
```

FL-noSnF 的 1 轮直接云聚合验证配置：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  torch_hierarchicalfl_mnist_lr_step_by_step_example.py --cf configs/fedml_config_smoke_fl_no_snf_u05.yaml
```

## 生成四组实验分析报告

分析相关代码统一放在 `result/originalData` 中，与项目根目录的训练和实验代码分开。`result/originalData/analyze_experiment_suite.py` 会从一个指定批次中识别且只识别以下四组完整实验：HFL-SnF、HFL-noSnF、FL-SnF 和 FL-noSnF。说明文件和其他非实验目录会被忽略；场景缺失、重复、轮数不一致或概率探针非法时会输出明确错误，不生成不完整报告。

### 在终端中运行

从项目目录执行：

```powershell
python result\originalData\analyze_experiment_suite.py `
  --input-dir "result\originalData\1"
```

可选参数如下：

- `--smooth-window 10`：设置趋势和共识曲线的尾随平滑窗口；
- `--output-root "result\1结果和分析"`：指定自动生成报告目录的根目录；
- `--output-dir "D:\temp\本次分析"`：直接指定本次输出目录，该目录必须不存在或为空；
- `--result-root`：旧命令的兼容参数，作用等同于 `--input-dir`；
- `--experiment-dir`：需要恰好重复四次，用于显式指定四个实验目录。

默认输出目录格式为：

```text
result\1结果和分析\analysis_<批次名>_<实际轮数>rounds_<数据日期>
```

若同名分析目录已经存在，脚本会自动追加序号，避免覆盖旧报告。完整分析包包括简体中文 `分析报告.md`、8张300 DPI图片、逐轮和汇总CSV、数据质量检查及带SHA-256来源哈希的 `analysis_manifest.json`。

### 在 VS Code 中点击运行

1. 打开 `result/originalData/run_report_from_ide.py`。
2. 在文件顶部“只需修改这里”区域设置 `INPUT_BATCH_FOLDER`，例如 `"1"` 或 `"2"`。
3. 点击编辑器右上角的“运行 Python 文件”按钮。
4. 在控制台查看输入目录、输出目录、报告绝对路径和四组实验主要指标。

### 在 PyCharm 中点击运行

1. 打开 `result/originalData/run_report_from_ide.py`。
2. 修改顶部的 `INPUT_BATCH_FOLDER`；需要时同时修改 `SMOOTH_WINDOW`。
3. 右键文件并选择“运行”，或点击编辑器旁的绿色运行按钮。

IDE入口根据脚本在 `result/originalData` 中的固定位置推导项目根目录，不依赖 VS Code、PyCharm 的当前工作目录，也不读取命令行参数。因此不需要个人化的 `.vscode/launch.json` 或 `.idea/workspace.xml`。

分析脚本测试可通过以下命令运行：

```powershell
python result\originalData\test_analyze_experiment_suite.py
```

分析程序优先读取固定探针NPZ；如果历史结果没有NPZ，则自动回退到原来的三份单图CSV。NPZ一旦存在但损坏，程序会直接报错，不会用旧CSV静默掩盖问题。四方案使用NPZ时，探针哈希、索引或真实标签有任一处不同，分析都会终止。

## 固定探针指标口径

对每个epoch、每张图片分别计算：

- 一致性 (A)：用归一化广义JSD衡量客户端之间的概率分歧；
- 确定性 (C)：一减去各客户端自身归一化熵的平均；
- 纯有效共识 (S=A\times C)；
- 正确有效共识：多数客户端类别等于真实标签时保留该图片的 (S)；
- 错误有效共识：多数客户端类别不等于真实标签时保留该图片的 (S)。

随后才对100张图片取均值和四分位数，禁止先平均不同图片的概率向量再计算熵或JSD。候选37人和当轮实际活跃客户端分别计算；活跃客户端少于2人时，其共识指标为空值，但活跃覆盖率仍保留。报告还给出覆盖率乘以活跃正确共识，用于同时反映“参与多少”和“参与者是否正确形成共识”。

“达成更快”通过前50轮、前100轮归一化曲线面积及MA10稳定越过0.60、0.70、0.80的epoch判断。历史最佳曲线仅作辅助展示，不能作为唯一结论。当前正式实验只有单随机种子，论文结论建议至少运行5个匹配随机种子。

## 单元测试

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  -m unittest -v test_topology_schedule.py test_hierarchical_sampling.py
```

## 输出文件

每个实验写入独立的 `result/` 子目录，包含：

- `train_acc.txt`、`train_loss.txt`
- `test_acc.txt`、`test_loss.txt`
- `probe_probabilities.npz`
- `probe_epoch_summary.csv`
- `topology_metadata.json`
- `topology_schedule.jsonl`

`probe_probabilities.npz` 主要字段如下：

- `client_probabilities`：`[epoch, 37, 100, K]`；
- `edge_probabilities`：`[epoch, edge_slot, 100, K]`，未启用槽位为NaN；
- `cloud_probabilities`：`[epoch, 100, K]`；
- `active_client_mask`、`edge_active_mask`：逐epoch的候选和边缘活跃掩码；
- `client_ids`、`probe_indices`、`true_labels`、`global_epochs`；
- `probe_set_hash`、`completed_epochs` 和格式版本。

`probe_epoch_summary.csv` 每个epoch一行，包含候选与活跃口径的A、C、纯/正确/错误有效共识、S的四分位数、活跃人数与覆盖率、覆盖加权活跃正确共识、固定探针云端准确率和云端真实类别平均概率。分析报告会从NPZ重新计算并核对该摘要，而不是直接信任派生结果。

没有配置 `probe_output_format` 的旧实验仍使用 `probe_client_pre.csv`、`probe_edge_post.csv`、`probe_cloud_post.csv` 和可选的 `probe_meta.csv`，以保证历史结果可继续分析。

`topology_schedule.jsonl` 每行对应一个展平后的本地 epoch，主要记录：

- 通信轮、组内通信轮、本地 epoch 和 `global_epoch`；
- 固定 37 个真实候选客户端及其槽位顺序；每行的 `candidate_client_indexes` 应保持一致；
- MAT 各组人数、槽位到真实客户端的分组结果和本轮实际训练与聚合的活跃客户端；
- 是否完成聚合、参数下发范围和实际接收模型的 200 个客户端编号。

## 文件说明

文件名带 `_test` 的模块主要从 FedML 对应实现调整而来；其余模块为本项目新增或重写的实验代码。

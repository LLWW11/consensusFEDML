# HFLSnF KGE V3：固定人数与动态拓扑四组实验

本工程用于在 FB15k-237 上训练四种联邦 TransE 场景：

- `HFLSnF`：分层联邦学习，使用 SnF 动态拓扑；
- `FLSnF`：普通联邦学习，使用 SnF 动态选人；
- `HFLnoSnF`：分层联邦学习，不使用 SnF；
- `FLnoSnF`：普通联邦学习，不使用 SnF。

当前保留两套互不混用的实验合同：

1. 固定人数对照：每轮人数分别固定为 35、25、15、5；
2. 动态 MATLAB 回放：每轮参与者和分组直接来自 `.mat`，不再裁剪、补齐或重新分组。

不再使用 5 轮 CUDA 门禁。长实验前只运行一个两轮 CPU 烟雾测试，成功后的临时结果会自动删除。

## 固定人数对照

| 实验臂 | 架构 | SnF | 每轮人数 | 每轮组数 | 配置 |
|---|---|---:|---:|---:|---|
| `hflsnf_k35` | HFL | 是 | 35 | 6 | `formal_fixed_count_hflsnf_k35_seed42_150round_cuda.yaml` |
| `flsnf_k25` | FL | 是 | 25 | 1 | `formal_fixed_count_flsnf_k25_seed42_150round_cuda.yaml` |
| `hflnosnf_k15` | HFL | 否 | 15 | 6 | `formal_fixed_count_hflnosnf_k15_seed42_150round_cuda.yaml` |
| `flnosnf_k5` | FL | 否 | 5 | 1 | `formal_fixed_count_flnosnf_k5_seed42_150round_cuda.yaml` |

固定人数实验中的 `.mat` 只为 SnF 场景提供候选信号。运行时会按照 YAML 的
`client_num_per_round` 裁剪或补齐参与者；HFL 场景再按固定组数重新分组。因此，这套实验适合复现已有四组结果，但不代表原始动态拓扑。

## 动态 MATLAB 回放

四组动态配置统一读取：

```text
matlab/result-U-6fixedge_epoch200_varAlpha_0p1_trainable.mat
```

实验使用 `util=0.5`、前 150 轮以及严格调度模式。以下统计由配置合同直接读取 `.mat` 得到：

| 实验臂 | 架构 | SnF | 每轮人数范围 | 每轮组数范围 | 参与集合种类 | 拓扑种类 | 配置 |
|---|---|---:|---:|---:|---:|---:|---|
| `hflsnf` | HFL | 是 | 30–36 | 3–6 | 49 | 147 | `formal_dynamic_mat_hflsnf_seed42_150round_cuda.yaml` |
| `flsnf` | FL | 是 | 21–31 | 1 | 141 | 147 | `formal_dynamic_mat_flsnf_seed42_150round_cuda.yaml` |
| `hflnosnf` | HFL | 否 | 14–25 | 1–6 | 143 | 144 | `formal_dynamic_mat_hflnosnf_seed42_150round_cuda.yaml` |
| `flnosnf` | FL | 否 | 5–9 | 1 | 117 | 117 | `formal_dynamic_mat_flnosnf_seed42_150round_cuda.yaml` |

动态配置中的 `client_num_per_round: 37` 表示训练器允许使用的候选池上限，不表示实际每轮固定有 37 人。实际人数、参与者编号和 HFL 分组全部由 `.mat` 决定，并写入结果目录的参与统计。

`MatlabTopologySchedule` 已内置在 `core/matlab_topology_schedule.py`，运行动态实验不依赖旁边的 `HFLSnF_dynEdge` 工程。

## 公共训练条件

两套实验共享以下设置：

- 数据集：FB15k-237；
- 客户端池：37 个客户端，按头实体均衡划分；
- 模型：256 维 TransE，L1 距离；
- 负采样：双向自对抗负采样，每个正样本 256 个负样本；
- 本地训练：每轮 2 个 epoch，客户端 Adam 每轮重置；
- 聚合：逐行出现次数加权；
- 服务器优化器：FedAdam；
- 通信轮数：150；
- 随机种子：42；
- 最终测试：完整 20,466 条测试三元组。

动态实验的目的，是先观察 `.mat` 原始参与和分组变化是否影响收敛。它还不是严格的单因素消融，因为四种场景的动态人数范围、SnF 状态和 HFL/FL 架构仍同时变化。

## 工程结构

```text
HFLSnF_KG_v3/
├─ configs/
│  ├─ smoke_four_scenario_pipeline_cpu.yaml
│  ├─ formal_fixed_count_*.yaml
│  └─ formal_dynamic_mat_*.yaml
├─ core/
│  ├─ matlab_topology_schedule.py
│  ├─ topology.py
│  └─ server_optimization.py
├─ fedml_kge/
├─ matlab/
│  └─ result-U-6fixedge_epoch200_varAlpha_0p1_trainable.mat
├─ tasks/kge/
├─ tests/
├─ run_federated_transe.py
├─ run_fixed_count_four_scenarios.py
├─ run_fixed_count_four_scenarios_from_ide.py
└─ run_best_checkpoint_official_evaluation.py
```

## 推荐执行顺序

所有命令均在工作区根目录 `D:\1\1myworkcode` 下执行。

### 1. 校验配置和 MATLAB 调度

```powershell
python -m HFLSnF_KG_v3.run_fixed_count_four_scenarios validate
```

该命令会校验四个固定人数 YAML、四个动态 YAML、唯一烟雾配置，以及四组动态调度的 150 轮哈希、人数、组数和拓扑变化。它不会启动训练。

### 2. 运行两轮 CPU 烟雾测试

```powershell
python -m HFLSnF_KG_v3.run_fixed_count_four_scenarios smoke
```

烟雾测试只检查本地 TransE、逐行聚合、服务器 FedAdam、评估和结果写入能否贯通，不用于判断最终 MRR。测试成功后临时结果自动删除，失败结果会保留用于诊断。

### 3. 运行四组动态拓扑正式实验

```powershell
python -m HFLSnF_KG_v3.run_fixed_count_four_scenarios dynamic150
```

当前动态正式配置在每个通信轮次内执行 3 个本地训练周期，并在该轮聚合完成后立即进行一次验证评估。每组结束后程序会核验 150 轮实际参与者、分组、调度哈希、非零更新、FedAdam 步数和结果元数据；若某组合同失败，将停止后续组。

只运行一个动态实验臂：

```powershell
python -m HFLSnF_KG_v3.run_fixed_count_four_scenarios dynamic150 `
  --arm hflsnf
```

动态实验臂可选：

- `hflsnf`
- `flsnf`
- `hflnosnf`
- `flnosnf`

### 4. 需要复现时再运行固定人数对照

```powershell
python -m HFLSnF_KG_v3.run_fixed_count_four_scenarios formal150
```

只运行一个固定人数实验臂：

```powershell
python -m HFLSnF_KG_v3.run_fixed_count_four_scenarios formal150 `
  --arm hflnosnf_k15
```

固定人数实验臂可选：

- `hflsnf_k35`
- `flsnf_k25`
- `hflnosnf_k15`
- `flnosnf_k5`

## 完整官方测试

训练完成后，把正式结果目录显式传给官方测试入口：

```powershell
python -m HFLSnF_KG_v3.run_best_checkpoint_official_evaluation `
  --result-dir "D:\1\1myworkcode\HFLSnF_KG_v3\results\正式结果目录" `
  --using-gpu `
  --require-cuda `
  --query-batch-size 64 `
  --candidate-batch-size 8192
```

默认读取 `model_best.pt`，并在该结果目录的 `full_official_evaluation/` 下写入完整头预测、尾预测和综合 filtered 指标。

## 结果解释边界

动态实验完成后，优先比较以下内容，而不是只看最终 MRR：

1. `participation_summary.json` 中的实际人数范围和参与频次；
2. 每轮客户端集合与 HFL 分组是否和 `.mat` 一致；
3. MRR 随累计有效样本量的变化，而不只是随通信轮数的变化；
4. 头预测、尾预测和综合 MRR；
5. 四组达到最佳验证 MRR 的轮次及其后是否出现回落。

如果动态拓扑组表现明显变化，下一步应设计“同一人数序列、只改变分组”以及“同一参与集合、只改变 SnF”的严格配对实验，避免把人数、覆盖率和架构差异误认为单独的拓扑收益。

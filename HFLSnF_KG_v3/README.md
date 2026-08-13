# HFLSnF KGE V3：最终动态拓扑配置与历史实验归档

本工程用于在 FB15k-237 上训练四种联邦 TransE 场景：

- `HFLSnF`：分层联邦学习，使用 SnF 动态拓扑；
- `FLSnF`：普通联邦学习，使用 SnF 动态选人；
- `HFLnoSnF`：分层联邦学习，不使用 SnF；
- `FLnoSnF`：普通联邦学习，不使用 SnF。

当前主结果采用 `topology_util=0.6`、关闭服务器偏差修正的动态MATLAB拓扑，覆盖HFLSnF/HFLnoSnF/FLnoSnF与随机种子42、2024、2025。九份正式YAML位于 `configs/` 根目录；此前四场景、参数筛选和人数消融配置均已归档到 `configs/zOld/`，只用于审计或复现。

历史实验仍保留四套互不混用的合同：

1. 固定人数对照：每轮人数分别固定为 35、25、15、5；
2. 动态 MATLAB 回放：每轮参与者和分组直接来自 `.mat`，不再裁剪、补齐或重新分组；
3. `varAlpha=0.5` 动态 MATLAB 回放：独立使用0p5可训练MAT文件，不覆盖0p1基线；
4. HFLKGE 人数单因素消融：固定 HFL 和全部训练条件，每轮随机选择客户端，比较 36、30、24、18、12、6 六档人数。

不再使用 5 轮 CUDA 门禁。长实验前只运行一个两轮 CPU 烟雾测试，成功后的临时结果会自动删除。

## 当前最终动态拓扑配置

最终配置统一使用MATLAB alpha=0.1调度、`topology_util=0.6`、FedAdam学习率0.05、`tau=0.001`、`betas=(0.9, 0.99)`、`server_bias_correction=false`、每轮3个本地epoch、150轮通信和逐轮验证。训练阶段不自动执行测试集评估。

九份配置按HFLSnF/HFLnoSnF/FLnoSnF和随机种子42、2024、2025组织，完整文件表与哈希合同见 [`configs/README.md`](configs/README.md)。运行单份配置示例：

```powershell
python -m HFLSnF_KG_v3.run_federated_transe --cf HFLSnF_KG_v3/configs/final_dynamic_fedadam_hflsnf_u0p6_bcfalse_seed42_150round_cuda.yaml
```

HFLSnF、HFLnoSnF与FLnoSnF使用不同的MAT动态参与、覆盖和分组过程，因此当前结果应表述为动态编排造成的系统级差异，不能单独归因于SnF或分层机制。FLnoSnF在`util=0.6`的前150轮只覆盖19/37个客户端，其中18个客户端永久缺席；这一限制会由最终批量合同显式校验。

## 历史固定人数对照

| 实验臂 | 架构 | SnF | 每轮人数 | 每轮组数 | 配置 |
|---|---|---:|---:|---:|---|
| `hflsnf_k35` | HFL | 是 | 35 | 6 | `configs/zOld/fixed_count_four_scenarios/formal_fixed_count_hflsnf_k35_seed42_150round_cuda.yaml` |
| `flsnf_k25` | FL | 是 | 25 | 1 | `configs/zOld/fixed_count_four_scenarios/formal_fixed_count_flsnf_k25_seed42_150round_cuda.yaml` |
| `hflnosnf_k15` | HFL | 否 | 15 | 6 | `configs/zOld/fixed_count_four_scenarios/formal_fixed_count_hflnosnf_k15_seed42_150round_cuda.yaml` |
| `flnosnf_k5` | FL | 否 | 5 | 1 | `configs/zOld/fixed_count_four_scenarios/formal_fixed_count_flnosnf_k5_seed42_150round_cuda.yaml` |

固定人数实验中的 `.mat` 只为 SnF 场景提供候选信号。运行时会按照 YAML 的
`client_num_per_round` 裁剪或补齐参与者；HFL 场景再按固定组数重新分组。因此，这套实验适合复现已有四组结果，但不代表原始动态拓扑。

## 历史动态MATLAB回放基线

四组动态配置统一读取：

```text
matlab/result-U-6fixedge_epoch200_varAlpha_0p1_trainable.mat
```

实验使用 `util=0.5`、前 150 轮以及严格调度模式。以下统计由配置合同直接读取 `.mat` 得到：

| 实验臂 | 架构 | SnF | 每轮人数范围 | 每轮组数范围 | 参与集合种类 | 拓扑种类 | 配置 |
|---|---|---:|---:|---:|---:|---:|---|
| `hflsnf` | HFL | 是 | 30–36 | 3–6 | 49 | 147 | `configs/zOld/dynamic_alpha0p1_legacy/formal_dynamic_mat_hflsnf_seed42_150round_cuda.yaml` |
| `flsnf` | FL | 是 | 21–31 | 1 | 141 | 147 | `configs/zOld/dynamic_alpha0p1_legacy/formal_dynamic_mat_flsnf_seed42_150round_cuda.yaml` |
| `hflnosnf` | HFL | 否 | 14–25 | 1–6 | 143 | 144 | `configs/zOld/dynamic_alpha0p1_legacy/formal_dynamic_mat_hflnosnf_seed42_150round_cuda.yaml` |
| `flnosnf` | FL | 否 | 5–9 | 1 | 117 | 117 | `configs/zOld/dynamic_alpha0p1_legacy/formal_dynamic_mat_flnosnf_seed42_150round_cuda.yaml` |

动态配置中的 `client_num_per_round: 37` 表示训练器允许使用的候选池上限，不表示实际每轮固定有 37 人。实际人数、参与者编号和 HFL 分组全部由 `.mat` 决定，并写入结果目录的参与统计。

`MatlabTopologySchedule` 已内置在 `core/matlab_topology_schedule.py`，运行动态实验不依赖旁边的 `HFLSnF_dynEdge` 工程。

## 历史varAlpha=0.5动态MAT回放

这套独立合同读取：

```text
matlab/result-U-6fixedge_epoch200_varAlpha_0p5_trainable.mat
```

前150轮没有空参与轮次，四个场景的统计如下：

| 实验臂 | 架构 | SnF | 每轮人数范围 | 每轮组数范围 | 参与集合种类 | 拓扑种类 | 配置 |
|---|---|---:|---:|---:|---:|---:|---|
| `hflsnf` | HFL | 是 | 10–37 | 3–6 | 48 | 148 | `configs/zOld/dynamic_alpha0p5_legacy/formal_dynamic_mat_varalpha0p5_hflsnf_seed42_150round_cuda.yaml` |
| `flsnf` | FL | 是 | 8–35 | 1 | 134 | 148 | `configs/zOld/dynamic_alpha0p5_legacy/formal_dynamic_mat_varalpha0p5_flsnf_seed42_150round_cuda.yaml` |
| `hflnosnf` | HFL | 否 | 6–32 | 1–6 | 145 | 145 | `configs/zOld/dynamic_alpha0p5_legacy/formal_dynamic_mat_varalpha0p5_hflnosnf_seed42_150round_cuda.yaml` |
| `flnosnf` | FL | 否 | 2–12 | 1 | 124 | 124 | `configs/zOld/dynamic_alpha0p5_legacy/formal_dynamic_mat_varalpha0p5_flnosnf_seed42_150round_cuda.yaml` |

专用入口默认只运行 `hflsnf`，显式传入 `--arm all` 才会顺序运行四个场景。每个正式结果完成后都会校验逐轮参与人数、分组、调度哈希和FedAdam步数。

## 历史HFLKGE客户端人数单因素消融

这组实验专门回答“平台期差距是否主要由每轮客户端数量造成”。六组都使用 HFL、六个边缘组、相同种子、相同本地训练和相同 FedAdam；行为配置中只有 `client_num_per_round` 分别为 36、30、24、18、12、6。

每一轮都从 37 个客户端中无放回随机抽取指定人数，然后平均分入六个组。随机数由实验种子和轮次共同决定，因此参与客户端会逐轮改变，同时相同配置可以完全复现。该实验不读取 SnF 或 MATLAB 的客户端选择结果。

| 实验臂 | 每轮人数 | 每轮组数 | 配置 |
|---|---:|---:|---|
| `hflkge_k36` | 36 | 6 | `configs/zOld/client_count_ablation/formal_hflkge_client_count_k36_seed42_150round_cuda.yaml` |
| `hflkge_k30` | 30 | 6 | `configs/zOld/client_count_ablation/formal_hflkge_client_count_k30_seed42_150round_cuda.yaml` |
| `hflkge_k24` | 24 | 6 | `configs/zOld/client_count_ablation/formal_hflkge_client_count_k24_seed42_150round_cuda.yaml` |
| `hflkge_k18` | 18 | 6 | `configs/zOld/client_count_ablation/formal_hflkge_client_count_k18_seed42_150round_cuda.yaml` |
| `hflkge_k12` | 12 | 6 | `configs/zOld/client_count_ablation/formal_hflkge_client_count_k12_seed42_150round_cuda.yaml` |
| `hflkge_k6` | 6 | 6 | `configs/zOld/client_count_ablation/formal_hflkge_client_count_k6_seed42_150round_cuda.yaml` |

六档人数都能被六整除，因此每组分别有 6、5、4、3、2、1 个客户端，不会出现组大小不均衡。`ablation_arm`、运行名和调度哈希会随人数派生变化，但不属于训练行为变量。

## 训练条件说明

当前九份最终主配置共享以下核心设置：

- 数据集：FB15k-237；
- 客户端池：37 个客户端，按头实体均衡划分；
- 模型：256 维 TransE，L1 距离；
- 负采样：双向自对抗负采样，每个正样本 256 个负样本；
- 本地训练：每轮3个epoch，客户端Adam每轮重置；
- 聚合：逐行出现次数加权；
- 服务器优化器：FedAdam；
- 通信轮数：150；
- 随机种子：42、2024、2025；
- 最终测试：训练YAML不自动评估测试集，验证选型后使用独立官方评估入口。

归档实验保留各自原有的训练合同，不应套用上述最终配置参数。动态实验仍不是严格的SnF单因素消融，因为不同实验臂的参与人数、覆盖和分组会同时变化。

## 工程结构

```text
HFLSnF_KG_v3/
├─ configs/
│  ├─ README.md
│  ├─ final_dynamic_fedadam_hflsnf_*.yaml
│  ├─ final_dynamic_fedadam_hflnosnf_*.yaml
│  ├─ final_dynamic_fedadam_flnosnf_*.yaml
│  └─ zOld/
│     ├─ README.md
│     ├─ fedadam_stage1/
│     ├─ fedadam_stage2_screen/
│     ├─ dynamic_alpha0p1_legacy/
│     ├─ dynamic_alpha0p5_legacy/
│     ├─ fixed_count_four_scenarios/
│     ├─ client_count_ablation/
│     └─ smoke/
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
├─ run_dynamic_mat_varalpha0p5.py
├─ run_hflkge_client_count_ablation.py
├─ run_fedadam_stage1.py
├─ run_fedadam_stage2.py
├─ run_final_dynamic_fedadam.py
└─ run_best_checkpoint_official_evaluation.py
```

## 推荐执行顺序

所有命令均在工作区根目录 `D:\1\1myworkcode` 下执行。

当前正式配置可以直接通过 `run_federated_transe --cf` 运行单组，示例见“当前最终动态拓扑配置”。推荐使用以下最终九组批量入口：

```powershell
python -m HFLSnF_KG_v3.run_final_dynamic_fedadam validate
python -m HFLSnF_KG_v3.run_final_dynamic_fedadam formal150
```

第二条命令按seed42、2024、2025执行，每个seed内依次运行HFLSnF、HFLnoSnF、FLnoSnF。失败后按终端输出恢复，或执行：

```powershell
python -m HFLSnF_KG_v3.run_final_dynamic_fedadam formal150 --resume "HFLSnF_KG_v3/results/final_dynamic_fedadam_batch_<时间戳>/batch_summary.json"
```

以下批量入口均属于历史合同校验或复现。

### 1. 校验历史配置和MATLAB调度

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

### 4. 运行varAlpha=0.5动态MAT实验

同步到服务器后先做只读校验：

```powershell
python -m HFLSnF_KG_v3.run_dynamic_mat_varalpha0p5 validate
```

默认只运行一次HFLSnF正式实验：

```powershell
python -m HFLSnF_KG_v3.run_dynamic_mat_varalpha0p5 dynamic150
```

指定其他单个实验臂：

```powershell
python -m HFLSnF_KG_v3.run_dynamic_mat_varalpha0p5 dynamic150 `
  --arm hflnosnf
```

顺序运行四个实验臂：

```powershell
python -m HFLSnF_KG_v3.run_dynamic_mat_varalpha0p5 dynamic150 `
  --arm all
```

### 5. 需要复现时再运行固定人数对照

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

### 6. 运行HFLKGE客户端人数单因素消融

先校验六份配置、150 轮随机调度哈希以及“除人数和派生身份字段外完全一致”的合同：

```powershell
python -m HFLSnF_KG_v3.run_hflkge_client_count_ablation validate
```

顺序运行 K=36、30、24、18、12、6 六组正式实验：

```powershell
python -m HFLSnF_KG_v3.run_hflkge_client_count_ablation formal150
```

只运行其中一组：

```powershell
python -m HFLSnF_KG_v3.run_hflkge_client_count_ablation formal150 `
  --arm hflkge_k24
```

已有五组结果时，只补跑新增的 K=6：

```powershell
python -m HFLSnF_KG_v3.run_hflkge_client_count_ablation formal150 `
  --arm hflkge_k6
```

### 7. 历史FedAdam阶段一八组实验

阶段一固定使用 `varAlpha=0.1` 的MAT调度、`topology_util=0.5`、本地3周期、每轮评估、40轮通信、随机种子42和开启偏差修正。四套服务器参数分别为：

| 参数组 | 服务器学习率 | tau |
| --- | ---: | ---: |
| 1 | 0.1 | 0.001 |
| 2 | 0.05 | 0.001 |
| 3 | 0.03 | 0.001 |
| 4 | 0.05 | 0.01 |

每套参数都按HFLSnF、HFLnoSnF的顺序运行，因此完整批次共8组。正式运行前先执行只读校验：

```powershell
python -m HFLSnF_KG_v3.run_fedadam_stage1 validate
```

校验通过后，一条命令顺序运行全部8组：

```powershell
python -m HFLSnF_KG_v3.run_fedadam_stage1 formal40
```

启动后会在 `results/fedadam_stage1_batch_<时间戳>/batch_summary.json` 写入批次清单。每组完成后，程序会核验40行逐轮指标、FedAdam步数、配置快照、MAT调度哈希和拓扑统计，并在该组结果目录写入 `fedadam_stage1_formal40_contract.json`。任一训练或结果合同失败时，后续实验不会启动。

修复故障后，从原批次失败位置继续：

```powershell
python -m HFLSnF_KG_v3.run_fedadam_stage1 formal40 `
  --resume "HFLSnF_KG_v3/results/fedadam_stage1_batch_<时间戳>/batch_summary.json"
```

归档后新建的阶段一批次仍可按原方式恢复。归档前生成的批次清单保存旧配置路径，移动配置后不再支持 `--resume`；不得为恢复旧批次而改写历史清单。

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
5. 两个最终实验臂达到最佳验证 MRR 的轮次及其后是否出现回落；历史四场景复现仍按四组比较。

如果动态拓扑组表现明显变化，下一步应设计“同一人数序列、只改变分组”以及“同一参与集合、只改变 SnF”的严格配对实验，避免把人数、覆盖率和架构差异误认为单独的拓扑收益。

## 历史FedAdam阶段二全因子实验

阶段二固定服务器学习率为0.05、tau为0.001，先使用随机种子42运行 `topology_util=0.5/0.6`、偏差修正开启/关闭和HFLSnF/HFLnoSnF的8组全因子筛选。程序随后自动选择胜出配置；没有配置达到预注册门槛时，会改为选择挑战组并明确标记，不能误称为胜出配置。

运行前只读校验8份筛选YAML、150轮MAT调度、客户端覆盖和完整哈希：

```powershell
python -m HFLSnF_KG_v3.run_fedadam_stage2 validate
```

启动8组筛选：

```powershell
python -m HFLSnF_KG_v3.run_fedadam_stage2 screen150
```

命令会创建 `results/fedadam_stage2_batch_<时间戳>/batch_summary.json`。筛选通过后，程序在该批次目录自动固化8组多种子复验配置和6组固定参与人数对照配置，共计最多22组；后续配置绑定原筛选清单，不能手工替换候选。使用归档配置新建的批次仍可按下述流程恢复。

继续运行随机种子2024和2025下的基线与候选复验：

```powershell
python -m HFLSnF_KG_v3.run_fedadam_stage2 confirm150 `
  --resume "HFLSnF_KG_v3/results/fedadam_stage2_batch_<时间戳>/batch_summary.json"
```

复验通过后，运行三个随机种子的固定参与人数预算对照：

```powershell
python -m HFLSnF_KG_v3.run_fedadam_stage2 controls150 `
  --resume "HFLSnF_KG_v3/results/fedadam_stage2_batch_<时间戳>/batch_summary.json"
```

任一训练、结果合同或阶段汇总失败时会立即停止。新批次修复后可使用相同动作和同一 `--resume` 路径恢复，已经通过的项目不会重复训练。归档前的阶段一、二清单绑定旧配置绝对路径，阶段二还绑定配置SHA256；这些旧清单只用于审计，不再支持恢复，也不得修改其路径或绑定哈希。每组结果目录写入 `fedadam_stage2_<阶段>_formal150_contract.json`；批次目录保存结构化分析JSON、简体中文分析说明和全程、冷启动、后20轮及参与预算图。

阶段二保持每轮验证。曲线使用全部150个数据点，marker每5轮显示一次；第131至150轮斜率绝对值超过0.0005时只称“后20轮均值”，不称最终平台。批量训练设置 `evaluate_test_after_training=false`，筛选、复验和人数对照均不读取测试集；完成验证选型后，再使用“完整官方测试”入口评估最终选定检查点。

# HFLSnF_KG_v3

## 项目定位

V3是只面向以下目标的独立工程：

- FB15k-237知识图谱补全；
- 37个候选知识客户端；
- HFLSnF动态客户端参与；
- MATLAB逐轮动态边缘分组；
- 全局头、尾filtered MRR；
- 37客户端最终目标量级为可信且可复现的`0.30至0.35`。

V3没有复制V2的`results/`、旧检查点、Python缓存和A/B/C/D消融入口。详细路线见[37客户端HFLSnF_KGE实施计划](./37客户端HFLSnF_KGE实施计划.md)。

## 阶段0至阶段2实现内容

### 阶段0：干净工程和回归链路

- 复制V2中经过验证的KGE数据、TransE模型、全局评估、动态拓扑和FedML客户端链路。
- 保留V2旧负采样随机序列，用于C臂回归。
- 保留37客户端划分哈希和MAT调度哈希校验。
- 新增V3专用运行时和HFLSnF入口。

### 阶段1：只读头尾方向诊断

- 读取V2或V3标准TransE检查点。
- 分别计算全局候选、全局filtered的头MRR和尾MRR。
- 输出综合指标、逐查询排名、逐关系指标和简体中文报告。
- 不创建优化器，不修改检查点。

### 阶段2：集中式强TransE校准

- GPU/CPU同设备批量头、尾filtered负采样。
- 头、尾批次持续交替。
- 自对抗困难负样本加权。
- 头关系和尾关系频率子采样权重一次预计算。
- 关系分层验证子集。
- 集中式强配方与CPU冒烟配置。

### 阶段2训练加速

- 负采样不再逐正样本执行Python和NumPy循环，而是在正样本所在设备生成整个`[批大小, 负样本数]`候选矩阵。
- 全部已知真三元组只编码一次，使用设备端排序键和二分查找严格过滤；37客户端复用同一索引。
- 集中式子采样权重在DataLoader建立前一次计算；联邦训练按客户端首次使用时计算并缓存。
- 训练数据在CUDA模式下启用页锁定内存和非阻塞传输。
- 训练期验证使用与完整方向诊断相同的批量精确头尾filtered排名，默认查询批次64、候选块8192。
- `metrics.csv`支持记录采样、传输和前后向分段耗时；正式训练关闭逐批同步剖析，避免测量逻辑拖慢训练。

阶段3的两级逐行计数加权聚合已经实施。V2兼容配置和旧吞吐基准仍保留`row_mask_presence`，V3正式HFLSnF配置已经切换为`row_count_weighted`。

### 阶段3：两级逐行计数加权聚合

- 客户端上传实体行和关系行在本地正训练事实中的出现次数。
- 边缘节点保存“参数行乘出现次数”的分子和逐行出现次数分母。
- 云端直接合并各边缘分子和分母，不对边缘平均值再次等权平均。
- 本轮没有任何正事实贡献的参数行严格回退到轮初全局值。
- 每轮记录有效行、回退行、贡献客户端数、累计出现次数及边缘和云端参数哈希。
- 正式配置的聚合权重口径为`local_positive_triple_row_occurrences`。

## 环境

推荐使用已经安装FedML和PyTorch的`py37` Conda环境：

```powershell
conda activate py37
```

如果Windows直接调用环境内`python.exe`出现动态库或`urllib3`导入问题，可使用：

```powershell
D:\Anaconda3\Scripts\conda.exe run --no-capture-output -n py37 python <参数>
```

## CPU冒烟

### 集中式强训练链路

```powershell
python -m HFLSnF_KG_v3.run_centralized_calibration `
  --cf HFLSnF_KG_v3/configs/smoke_centralized_cpu.yaml
```

### HFLSnF动态训练链路

```powershell
python -m HFLSnF_KG_v3.run_hflsnf37 `
  --cf HFLSnF_KG_v3/configs/smoke_hflsnf37_cpu.yaml
```

CPU冒烟使用合成知识图谱，只验证代码链路，不能作为MRR结论。

## 阶段1方向诊断

默认只评估8条测试事实：

```powershell
python -m HFLSnF_KG_v3.run_directional_diagnostics
```

在CUDA服务器上对V2 C臂执行完整官方测试诊断：

```bash
python -m HFLSnF_KG_v3.run_directional_diagnostics \
  --using-gpu \
  --require-cuda \
  --max-triples 0 \
  --query-batch-size 16 \
  --candidate-batch-size 4096
```

完整模式会评估20,466条测试事实，即40,932个头尾查询。

## 阶段2集中式正式校准

```bash
python -m HFLSnF_KG_v3.run_centralized_calibration \
  --cf HFLSnF_KG_v3/configs/centralized_fb15k237_strong_transe_cuda.yaml
```

正式配置：

- TransE，256维，L1距离；
- 头尾双向自对抗训练；
- 256个负样本；
- 批次1024；
- `gamma=9`；
- 学习率`5e-5`；
- 380个全量epoch；
- 4,096条关系分层验证子集；
- 完整验证和完整测试。

2026-07-28正式结果已经完成完整官方测试：

- 头MRR：`0.300992`；
- 尾MRR：`0.472941`；
- 综合MRR：`0.386967`；
- 最佳轮次：第380轮。

综合指标与只读方向诊断完全一致，因此集中式门禁已经通过。高于原目标时应如实保留，不人为压低结果。

## 加速基准

先在4090D服务器运行五轮真实吞吐基准：

```bash
python -m HFLSnF_KG_v3.run_centralized_calibration \
  --cf HFLSnF_KG_v3/configs/benchmark_accelerated_cuda.yaml
```

重点读取第2至第4轮的`epoch_seconds`，避开首次设备缓存和最后一轮评估。旧实现无验证轮平均约为`73.03秒/轮`，新结果应与它直接比较。

如果需要定位采样、传输和前后向各自耗时，再运行同步剖析配置：

```bash
python -m HFLSnF_KG_v3.run_centralized_calibration \
  --cf HFLSnF_KG_v3/configs/benchmark_accelerated_profile_cuda.yaml
```

同步剖析会故意增加CUDA同步，只用于分析耗时构成，不作为正式吞吐结论。

## 37客户端正式配置

加速训练链路、阶段3逐行计数聚合和40轮趋势筛选均已通过。原200轮配置继续保留用于历史口径回归：

```text
configs/hflsnf37_strong_transe_cuda.yaml
```

### 37客户端五轮CUDA吞吐基准

该配置保持正式实验的37客户端划分、强TransE目标、每客户端2个本地epoch、MAT动态分组和逐行存在聚合，只把通信轮数缩短为5，并将验证和测试限制为512条：

```powershell
python -m HFLSnF_KG_v3.run_hflsnf37 `
  --cf HFLSnF_KG_v3/configs/benchmark_hflsnf37_accelerated_cuda.yaml
```

前5轮均为37个客户端参与，动态边缘组数依次为9、8、9、9、7。配置写入前5轮独立调度哈希，不能用200轮正式调度哈希替代。

运行完成后读取`metrics.csv`：

- 使用第2至第4轮的`round_seconds`计算稳态轮耗时。
- 第1轮可能包含CUDA缓存和37客户端权重缓存初始化。
- 第5轮包含512条验证评估，不纳入纯训练吞吐均值。
- 五轮MRR仅用于检查训练链路，不能与200轮完整测试结果比较。

若需要拆分37客户端本地采样、传输和前后向耗时，运行两轮同步剖析：

```powershell
python -m HFLSnF_KG_v3.run_hflsnf37 `
  --cf HFLSnF_KG_v3/configs/benchmark_hflsnf37_profile_cuda.yaml
```

剖析配置读取第1轮的`client_sampling_seconds`、`client_transfer_seconds`和`client_forward_backward_seconds`；第2轮包含验证，不用于纯训练分解。

### 阶段3五轮CUDA验证

旧五轮吞吐结果继续作为`row_mask_presence`速度和短期指标基线。阶段3使用相同前5轮MAT、相同初始模型和相同强TransE参数，只把聚合切换为逐行计数加权：

```powershell
python -m HFLSnF_KG_v3.run_hflsnf37 `
  --cf HFLSnF_KG_v3/configs/benchmark_hflsnf37_row_count_cuda.yaml
```

运行后检查：

- `summary.json`中的`aggregation_mode`必须是`row_count_weighted`。
- `aggregation`必须是`hierarchical_two_level_row_count_weighted`。
- `aggregation_weight_basis`必须是`local_positive_triple_row_occurrences`。
- `metrics.csv`必须包含实体和关系的贡献客户端数及行出现次数统计。
- `dynamic_topology_schedule.jsonl`每轮必须包含边缘参数哈希和云端参数哈希。
- 第2至第4轮`round_seconds`不应相对旧五轮吞吐基准异常增加。

五轮训练只用于聚合链路、速度和早期方向检查，不能据此判断最终MRR是否达到目标。

### 种子42四十轮趋势筛选

阶段3五轮门禁通过后，使用前40轮真实动态MAT进行趋势筛选：

```powershell
python -m HFLSnF_KG_v3.run_hflsnf37 `
  --cf HFLSnF_KG_v3/configs/screen_hflsnf37_row_count_seed42_40round_cuda.yaml
```

该配置具有以下边界：

- 固定种子42和原37客户端划分。
- 使用正式MAT前40轮，参与客户端数为29至37，平均36.475。
- 动态边缘组数为5至10，平均8.225。
- 在第10、20、30、40轮使用同一4,096条关系分层验证子集。
- 最终验证仍使用该4,096条子集，测试只取512条用于链路检查。
- 只依据验证MRR和验证Hits判断是否继续，不能根据512条测试子集调参。

建议按以下规则判断：

- 第40轮是最佳验证轮，且第20至40轮MRR仍有明显上升：进入正式种子42实验。
- 第20至40轮绝对提升不足`0.01`且MRR仍低于`0.02`：先诊断客户端Adam重置、全局更新幅度和聚合后的实体归一化。
- 其他情况结合损失下降速度和验证Hits@3、Hits@10斜率判断是否增加到80轮筛选。

本次种子42筛选的验证MRR依次为：

- 第10轮：`0.005446`。
- 第20轮：`0.032647`。
- 第30轮：`0.079157`。
- 第40轮：`0.102878`。

第40轮仍为最佳，且第20至40轮绝对提高`0.070231`，已经通过继续训练门槛。

### varAlpha 0.1种子42三百轮正式训练

正式实验改用参与人数波动更小的`varAlpha=0.1`拓扑，并从相同种子和初始模型重新训练：

```powershell
python -m HFLSnF_KG_v3.run_hflsnf37 `
  --cf HFLSnF_KG_v3/configs/hflsnf37_row_count_varalpha0p1_seed42_300round_cuda.yaml
```

该配置具有以下边界：

- 固定37客户端头实体均衡划分、种子42和两级逐行计数加权聚合。
- 使用`result-U-6fixedge_epoch200_varAlpha_0p1_trainable.mat`。
- 源MAT只有200轮，因此显式使用`cycle`策略：第201至300轮依次复用源MAT第1至100轮。
- 300轮展开后的调度哈希为`8bc1f635b4126dde5454b6bc776381fb779c8c459d36a4171cf2ac3a523ccda3`。
- 300轮参与客户端数为29至37，平均36.373；动态边缘组数为2至12，平均8.157。
- 每10轮使用固定4,096条关系分层验证子集选模。
- 训练结束后恢复最佳验证检查点，并执行完整官方验证集和完整官方测试集评估。
- `summary.json`和`dynamic_participation_summary.json`会明确记录`cycle`策略、200个源拓扑轮和100个循环复用轮。

不得从旧`varAlpha=0.5`第40轮检查点续训；改变MAT后必须从头训练，才能把结果归因于`varAlpha=0.1`正式方案。

### 固定37客户端六组八十轮基线

300轮完整方向诊断得到头MRR `0.110929`、尾MRR `0.312889`和综合MRR `0.211909`。为先隔离联邦优化与数据划分问题，建立固定参与和固定分组的步骤一基线：

```powershell
python -m HFLSnF_KG_v3.run_hflsnf37 `
  --cf HFLSnF_KG_v3/configs/screen_fixed37_fixed6_reset_adam_seed42_80round_cuda.yaml
```

该基线具有以下边界：

- 37个客户端每轮全部参与，不读取MAT，不执行SnF动态选择。
- 使用固定6个边缘组，组规模依次为`7、6、6、6、6、6`。
- 继续使用种子42、原头实体均衡划分、强TransE双向目标和两级逐行计数加权。
- 保持当前“每次客户端本地调用重新创建Adam”的行为，作为后续持久化Adam的直接对照。
- 每10轮使用同一4,096条关系分层验证子集，最终验证仍取4,096条，测试只取512条用于链路检查。
- 80轮固定调度哈希为`69b9d605d4b8fa7da0d1f1900a081966f35ec1ffa84ca3d96baf1047ad25fc9c`。

动态`varAlpha=0.1`方案第40轮和第80轮验证MRR分别为`0.102596`和`0.150945`。固定基线必须与这两个同口径点比较：

- 固定基线差异在`±0.005`内：动态拓扑不是当前低MRR主因。
- 固定基线提高至少`0.01`：参与波动或动态拓扑存在可见影响。
- 无论固定基线结果如何，下一项单变量实验都是固定分组下的逐客户端持久化Adam。

步骤一实测第80轮验证MRR为`0.150739`，512条测试链路MRR为`0.159721`。相对动态方案第80轮仅低`0.000206`，确认动态分组不是当前收敛瓶颈。

### 固定分组逐客户端持久化Adam八十轮对照

步骤二保持固定拓扑基线的全部设置，只把Adam状态模式从`reset`改为`persistent_per_client`：

```powershell
python -m HFLSnF_KG_v3.run_hflsnf37 `
  --cf HFLSnF_KG_v3/configs/screen_fixed37_fixed6_persistent_adam_seed42_80round_cuda.yaml
```

持久化语义：

- 每个客户端拥有独立Adam一阶矩、二阶矩和步数。
- 每次参与仍先加载最新全局模型参数，优化器状态不会恢复旧模型参数。
- 第1轮37个客户端均为首次建立状态；从第2轮开始应复用37份独立状态。
- 每个客户端每轮约执行16个Adam步骤，因此第2轮开始前的步数应约为16，第80轮结束后应约为1,280。
- 37份Adam状态保留在训练设备上，预计额外占用约1.1GB显存，4090D能够承受。

结果审计：

- `metrics.csv`记录每轮复用状态的客户端数量，以及Adam步数的最小值、均值和最大值。
- `dynamic_topology_schedule.jsonl`同步保存状态模式、复用数量和步数范围。
- `summary.json`记录最终状态模式和已缓存优化器状态的客户端数量。
- 第1轮损失、初始模型哈希、划分哈希和80轮拓扑哈希应与重置Adam基线完全一致。

判定门槛：

- 第80轮验证MRR至少达到`0.160739`，即相对步骤一提高`0.01`，才认为持久化Adam有明确正收益。
- 若达到`0.17`以上，可直接准备固定分组300轮正式实验。
- 若没有明显提升，则停止增加轮数，转向服务器端FedAdam或允许实体跨客户端出现的数据划分。

旧C臂回归配置：

```text
configs/hflsnf37_v2_c_regression_cuda.yaml
```

## IDE运行

打开`run_from_ide.py`直接运行，默认执行安全的集中式CPU冒烟。

可通过环境变量切换：

```powershell
$env:HFLSNF_KG_V3_IDE_PROFILE = "hflsnf37_smoke_cpu"
python HFLSnF_KG_v3\run_from_ide.py
```

可选方案：

- `centralized_smoke_cpu`
- `centralized_strong_cuda`
- `hflsnf37_smoke_cpu`
- `hflsnf37_strong_cuda`
- `hflsnf37_benchmark_cuda`
- `hflsnf37_profile_cuda`
- `hflsnf37_row_count_benchmark_cuda`
- `hflsnf37_row_count_screen40_cuda`
- `hflsnf37_varalpha0p1_formal300_cuda`
- `fixed37_fixed6_reset_adam_screen80_cuda`
- `fixed37_fixed6_persistent_adam_screen80_cuda`

## 自动化测试

```powershell
python -m unittest discover -s HFLSnF_KG_v3\tests -v
```

测试覆盖：

- V2旧采样随机序列回归；
- FB15k-237的37客户端划分哈希；
- 向量化头尾负采样；
- 频率子采样；
- 自对抗损失梯度；
- 关系分层验证；
- CPU集中式双向训练；
- 检查点方向诊断和输出文件。

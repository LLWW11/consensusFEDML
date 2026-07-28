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

阶段3的两级逐行计数加权聚合尚未实施。当前HFLSnF强配方配置仍明确使用V2验证过的`row_mask_presence`，不能提前称为最终V3聚合方案。

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

加速训练链路已经接入正式配置。阶段3实施前可以运行五轮链路检查，但暂不建议直接启动200轮最终实验：

```text
configs/hflsnf37_strong_transe_cuda.yaml
```

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

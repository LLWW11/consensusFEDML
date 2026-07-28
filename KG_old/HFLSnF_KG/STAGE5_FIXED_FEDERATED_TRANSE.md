# 阶段五：四种固定参与拓扑的联邦TransE

## 实验目标

本阶段在37个头实体均衡知识客户端上实现四个对比对象：

| 方案 | 聚合结构 | 固定参与人数 | 固定组数 |
| --- | --- | ---: | ---: |
| FLnoSnF | 客户端直接到云端 | 5 | 1 |
| FLSnF | 客户端直接到云端 | 25 | 1 |
| HFLnoSnF | 客户端到边缘组再到云端 | 15 | 6 |
| HFLSnF | 客户端到边缘组再到云端 | 35 | 6 |

四个方案都设置`client_num_in_total: 37`。本阶段严格关闭动态客户端选择：每个方案只在实验开始时按`fixed_client_seed`生成一次37客户端固定排列，再截取配置人数作为固定参与集合，后续所有通信轮复用完全相同的客户端编号。

`SnF/noSnF`当前只表示实验场景及对应固定参与预算，不执行逐轮SnF评分、MAT调度或客户端替换。这样先得到可复现的固定拓扑基线，后续再单独加入动态选择，避免同时改变参与人数、参与身份和拓扑。

## 固定参与集合

四份正式配置都使用`fixed_client_seed: 42`。实现先生成同一份37客户端随机排列，再按人数截取前缀，因此固定集合满足：

```text
FLnoSnF的5人
    ⊂ HFLnoSnF的15人
    ⊂ FLSnF的25人
    ⊂ HFLSnF的35人
```

这种嵌套设计减少了不同方案之间由完全不同客户端身份造成的混杂。每次运行都会把实际抽中的顺序、排序后的客户端编号、SHA-256参与集合指纹和拓扑指纹保存到`fixed_participation.json`。

## 六组分层聚合

HFLnoSnF和HFLSnF将固定采样顺序按轮转方式分到6组：

- HFLnoSnF的15人组规模为`3、3、3、2、2、2`；
- HFLSnF的35人组规模为`6、6、6、6、6、5`。

每轮所有固定参与客户端都从同一份全局TransE参数开始本地训练。HFL先在每组内按本地正三元组数保留加权分子和分母，再由云端合并6组统计并统一归一化。客户端集合与权重相同时，这种两级稠密FedAvg与直接FedAvg数学等价；本阶段四个方案的主要实验差异来自固定参与预算和FL/HFL通信结构，而不是人为改变聚合公式。

## 分别保存的数据

每个方案使用独立`run_name`和微秒时间戳结果目录，互不覆盖。除实体关系映射、分区摘要、指标、汇总和最佳模型外，还保存：

- `fixed_participation.json`：固定参与编号、6组映射、参与集合指纹、拓扑指纹、选中训练三元组比例和各组知识规模；
- `participation_schedule.jsonl`：每轮实际参与编号、组映射、客户端权重、组聚合权重和聚合方式；
- `metrics.csv`：每轮方案名、FL/HFL结构、SnF标识、参与人数、组数、本地加权损失、轮次耗时和验证指标；
- `summary.json`：固定参与集合、组映射、最佳轮、完整验证测试指标和集中式参考差值；
- `model_best.pt`：最佳全局参数，以及固定拓扑、统一编号和客户端分区摘要。

运行过程中每个通信epoch都会立即打印方案、轮次、固定参与人数、分组数、本地加权损失、聚合权重和耗时。配置指定的选模轮会同时打印验证MRR和Hits@3；其他轮次显示“未评估”，避免为了终端输出而额外执行昂贵的filtered排名评估。

## 服务器运行

单独运行四种方案：

```bash
python -m HFLSnF_KG.run_fixed_federated_transe \
  --cf HFLSnF_KG/configs/server_fb15k237_flnosnf_fixed_cuda.yaml

python -m HFLSnF_KG.run_fixed_federated_transe \
  --cf HFLSnF_KG/configs/server_fb15k237_flsnf_fixed_cuda.yaml

python -m HFLSnF_KG.run_fixed_federated_transe \
  --cf HFLSnF_KG/configs/server_fb15k237_hflnosnf_fixed_cuda.yaml

python -m HFLSnF_KG.run_fixed_federated_transe \
  --cf HFLSnF_KG/configs/server_fb15k237_hflsnf_fixed_cuda.yaml
```

在Windows服务器上依次运行全部四种方案：

```powershell
.\HFLSnF_KG\run_fixed_fedtranse_comparisons.ps1
```

所有正式配置都使用`require_cuda: true`。CUDA不可用时会在读取FB15k-237之前快速报错。

## IDE运行

在`run_from_ide.py`中把`DEFAULT_PROFILE`设置为以下任意值：

- `fixed_fedtranse_flnosnf_cuda`
- `fixed_fedtranse_flsnf_cuda`
- `fixed_fedtranse_hflnosnf_cuda`
- `fixed_fedtranse_hflsnf_cuda`

IDE入口与终端入口调用同一训练主函数，不复制训练逻辑。

## 当前边界

本阶段不实现：

- 逐轮动态SnF客户端选择；
- MAT拓扑回放；
- 客户端掉线或边缘掉线；
- 实体关系行级掩码聚合；
- 不同组内本地通信轮数；
- 在线重分组。

这些因素将在固定四方案能够正确运行并生成完整对照数据后逐项加入。

# 阶段三：FB15k-237集中式TransE基线

## 阶段目标

阶段三用于建立后续联邦知识图谱实验的集中式准确性基线。它回答的是：

> 在不引入客户端数据异质性、拓扑掉线和两级聚合误差时，当前TransE训练与filtered评估实现能否正确工作？

本阶段不执行客户端划分、FedAvg或MAT拓扑调度。FedML在本阶段负责YAML配置解析、随机种子入口和 `torch.device` 设备选择；阶段四再把相同TransE模型接入FedML客户端训练和聚合链。

## 训练目标

TransE使用距离：

```text
f(h, r, t) = ||e_h + e_r - e_t||
```

距离越小表示三元组越可信。训练采用间隔排序损失：

```text
max(0, margin + positive_score - negative_score)
```

负样本通过替换头实体或尾实体得到。采样器会检查训练集、验证集和测试集中的全部已知真三元组，避免把已知事实错误地作为负样本。

## filtered评估口径

验证集和测试集同时执行头预测与尾预测。以尾预测为例，程序会把目标尾实体替换为全部实体候选；如果某个替换结果也是数据集中的已知真三元组，则在计算目标排名前将其过滤。

当前实现采用距离的乐观排名：

```text
rank = 1 + 严格小于目标距离的未过滤候选数
```

最终报告：

- filtered MRR；
- filtered Mean Rank；
- filtered Hits@1；
- filtered Hits@3；
- filtered Hits@10。

验证集指标只用于选择最佳模型。测试集不会参与选模或参数更新。

## 数据目录

将FB15k-237放置为：

```text
HFLSnF_KG/data/FB15k-237/train.txt
HFLSnF_KG/data/FB15k-237/valid.txt
HFLSnF_KG/data/FB15k-237/test.txt
```

三个文件都必须使用每行三个字段的格式。程序按训练、验证、测试的顺序建立统一的实体和关系编号。

## 本地轻量验证

本地无需下载数据即可运行内置微型知识图谱：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  -m HFLSnF_KG.run_transe `
  --cf HFLSnF_KG/configs/smoke_transe_synthetic_cpu.yaml
```

该配置只有少量实体和三元组，结果只证明训练、选模、检查点和filtered评估链路能够完成，不代表FB15k-237精度。

## 服务器正式运行

先安装与服务器CUDA驱动匹配的PyTorch，并放置FB15k-237数据，然后运行：

```bash
python -m HFLSnF_KG.run_transe \
  --cf HFLSnF_KG/configs/server_fb15k237_transe_cuda.yaml
```

正式配置中的 `require_cuda: true` 会在CUDA不可用时于数据读取前报错，不会静默退回CPU。

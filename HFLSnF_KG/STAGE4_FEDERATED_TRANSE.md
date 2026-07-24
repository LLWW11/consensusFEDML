# 阶段四：普通联邦TransE

## 实验定位

阶段四是阶段三集中式TransE与后续MAT云边端分层联邦之间的单层联邦对照组：

```text
37个知识客户端
        ↓
FedML Client与ClientTrainer本地训练
        ↓
按本地正三元组数直接云端FedAvg
        ↓
全局filtered验证与测试
```

本阶段不使用边缘节点、MAT拓扑、客户端掉线或行级掩码。

## 头实体均衡划分

对每个训练头实体统计其三元组数，然后按数量降序处理。每个头实体及其全部训练事实被分给当前三元组负载最低的客户端，相同频次使用固定随机种子打破顺序。

该划分满足：

- 同一头实体只属于一个客户端；
- 训练三元组无重复、无遗漏；
- 客户端全部非空；
- 客户端共享尾实体，自然形成局部知识重叠；
- 划分结果写入SHA-256指纹，便于服务器实验复核。

当前FB15k-237在种子42下的统计为：

```text
客户端数：37
训练三元组总数：272115
每客户端三元组数：7354—7355
实体集合两两Jaccard均值：约0.268
```

## 本地训练与聚合

所有客户端使用同一全局 `entity2id` 和 `relation2id`，因此实体与关系嵌入表形状一致。本地训练使用：

- Adam优化器，每次客户端调用重新创建优化器状态；
- TransE L1距离；
- margin ranking loss；
- 每个正三元组一个filtered负样本；
- 全局已知真三元组集合过滤负样本。

客户端上传完整实体和关系嵌入表，聚合权重为本地正三元组数：

```text
global = Σ(local_triple_count × local_model) / Σ(local_triple_count)
```

FedAvg完成后重新将全局实体嵌入投影到单位L2球面。未在某客户端出现的嵌入行仍参与普通完整参数平均，可能稀释其他客户端的有效更新；这是阶段四需要保留并量化的基线现象。

## 公平对照口径

服务器配置运行100个通信轮，每轮37个客户端全部参与且各训练1个本地epoch。每一轮合计遍历一次完整训练集，因此100轮约等于阶段三集中式训练的100次完整数据遍历。

客户端本地Adam状态每次重新创建，而阶段三集中式Adam状态跨epoch保留。因此两者的数据遍历预算相近，但优化轨迹并不完全相同；结果解释时必须将这一点与全参数FedAvg稀释共同说明。

每10轮在固定1000条验证三元组上选择最佳全局模型。训练结束恢复最佳模型，再评估完整验证集和测试集。测试集不参与参数更新或最佳轮选择。

## 运行方式

本地CPU冒烟：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  -m HFLSnF_KG.run_federated_transe `
  --cf HFLSnF_KG/configs/smoke_fedtranse_synthetic_cpu.yaml
```

服务器CUDA正式实验：

```bash
python -m HFLSnF_KG.run_federated_transe \
  --cf HFLSnF_KG/configs/server_fb15k237_fedtranse_cuda.yaml
```

IDE中使用 `fedtranse_smoke_cpu` 或 `fedtranse_server_cuda` 运行方案。正式配置要求CUDA，不允许静默回退CPU。

## 验收边界

阶段四的成功标准是客户端分区、FedML生命周期、直接FedAvg、最佳模型恢复、filtered评估和结果记录正确可复现。普通全参数FedAvg的MRR不要求接近阶段三集中式结果；程序会在汇总中直接记录两者的测试MRR差值，为后续MAT分层聚合和行级掩码实验提供对照。

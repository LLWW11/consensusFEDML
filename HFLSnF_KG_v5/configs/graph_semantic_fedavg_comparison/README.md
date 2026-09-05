# 图语义FedAvg三拓扑正式配置

本目录包含HFLSnF、HFLnoSnF与FLnoSnF三个拓扑在随机种子42、2024、2025下的9份正式配置。

九份配置从对应的现有FedAdam正式配置受控派生，只把服务器优化器改为直接FedAvg，并更新实验身份与隔离输出路径。FedAdam专用参数已删除，客户端仍使用Adam。

正式配置全部保留`require_cuda: true`。运行前使用下列命令复算分区并验证完整配置：

```powershell
python -m HFLSnF_KG_v5.run_graph_semantic_fedavg_comparison validate
```

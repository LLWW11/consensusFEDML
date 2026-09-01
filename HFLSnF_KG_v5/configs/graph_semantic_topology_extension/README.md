# V5图语义拓扑对照扩展配置

本目录包含6份CUDA正式配置，用于在现有V5图语义HFLSnF三种子结果之外，新增HFLnoSnF和FLnoSnF两个组别。

| 组别 | 种子42 | 种子2024 | 种子2025 |
| --- | --- | --- | --- |
| HFLnoSnF | `graph_semantic_hflnosnf_seed42_150round_cuda.yaml` | `graph_semantic_hflnosnf_seed2024_150round_cuda.yaml` | `graph_semantic_hflnosnf_seed2025_150round_cuda.yaml` |
| FLnoSnF | `graph_semantic_flnosnf_seed42_150round_cuda.yaml` | `graph_semantic_flnosnf_seed2024_150round_cuda.yaml` | `graph_semantic_flnosnf_seed2025_150round_cuda.yaml` |

六份配置复用`configs/graph_semantic/partition_calibration_contract.json`中的同种子图语义分区哈希。正式训练要求CUDA，结果单独写入`results/graph_semantic_topology_extension/`。

先执行配置校验：

```powershell
python -m HFLSnF_KG_v5.run_graph_semantic_topology_extension validate
```

校验通过后顺序执行6次正式训练：

```powershell
python -m HFLSnF_KG_v5.run_graph_semantic_topology_extension formal6
```

如批次中断，使用输出的批次清单恢复：

```powershell
python -m HFLSnF_KG_v5.run_graph_semantic_topology_extension formal6 --resume "批次清单绝对路径"
```

配置中的YAML注释统一使用英文。现有`configs/graph_semantic/`三份HFLSnF配置及其结果保持冻结，不由本批次覆盖或重跑。

# 集中式代码来源映射

## 复原原则

本工程把已经验证过的集中式算法内核逐文件恢复到新目录，不重新发明同名模型和评估器。这样可以保证新目录与历史MRR使用相同的数学定义和过滤规则，同时不再产生V3运行依赖。

| KGE_central职责 | 已验证来源 |
|---|---|
| TransE模型 | `KGE_central/model.py`，恢复自`HFLSnF_KG_v3/tasks/kge/model.py` |
| FB15k-237加载 | `KGE_central/data.py`，恢复自`HFLSnF_KG_v3/tasks/kge/data.py` |
| 集中式训练与选模 | `KGE_central/trainer.py`，恢复自`HFLSnF_KG_v3/tasks/kge/trainer.py` |
| 精确filtered评估 | `KGE_central/evaluator.py`，恢复自`HFLSnF_KG_v3/tasks/kge/evaluator.py` |
| 批量头尾排名内核 | `KGE_central/directional.py`，恢复自V3同名文件 |
| 向量化负采样 | `KGE_central/negative_sampling.py`，恢复自V3同名文件 |
| 频率子采样 | `KGE_central/subsampling.py`，恢复自V3同名文件 |
| 自对抗目标 | `KGE_central/objectives.py`，恢复自V3同名文件 |
| 独立YAML运行时 | `KGE_central/configuration.py`与`runtime.py` |
| 独立训练入口 | `KGE_central/experiment.py`与`run.py` |

`KGE_central/engine.py`是算法内核的显式边界。集中式入口不会导入`HFLSnF_KG_v3`、FedML Runner、客户端划分或拓扑调度。

# V5配置说明

V5当前正式主线位于 `graph_semantic/`，包含三种子图语义HFLSnF配置和冻结分区合同。

- `dynamic/`：冻结动态拓扑基线配置；
- `stochastic/`：历史随机拓扑配置；
- `overlap/`：V4实体重叠率配置副本，仅供审计；
- `graph_semantic/`：V5当前正式配置；
- `zOld/`：更早阶段配置。

三份图语义YAML只相对同种子HFLSnF基线修改数据划分字段、V5实验身份、分区哈希和独立结果目录；模型、拓扑、训练、FedAdam和评估字段保持一致。所有YAML注释使用英文。

正式训练必须使用CUDA，不允许将 `require_cuda` 改为 `false`。执行方法见[项目README](../README.md)。

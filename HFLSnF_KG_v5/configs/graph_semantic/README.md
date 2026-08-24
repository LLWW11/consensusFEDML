# V5图语义正式配置

本目录包含三份正式YAML、一个三种子无训练校准合同，以及不依赖V4目录的十二格冻结参考。

| 种子 | 配置 |
| ---: | --- |
| 42 | `graph_semantic_hflsnf_seed42_150round_cuda.yaml` |
| 2024 | `graph_semantic_hflsnf_seed2024_150round_cuda.yaml` |
| 2025 | `graph_semantic_hflsnf_seed2025_150round_cuda.yaml` |

`partition_calibration_contract.json` 状态为 `passed`，记录数据哈希、原始基线指标、正式分区摘要、门禁和独立复算结果。不得手工修改合同、正式分区哈希或搜索参数；算法变化后必须生成新合同并重新验证全部配置。

正式配置要求CUDA，并把新结果写入 `results/graph_semantic/`。YAML注释统一使用英文。

历史参考位于 [`frozen_v4_reference/`](frozen_v4_reference/README.md)，三份JSON按原始字节保存并继续执行固定SHA-256门禁。

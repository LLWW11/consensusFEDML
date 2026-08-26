# V5语义—图局部性双消融正式配置

本目录隔离保存消融A和消融B的六份正式CUDA配置、六分区无训练校准合同及完整V5冻结参考。

| 实验臂 | 划分策略 | 种子 |
| --- | --- | --- |
| A：仅图局部性 | `domain_head_graph_local_no_primary_balanced` | 42、2024、2025 |
| B：仅语义集中 | `semantic_domain_no_graph_local_balanced` | 42、2024、2025 |

最终无训练校准已经锁定以下六个分区：

| 实验臂 | 种子 | 分区哈希 |
| --- | ---: | --- |
| A | 42 | `4a8b0e12babb956e7b9d3fdb0a6dffd28c0e4cb18ae0a079edc515912d120be4` |
| A | 2024 | `9d4aa19eabdefc5f6dc215345aa2c305e4a6a549e4380f66e7d0b35b77b502fb` |
| A | 2025 | `1223846a543941f6a00c431c523327a374f62d019e0e0be0a0ddfd56b1c74637` |
| B | 42 | `1a11dea64d7a38d3fb3080490c39cfc500a06ce4c267382511b9ab1bd39275fb` |
| B | 2024 | `c0c16466987acc7cb2c4a6934569c24f4b51765ead52b96d62ee4fc35abfb627` |
| B | 2025 | `40cdf6915ded224d9bbdcb378a74dac43430353ec9d0d270c39642d5d0a61bc8` |

所有正式YAML都要求CUDA，训练结果写入 `results/graph_semantic_mechanism_ablation/`。YAML注释必须使用英文。

`partition_calibration_contract.json` 由无训练校准入口生成，绑定数据文件、六个分区哈希、独立复算结果和完整V5冻结参考。不得手工修改合同或YAML中的预期分区哈希；划分算法变化后必须重新生成合同并重新验证六份配置。

`frozen_full_v5_reference/` 保存现有完整V5的校准合同、正式批次、官方全量评估、分析摘要和统一指标快照。新消融报告只读取这些冻结JSON，不重新训练完整V5，也不依赖可变的结果目录摘要。

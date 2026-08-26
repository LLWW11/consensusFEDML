# KGE_pykeen来源与语义边界

## 隔离原则

`KGE_pykeen`复制自已经验证的`KGE_central`，但运行时不导入原包。原目录继续作为历史MRR为0.386967的冻结自研基线，本目录仅承载PyKEEN对照实现。

复制时没有带入`results/`、`__pycache__/`和`results.rar`。

## 模块来源

| 职责 | 来源与改造 |
|---|---|
| 数据加载与全局编号 | 从`KGE_central/data.py`复制，保持不变 |
| 严格负采样与频率权重 | 从原基线复制，仅供matched模式使用 |
| canonical filtered评估 | 从原基线复制，作为两个模式的共同主口径 |
| PyKEEN模型适配 | `model.py`封装PyKEEN 1.10.1 TransE并恢复正距离接口 |
| 共享编号桥与审计哈希 | `pykeen_bridge.py`创建TriplesFactory并记录环境、映射和划分合同 |
| matched损失 | `objectives.py`按PyKEEN 1.10.1 NSSA参数语义增加频率加权归约 |
| 原生流水线 | `native_runner.py`调用PyKEEN pipeline、sLCWA、BasicNegativeSampler、NSSALoss、EarlyStopper和RankBasedEvaluator |
| 实验调度与检查点 | `experiment.py`按comparison_mode分派并输出规范化与原始双状态 |
| 三臂报告 | `compare.py`校验共同合同并公开原生模式差异 |

## 解释边界

matched模式验证在相同采样、权重和评估口径下，PyKEEN TransE分数与自研距离及NSSA梯度是否一致。

native模式验证标准库端到端实现能否在相同主要超参数下得到可比较结果。它与matched模式存在训练事实过滤、破坏方向、频率权重和验证时点差异，因此报告中不得称为严格等价复现。

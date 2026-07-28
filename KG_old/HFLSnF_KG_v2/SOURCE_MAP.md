# FedML框架迁移来源说明

本目录没有修改原来的 `HFLSnF_dynEdge/`。以下文件根据原项目结构迁移到 `HFLSnF_KG_v2/fedml_framework/`，并针对 `model(features, adjacency)` 的GCN任务重新实现。

| 新文件 | 结构来源 | 主要调整 |
| --- | --- | --- |
| `fedml_framework/runner.py` | `HFLSnF_dynEdge/runner_test.py` | 只保留FedML simulation和sp后端，路由到分层GCN模拟器 |
| `fedml_framework/simulator.py` | `HFLSnF_dynEdge/simulator_test.py` | 新增 `HierarchicalGCN` 优化器路由 |
| `fedml_framework/client.py` | `HFLSnF_dynEdge/client_test.py` | 继承FedML FedAvg Client，数据改为局部诱导子图 |
| `fedml_framework/group.py` | `HFLSnF_dynEdge/group_test.py` | 组内训练改用FedML GCN客户端，输出可继续合并的边缘统计 |
| `fedml_framework/hierarchical_trainer.py` | `HFLSnF_dynEdge/trainer_test.py` | 保留客户端—边缘—云流程，删除MNIST概率探针和本地图片评估 |
| `fedml_framework/model_trainer.py` | FedML `ClientTrainer` 接口 | 实现GCN双输入前向、本地NLL训练和完整图评估 |

原项目的 `init_test.py` 没有复制，因为其中存在写死随机种子和与本实验无关的MLOps初始化。新入口继续使用FedML官方 `load_arguments` 解析YAML、FedML设备接口选择设备，并统一读取 `random_seed`。

原项目的MAT拓扑解析通过 `MatlabTopologyProvider` 只读复用，阶段二的Cora冒烟使用静态拓扑。阶段六已经把MAT的37个候选槽位与37个FB15k-237知识客户端一一对应，每一行MAT严格对应一个通信epoch。

阶段三的 `tasks/kge/` 是新增集中式TransE准确性基线，不从MNIST训练器复制任务逻辑。它继续复用 `run_hfl_kg.py` 中的FedML YAML初始化、统一随机种子、设备解析和独立结果目录接口。

阶段四新增 `fedml_kge/`，其中Client和ClientTrainer分别继承FedML官方基类，Runner与训练循环沿用 `HFLSnF_dynEdge` 的单进程结构，但只执行客户端到云端的直接FedAvg，不创建边缘组。阶段五在这一普通联邦基线上实现固定四方案；阶段六由 `fedml_kge/dynamic_trainer.py` 读取MAT逐轮拓扑，并完成动态客户端训练、组内聚合和云端合并。

## V2的FedE式实现来源

V2不会在运行时导入`1paperAbout/FedE-master/`。相关语义按以下对应关系重新实现：

| V2文件 | FedE开源来源 | 迁移内容 |
| --- | --- | --- |
| `core/aggregation.py` | `1paperAbout/FedE-master/fede.py`中的`Server.aggregation` | 把实体出现次数二值化，并在拥有该行的客户端之间等权平均 |
| `fedml_kge/client.py` | `fede.py`训练循环传入的`ent_update_weights` | 从本地正三元组建立实体存在掩码；V2还建立关系存在掩码 |
| `tasks/kge/negative_sampling.py` | `1paperAbout/FedE-master/dataloader.py`中的`TrainDataset` | 增加只替换尾实体的负采样模式 |
| `fedml_kge/model_trainer.py` | `1paperAbout/FedE-master/fede.py`中的`Client.client_update` | 实现`gamma - L1距离`分数、自对抗权重和正负逻辑损失 |
| `tasks/kge/model.py` | `1paperAbout/FedE-master/kge_model.py`中的`KGEModel.TransE` | 复用V2统一嵌入表并通过配置切换L1距离 |

V2做了四项有意适配：关系嵌入也保持全局表并按关系掩码聚合；只允许MAT当前轮选中客户端贡献；尾负采样使用训练、验证和测试的全局真三元组过滤；评估继续使用当前全局头尾filtered排名。这些适配用于保持V2与现有动态MAT实验的统一对照口径，不代表对FedE论文数据划分和私有关系设定的逐项复刻。

## 无需重训评估桥接来源

评估桥接不会导入或修改FedE训练器，只读取可信的Fed3 pickle和最佳检查点。

| V2文件 | 对应来源 | 作用 |
| --- | --- | --- |
| `tasks/kge/evaluation_bridge.py` | FedE `dataloader.py`中的`TestDataset`和`fede.py`中的`Client.client_eval` | 复现局部候选、局部filtered和仅尾预测的E0口径 |
| `tasks/kge/evaluation_bridge.py` | FedE检查点中的`ent_embed`与三个`rel_embed` | 根据`edge_type → edge_type_ori`映射拼回237行全局关系表 |
| `tasks/kge/evaluation_bridge.py` | V2 `tasks/kge/evaluator.py`的全局头尾filtered定义 | 实现候选分块、查询批次和乐观并列排名的E1/E2口径 |
| `run_evaluation_bridge.py` | V2独立运行与结果快照约定 | 数据审计、检查点加载、E0/E1/E2调度和结果输出 |
| `run_evaluation_bridge_from_ide.py` | V2统一IDE入口约定 | 默认执行CPU轻量冒烟，可切换CUDA完整复评 |

## 同MAT三臂消融执行层

| V2文件 | 作用 | 关键约束 |
|---|---|---|
| `tasks/kge/ablation.py` | 三份配置公平合同、结果可比性审计和中文汇总 | 同数据、同划分、同MAT、同初始模型、同预算才允许比较 |
| `run_three_arm_ablation.py` | 终端校验、顺序训练和已有结果汇总 | 正式训练要求CUDA，每个实验臂使用独立子进程和结果目录 |
| `run_three_arm_ablation_from_ide.py` | PyCharm与VSCode一键入口 | 默认只校验；服务器可切换为连续运行A、B、C |

当前Fed3与标准FB15k-237在全局编号下的310,116条唯一事实完全相同。代码仍会在每次运行时重新核对全集哈希、文件哈希、关系映射和公共留出集，任何不一致都会快速失败。

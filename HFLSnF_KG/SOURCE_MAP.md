# FedML框架迁移来源说明

本目录没有修改原来的 `HFLSnF_dynEdge/`。以下文件根据原项目结构迁移到 `HFLSnF_KG/fedml_framework/`，并针对 `model(features, adjacency)` 的GCN任务重新实现。

| 新文件 | 结构来源 | 主要调整 |
| --- | --- | --- |
| `fedml_framework/runner.py` | `HFLSnF_dynEdge/runner_test.py` | 只保留FedML simulation和sp后端，路由到分层GCN模拟器 |
| `fedml_framework/simulator.py` | `HFLSnF_dynEdge/simulator_test.py` | 新增 `HierarchicalGCN` 优化器路由 |
| `fedml_framework/client.py` | `HFLSnF_dynEdge/client_test.py` | 继承FedML FedAvg Client，数据改为局部诱导子图 |
| `fedml_framework/group.py` | `HFLSnF_dynEdge/group_test.py` | 组内训练改用FedML GCN客户端，输出可继续合并的边缘统计 |
| `fedml_framework/hierarchical_trainer.py` | `HFLSnF_dynEdge/trainer_test.py` | 保留客户端—边缘—云流程，删除MNIST概率探针和本地图片评估 |
| `fedml_framework/model_trainer.py` | FedML `ClientTrainer` 接口 | 实现GCN双输入前向、本地NLL训练和完整图评估 |

原项目的 `init_test.py` 没有复制，因为其中存在写死随机种子和与本实验无关的MLOps初始化。新入口继续使用FedML官方 `load_arguments` 解析YAML、FedML设备接口选择设备，并统一读取 `random_seed`。

原项目的MAT拓扑解析当前通过 `MatlabTopologyProvider` 只读复用，阶段二的Cora冒烟使用静态拓扑。后续接入37槽位FB15k实验时，再将MAT候选槽位与37个知识客户端一一对应。

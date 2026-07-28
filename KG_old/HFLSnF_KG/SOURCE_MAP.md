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

原项目的MAT拓扑解析通过 `MatlabTopologyProvider` 只读复用，阶段二的Cora冒烟使用静态拓扑。阶段六已经把MAT的37个候选槽位与37个FB15k-237知识客户端一一对应，每一行MAT严格对应一个通信epoch。

阶段三的 `tasks/kge/` 是新增集中式TransE准确性基线，不从MNIST训练器复制任务逻辑。它继续复用 `run_hfl_kg.py` 中的FedML YAML初始化、统一随机种子、设备解析和独立结果目录接口。

阶段四新增 `fedml_kge/`，其中Client和ClientTrainer分别继承FedML官方基类，Runner与训练循环沿用 `HFLSnF_dynEdge` 的单进程结构，但只执行客户端到云端的直接FedAvg，不创建边缘组。阶段五在这一普通联邦基线上实现固定四方案；阶段六由 `fedml_kge/dynamic_trainer.py` 读取MAT逐轮拓扑，并完成动态客户端训练、组内聚合和云端合并。

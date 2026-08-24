# TopoOptV2 单轮耗散流拓扑优化

`TopoOptV2` 是对 `Topo_opt` 的隔离式实验实现。原目录中的正式调用链以只读副本保存在 `legacy/`，新算法全部放在 MATLAB 包 `+topooptv2/` 中，不覆盖旧函数名。

## 当前实现边界

- 实现单轮 S1 聚合节点预筛选、S2 非对称耗散网络和 S3 五阶段搜索。
- 使用自包含的整数最小费用最大流求解器，不依赖 Optimization Toolbox。
- 对实际输入不少于两股却被费用路径诱导为全耗散的阶段，使用固定流量下界
  变换重求解，保证普通输出一股、其余输入耗散后再执行伪聚合检查。
- 串联或多层聚合会沿普通输出传播完整客户端谱系，同时保留每个客户端的
  唯一首次聚合点和时间层。
- 返回客户端参与、实际聚合、成本、时延、耗散和伪聚合剔除等诊断信息。
- 不实现跨轮动态权重，不保存正式结果，不绘图，也不启动 200 轮 Metro 实验。

## 目录

- `legacy/`：从 `Topo_opt` 复制的 19 个正式运行依赖，保持字节一致。
- `+topooptv2/`：单轮新方法的图构建、求解、校验和搜索代码。
- `tests/`：人工小图测试与两个旧拓扑生成回归测试。
- `examples/`：不保存结果的最小调用示例。
- `results/`：正式结果预留目录，本次实现不向其中写入数据。

## 最小验证

在 MATLAB 中进入本目录的父目录后执行：

```matlab
cd('D:/1/1myworkcode/TopoOptV2');
summary = run_smoke_tests();
```

该入口只运行静态分析、路径隔离检查、最小费用流小图、耗散约束小图、
高后续成本条件输出小图和连续两层谱系小图。`legacy/test1_varControl.m`
仅作为冻结基线保留，不会由冒烟入口调用。

## 单轮接口

```matlab
config = topooptv2.default_config();
[solution, screening, scenario] = topooptv2.run_single_round( ...
    TSML_BdwMat, clientIds, cloudId, config);
```

`TSML_BdwMat` 为 `节点数 × 节点数 × 时间层数` 的非负整数容量数组。
`solution` 返回最终方案和五阶段提交或回滚记录；其中
`group_mapping.client_ids` 是该阶段的完整客户端谱系，
`first_aggregation_client_ids` 是首次在该阶段进入聚合的客户端。
`screening` 返回初始、候选和外围节点集合，`scenario` 返回实际求解使用的
统一边表。

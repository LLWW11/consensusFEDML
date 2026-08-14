# 拓扑后处理目录

本目录集中保存 `Topo_opt` 的后处理代码、入口脚本、专项测试和后处理结果。原始两阶段最大流拓扑结果仍保存在上一级 `Topo_opt` 目录，后处理不会覆盖原始文件。

## 常用入口

- `run_build_trainable_varalpha.m`：修复零客户端轮次、收缩客户端人数方差、重建训练映射，并按配置执行客户端覆盖修复。
- `run_fill_zero_client_rounds.m`：单独修复客户端人数为零的轮次，并可选执行客户端覆盖修复。
- `plot_client_variance_controlled.m`：绘制原始或方差受控的客户端人数统计。

两个运行脚本默认使用：

```matlab
coverage_mode = 'preserve';
coverage_horizon = 150;
```

需要保证前 150 轮尽可能覆盖全部客户端时，将 `coverage_mode` 改为 `'hard'`。脚本会自动定位上一级目录中的 `result-U-6fixedge_epoch200.mat`，生成的 MAT 文件默认保存在本目录。

## 测试

在 MATLAB 中运行：

```matlab
cd('D:/1/1myworkcode/Topo_opt/postprocess');
test_control_client_variance();
test_client_coverage_postprocess();
test_trainable_client_coverage();
```

测试产生的临时 MAT 文件会自动清理，不会修改本目录中的既有结果。

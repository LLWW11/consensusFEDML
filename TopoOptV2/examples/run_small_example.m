function [solution, screening, scenario] = run_small_example()
%RUN_SMALL_EXAMPLE 构造固定小图并执行一次 TopoOptV2 单轮搜索。
%   示例不保存结果、不绘图，适合在 MATLAB 命令行检查返回结构。

config = topooptv2.default_config();
config.storage_cost = 10;
config.compute_capacity = 3;

capacityTensor = zeros(4, 4, 3);
capacityTensor(1, 3, 1) = 1;
capacityTensor(2, 3, 1) = 1;
capacityTensor(3, 4, 2) = 2;

[solution, screening, scenario] = topooptv2.run_single_round( ...
    capacityTensor, [1, 2], 4, config);
end


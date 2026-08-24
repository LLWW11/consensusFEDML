function config = default_config()
%DEFAULT_CONFIG 返回 TopoOptV2 单轮算法的默认配置。
%   配置采用归一化的容量、成本和时延单位。调用方可以覆盖任意字段，
%   再通过 topooptv2.validate_config 完成默认值合并与合法性校验。

config = struct();
config.schema_version = '1.0';

% 四类正常边的单位成本均默认为 1，耗散边成本严格更高。
config.communication_cost = 1;
config.storage_cost = 1;
config.compute_cost = 1;
config.aggregate_output_cost = 1;
config.dissipation_cost = 2;

% 容量以“一个客户端模型”为一个流量单位。
config.storage_capacity = 15;
config.compute_capacity = 37;

% 各类边时延可以独立覆盖，虚拟终端边默认不引入额外时延。
config.communication_delay = 1;
config.storage_delay = 1;
config.compute_delay = 1;
config.aggregate_output_delay = 0;
config.dissipation_delay = 0;

% 效用函数默认奖励客户端数量，并惩罚总成本与瓶颈时延。
config.utility_client_weight = 1;
config.utility_cost_weight = -0.5;
config.utility_delay_weight = -0.5;

% 所有严格改进、容量和流守恒判断共用同一个数值容差。
config.tolerance = 1e-9;
end

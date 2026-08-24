function summary = test_dissipation_network()
%TEST_DISSIPATION_NETWORK 验证统一场景和耗散网络的结构约束。
%   测试只构造固定的小型容量图，不调用正式实验入口或求解器。

config = topooptv2.default_config();
config.storage_capacity = 5;
config.compute_capacity = 4;
config.communication_cost = 1.5;
config.dissipation_cost = 3;
config = topooptv2.validate_config(config);

capacityTensor = zeros(3, 3, 3);
for layerIndex = 1:3
    capacityTensor(1, 2, layerIndex) = 2;
    capacityTensor(2, 3, layerIndex) = 2;
end
scenario = topooptv2.build_scenario(capacityTensor, 1, 3, config);

expectedVariables = {'From', 'To', 'Capacity', 'UnitCost', 'Delay', ...
    'EdgeType', 'PhysicalFrom', 'PhysicalTo', 'LayerFrom', 'LayerTo', ...
    'AggregatorId', 'Role'};
assert(isequal(scenario.edges.Properties.VariableNames, expectedVariables), ...
    '统一边表变量或顺序不正确。');
assert(scenario.source_id == 10 && scenario.sink_id == 11, ...
    '超级源或超级汇编号不符合展开规则。');
assert(scenario.node_count == 11 && scenario.physical_node_count == 3 && ...
    scenario.layer_count == 3, '场景节点元数据不正确。');
assert(height(scenario.edges) == 16, '基础场景边数量不正确。');
assert(nnz(scenario.edges.Role == "communication") == 6, ...
    '通信边数量不正确。');
assert(nnz(scenario.edges.Role == "storage") == 6, ...
    '存储边数量不正确。');
assert(nnz(scenario.edges.Role == "source_input") == 1 && ...
    nnz(scenario.edges.Role == "cloud_sink") == 3, ...
    '终端边数量不正确。');

network = topooptv2.build_dissipation_network(scenario, 2, config);
assert(network.node_count == 14 && network.dissipation_node == 14, ...
    '计算节点或耗散节点编号不正确。');
assert(isequal(network.compute_node_ids, [12; 13]), ...
    '计算节点编号顺序不正确。');
assert(height(network.compute_map) == 2, '计算映射阶段数量不正确。');
assert(network.removed_storage_edge_count == 2, ...
    '未准确移除聚合点对应的普通存储边。');
assert(height(network.edges) == 21, '耗散网络边数量不正确。');

assert_role_capacity_cost(network.edges, "compute_input", 4, 1, 2);
assert_role_capacity_cost(network.edges, "aggregate_output", 1, 1, 2);
assert_role_capacity_cost(network.edges, "dissipation", 3, 3, 2);
assert_role_capacity_cost(network.edges, "dissipation_sink", 6, 0, 1);
assert(~any(network.edges.Role == "storage" & ...
    network.edges.PhysicalFrom == 2), ...
    '聚合点仍残留普通存储边。');

% 映射表内的索引必须精确指向对应的三类边。
for mapIndex = 1:height(network.compute_map)
    mapRow = network.compute_map(mapIndex, :);
    assert(network.edges.Role(mapRow.ComputeInputEdgeIndex) == ...
        mapRow.ComputeInputRole, '计算输入边索引映射错误。');
    assert(network.edges.Role(mapRow.AggregateOutputEdgeIndex) == ...
        mapRow.AggregateOutputRole, '聚合输出边索引映射错误。');
    assert(network.edges.Role(mapRow.DissipationEdgeIndex) == ...
        mapRow.DissipationRole, '耗散边索引映射错误。');
end

% 多聚合点必须按物理节点编号排序，并为每个相邻层创建计算阶段。
multiNetwork = topooptv2.build_dissipation_network(scenario, [2, 1], config);
assert(isequal(multiNetwork.aggregator_set, [1; 2]), ...
    '聚合点集合未按确定性顺序整理。');
assert(height(multiNetwork.compute_map) == 4 && ...
    multiNetwork.removed_storage_edge_count == 4, ...
    '多聚合点的计算阶段或存储边替换数量不正确。');

% 空集合也必须返回稳定字段和一个全局耗散节点。
emptyNetwork = topooptv2.build_dissipation_network(scenario, [], config);
assert(isempty(emptyNetwork.compute_map) && ...
    isempty(emptyNetwork.compute_node_ids), ...
    '空聚合点场景不应创建计算阶段。');
assert(nnz(emptyNetwork.edges.Role == "dissipation_sink") == 1 && ...
    emptyNetwork.edges.Capacity(end) == 0, ...
    '空聚合点场景缺少零容量耗散终端边。');

summary = struct();
summary.schema_version = '1.0-test';
summary.scenario_edge_count = height(scenario.edges);
summary.network_edge_count = height(network.edges);
summary.compute_stage_count = height(network.compute_map);
summary.passed = true;
fprintf('耗散网络结构测试通过：%d 条基础边，%d 个计算阶段。\n', ...
    summary.scenario_edge_count, summary.compute_stage_count);
end


function assert_role_capacity_cost(edges, role, expectedCapacity, ...
        expectedCost, expectedCount)
%ASSERT_ROLE_CAPACITY_COST 校验指定角色边的数量、容量和单位成本。

roleMask = edges.Role == role;
assert(nnz(roleMask) == expectedCount, '%s 边数量不正确。', role);
assert(all(edges.Capacity(roleMask) == expectedCapacity), ...
    '%s 边容量不正确。', role);
assert(all(edges.UnitCost(roleMask) == expectedCost), ...
    '%s 边单位成本不正确。', role);
end

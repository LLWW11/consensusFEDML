function summary = test_topooptv2_smoke()
%TEST_TOPOOPTV2_SMOKE 使用固定人工小图验证单轮 S1 至 S3 调用链。
%   测试覆盖两客户端真实聚合、耗散流、五阶段搜索、伪聚合剔除和空聚合
%   集合。函数不使用随机拓扑，不保存文件，也不生成图形。

config = topooptv2.default_config();
% 提高跨层存储成本，使人工图中的真实聚合方案优于直接逐流存储。
config.storage_cost = 10;
config.compute_capacity = 3;
config = topooptv2.validate_config(config);

capacityTensor = build_two_client_tensor();
[solution, screening, scenario] = topooptv2.run_single_round( ...
    capacityTensor, [1, 2], 4, config);

verify_required_solution_fields(solution);
assert(strcmp(solution.status, 'optimal'), ...
    '两客户端单轮入口没有返回最优状态：%s', solution.error_message);
assert(isequal(screening.initial, 3), ...
    'S1 未把两条客户端路径的物理交汇点 3 识别为初始聚合点。');
assert(isequal(solution.effective_aggregators, 3), ...
    '五阶段搜索没有保留有效聚合点 3。');
assert(solution.successful_client_count == 2 && ...
    solution.aggregated_client_count == 2 && ...
    solution.direct_client_count == 0, ...
    '参与、聚合或直达客户端计数不正确。');
assert(solution.dissipation_flow == 1, ...
    '两股输入合并为一股输出时应产生一单位耗散流。');
assert(solution.validation.is_valid && ...
    solution.validation.invalid_compute_count == 0, ...
    '最终方案仍包含伪聚合计算节点。');
assert(solution.solver_diagnostics.conservation_violation == 0 && ...
    solution.solver_diagnostics.capacity_violation == 0, ...
    '最终流量违反流守恒或容量约束。');
assert(isequal(solution.successful_client_ids, [1, 2]) && ...
    isequal(solution.aggregated_client_ids, [1, 2]), ...
    '客户端编号没有被唯一、完整地映射。');
pathClientIds = [solution.flow_paths.client_id];
assert(numel(pathClientIds) == numel(unique(pathClientIds)) && ...
    isequal(sort(pathClientIds), solution.successful_client_ids), ...
    '每个成功客户端必须唯一对应一条源汇单位流路径。');
assert(~isempty(solution.group_mapping) && ...
    isequal(solution.group_mapping(1).client_ids, [1, 2]), ...
    '聚合组没有保存两个客户端的映射。');
assert(solution.search_stage_count == 5 && ...
    ~isempty(solution.search_trace), ...
    '五阶段搜索没有返回完整的阶段标记或搜索轨迹。');

midpointSummary = verify_independent_midpoint_tie(config);
pseudoSummary = verify_pseudo_aggregation_removal(config);
emptySolution = topooptv2.solve_aggregation_scheme(scenario, [], config);
assert(isempty(emptySolution.effective_aggregators), ...
    '空聚合点调用不应产生有效聚合节点。');

summary = struct();
summary.schema_version = '1.0-smoke';
summary.initial_aggregators = screening.initial;
summary.final_aggregators = solution.effective_aggregators;
summary.successful_client_count = solution.successful_client_count;
summary.aggregated_client_count = solution.aggregated_client_count;
summary.dissipation_flow = solution.dissipation_flow;
summary.independent_midpoint = midpointSummary.midpoint;
summary.pseudo_removed = pseudoSummary.removed_aggregators;
summary.passed = true;
fprintf('TopoOptV2 单轮冒烟通过：%d 个客户端，%d 个有效聚合点。\n', ...
    summary.successful_client_count, numel(summary.final_aggregators));
end


function verify_required_solution_fields(solution)
%VERIFY_REQUIRED_SOLUTION_FIELDS 校验单轮解具备计划约定的稳定返回字段。

requiredFields = { ...
    'successful_client_count', 'aggregated_client_count', ...
    'direct_client_count', 'effective_aggregators', 'group_mapping', ...
    'max_layer', 'flow_paths', 'communication_cost', 'storage_cost', ...
    'compute_cost', 'dissipation_cost', 'total_cost', ...
    'bottleneck_delay', 'dissipation_flow', ...
    'pseudo_aggregation_removals', 'resolve_count', 'max_flow', ...
    'status', 'error_message', 'solve_time'};
missingFields = setdiff(requiredFields, fieldnames(solution), 'stable');
assert(isempty(missingFields), '单轮解缺少约定字段：%s。', ...
    strjoin(missingFields, ', '));
end


function summary = verify_independent_midpoint_tie(config)
%VERIFY_INDEPENDENT_MIDPOINT_TIE 验证偶数路径中点按较小物理编号破平局。

midpointTensor = zeros(6, 6, 1);
midpointTensor(5, 4, 1) = 1;
midpointTensor(4, 2, 1) = 1;
midpointTensor(2, 3, 1) = 1;
midpointTensor(3, 6, 1) = 1;
scenario = topooptv2.build_scenario(midpointTensor, 5, 6, config);
screening = topooptv2.prescreen_aggregators(scenario, config);

% 可选路径为 [5, 4, 2, 3]，两个空间中点 4 和 2 应按编号选择 2。
assert(isequal(screening.initial, 2), ...
    '独立路径的偶数中点没有按较小物理节点编号确定性破平局。');
summary = struct('midpoint', screening.initial, 'passed', true);
end


function capacityTensor = build_two_client_tensor()
%BUILD_TWO_CLIENT_TENSOR 构造两条路径在节点 3 交汇的三层容量图。

capacityTensor = zeros(4, 4, 3);
capacityTensor(1, 3, 1) = 1;
capacityTensor(2, 3, 1) = 1;
capacityTensor(3, 4, 2) = 2;
end


function summary = verify_pseudo_aggregation_removal(baseConfig)
%VERIFY_PSEUDO_AGGREGATION_REMOVAL 验证单输入计算节点会被剔除并重建。

pseudoTensor = zeros(3, 3, 2);
pseudoTensor(1, 2, 1) = 1;
pseudoTensor(2, 3, 2) = 1;
scenario = topooptv2.build_scenario(pseudoTensor, 1, 3, baseConfig);
solution = topooptv2.solve_aggregation_scheme(scenario, 2, baseConfig);

assert(strcmp(solution.status, 'optimal'), ...
    '伪聚合剔除后的直接路径没有得到可行解。');
assert(isempty(solution.effective_aggregators), ...
    '只有一股输入的节点不应被保留为有效聚合点。');
assert(isequal(solution.removed_pseudo_aggregators, 2) && ...
    solution.resolve_count == 1, ...
    '伪聚合节点没有被删除并触发一次重建。');
assert(solution.successful_client_count == 1 && ...
    solution.direct_client_count == 1, ...
    '伪聚合剔除后客户端没有沿基础路径直达云端。');

summary = struct('removed_aggregators', ...
    solution.removed_pseudo_aggregators, 'passed', true);
end

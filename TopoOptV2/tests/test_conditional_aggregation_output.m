function summary = test_conditional_aggregation_output()
%TEST_CONDITIONAL_AGGREGATION_OUTPUT 验证高后续成本下仍保留单位普通输出。
%   人工图中两客户端在节点 3 汇合，但节点 3 的普通输出还需经过一条
%   高费用通信边才能到云端。无条件的最小费用流会把两股输入全部耗散；
%   聚合求解器应触发一次带下界重求解，得到“输入 2、普通输出 1、耗散
%   1”的合法聚合，而不是把节点 3 误判为伪聚合点。

config = topooptv2.default_config();
config.communication_cost = 10;
config.compute_capacity = 3;
config = topooptv2.validate_config(config);

capacityTensor = zeros(4, 4, 2);
capacityTensor(1, 3, 1) = 1;
capacityTensor(2, 3, 1) = 1;
capacityTensor(3, 4, 2) = 1;
scenario = topooptv2.build_scenario(capacityTensor, [1, 2], 4, config);
network = topooptv2.build_dissipation_network(scenario, 3, config);

% 先直接调用无条件求解器，明确复现“后续路径太贵导致全耗散”的问题。
[rawFlowValue, ~, rawEdgeFlow] = topooptv2.min_cost_max_flow( ...
    network.node_count, network.edges, network.source_id, ...
    network.sink_id, 2);
stage = network.compute_map(1, :);
rawInput = rawEdgeFlow(stage.ComputeInputEdgeIndex);
rawOutput = rawEdgeFlow(stage.AggregateOutputEdgeIndex);
rawDissipation = rawEdgeFlow(stage.DissipationEdgeIndex);
assert(rawFlowValue == 2 && rawInput == 2 && rawOutput == 0 && ...
    rawDissipation == 2, ...
    '人工图未能复现无条件最小费用流的全耗散行为。');

solution = topooptv2.solve_aggregation_scheme(scenario, 3, config);
record = solution.validation.records(1, :);
assert(strcmp(solution.status, 'optimal'), ...
    '条件下界重求解未返回最优状态：%s', solution.error_message);
assert(solution.max_flow == 2 && solution.validation.is_valid, ...
    '条件下界重求解没有保持两单位最大流或合法聚合。');
assert(record.InputFlow == 2 && record.OutputFlow == 1 && ...
    record.DissipationFlow == 1, ...
    '合法聚合阶段必须满足输入 2、普通输出 1、耗散 1。');
assert(isequal(solution.effective_aggregators, 3) && ...
    isempty(solution.removed_pseudo_aggregators), ...
    '高后续成本导致合法聚合点被错误剔除。');
assert(solution.constraint_resolve_count == 1 && ...
    solution.pseudo_rebuild_count == 0 && solution.resolve_count == 1 && ...
    solution.solve_attempts == 2, ...
    '条件重求解次数没有被准确计入诊断字段。');
assert(solution.solver_diagnostics.conservation_violation == 0 && ...
    solution.solver_diagnostics.capacity_violation == 0 && ...
    solution.solver_diagnostics.lower_bound_violation == 0, ...
    '条件解违反流守恒、容量或边流量下界。');

partialSummary = verify_partial_status_preserved(config);
summary = struct();
summary.schema_version = '1.0-test';
summary.raw_output_flow = rawOutput;
summary.constrained_output_flow = record.OutputFlow;
summary.constrained_dissipation_flow = record.DissipationFlow;
summary.constraint_resolve_count = solution.constraint_resolve_count;
summary.partial_status = partialSummary.status;
summary.passed = true;
fprintf('条件聚合输出测试通过：全耗散已修正为普通输出 1、耗散 1。\n');
end


function summary = verify_partial_status_preserved(config)
%VERIFY_PARTIAL_STATUS_PRESERVED 验证条件重求解不会把部分最大流标为最优。

capacityTensor = zeros(5, 5, 2);
capacityTensor(1, 4, 1) = 1;
capacityTensor(2, 4, 1) = 1;
capacityTensor(4, 5, 2) = 1;
scenario = topooptv2.build_scenario( ...
    capacityTensor, [1, 2, 3], 5, config);
solution = topooptv2.solve_aggregation_scheme(scenario, 4, config);

assert(strcmp(solution.status, 'partial') && solution.max_flow == 2 && ...
    solution.target_flow == 3 && solution.validation.is_valid, ...
    '带条件下界的两单位可行流必须保留 partial 2/3 状态。');
assert(solution.constraint_resolve_count == 1 && ...
    solution.solver_diagnostics.requested_flow == 3 && ...
    solution.solver_diagnostics.constrained_fixed_flow == 2, ...
    '部分最大流的目标流量、固定流量或重求解次数记录不正确。');
summary = struct('status', solution.status, 'passed', true);
end

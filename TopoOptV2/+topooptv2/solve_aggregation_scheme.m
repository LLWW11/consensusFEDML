function solution = solve_aggregation_scheme(scenario, aggregatorSet, config)
%SOLVE_AGGREGATION_SCHEME 求解给定聚合集合并闭环剔除伪聚合点。
%   SOLUTION = topooptv2.solve_aggregation_scheme(SCENARIO,
%   AGGREGATORSET, CONFIG) 构造非对称耗散网络，执行最小费用最大流，
%   并依据“输入不小于 2、普通输出等于 1、耗散不小于 1”的条件删除
%   伪聚合点。每次删除后都会重建网络并重新求解，重建次数最多等于初始
%   聚合点数量。
%
%   函数将异常封装到 SOLUTION.status 和 SOLUTION.error_message 中，便于
%   五阶段搜索完整回滚失败的试探方案。

startTime = tic;
if nargin < 3
    config = [];
end
if nargin < 2 || isempty(aggregatorSet)
    aggregatorSet = zeros(1, 0);
end

solution = empty_solution(aggregatorSet);
try
    config = resolve_config(scenario, config);
    requestedAggregators = validate_aggregator_set(aggregatorSet, scenario);
    activeAggregators = requestedAggregators;
    removedAggregators = zeros(1, 0);
    rebuildCount = 0;
    maximumRebuilds = numel(requestedAggregators);
    totalConstraintResolveCount = 0;
    totalFlowSolveCalls = 0;

    while true
        network = topooptv2.build_dissipation_network( ...
            scenario, activeAggregators, config);
        targetFlow = numel(scenario.client_ids);
        [flowValue, totalCost, edgeFlow, solverDiagnostics] = ...
            solve_network_flow(network, targetFlow, config.tolerance);
        totalConstraintResolveCount = totalConstraintResolveCount + ...
            solverDiagnostics.constraint_resolve_count;
        totalFlowSolveCalls = totalFlowSolveCalls + ...
            solverDiagnostics.solve_calls;
        validation = topooptv2.validate_aggregation( ...
            network, edgeFlow, activeAggregators, config.tolerance);

        if validation.is_valid
            break;
        end
        if rebuildCount >= maximumRebuilds
            break;
        end

        invalidAggregators = validation.invalid_aggregators;
        nextAggregators = setdiff(activeAggregators, ...
            invalidAggregators, 'stable');
        if isequal(nextAggregators, activeAggregators)
            break;
        end
        removedAggregators = unique([removedAggregators, ...
            invalidAggregators], 'stable');
        activeAggregators = nextAggregators;
        rebuildCount = rebuildCount + 1;
    end

    [nodePaths, edgePaths, remainingFlow] = ...
        topooptv2.decompose_flow_paths(network.node_count, network.edges, ...
        edgeFlow, network.source_id, network.sink_id);
    [flowPathRecords, successfulClientIds, aggregatedClientIds, ...
        groupMapping] = summarize_paths(network, edgeFlow, nodePaths, ...
        edgePaths, validation, config.tolerance);
    costs = summarize_costs(network.edges, edgeFlow);
    bottleneckDelay = summarize_bottleneck_delay(network.edges, edgePaths);
    maximumLayer = summarize_maximum_layer(network.edges, edgeFlow, ...
        config.tolerance);
    dissipationFlow = sum(edgeFlow(string(network.edges.Role) == ...
        "dissipation"));

    solution.schema_version = '1.0-solution';
    solution.requested_aggregators = requestedAggregators;
    solution.effective_aggregators = validation.valid_aggregators;
    solution.removed_pseudo_aggregators = sort(unique(removedAggregators));
    solution.pseudo_aggregation_removals = ...
        numel(solution.removed_pseudo_aggregators);
    solution.pseudo_rebuild_count = rebuildCount;
    solution.constraint_resolve_count = totalConstraintResolveCount;
    solution.resolve_count = rebuildCount + totalConstraintResolveCount;
    solution.solve_attempts = totalFlowSolveCalls;
    solution.successful_client_count = flowValue;
    solution.successful_client_ids = successfulClientIds;
    solution.aggregated_client_count = numel(aggregatedClientIds);
    solution.aggregated_client_ids = aggregatedClientIds;
    solution.direct_client_count = max(0, flowValue - numel(aggregatedClientIds));
    solution.group_mapping = groupMapping;
    solution.max_layer = maximumLayer;
    solution.maximum_layer = maximumLayer;
    solution.flow_paths = flowPathRecords;
    solution.communication_cost = costs.communication;
    solution.storage_cost = costs.storage;
    solution.compute_cost = costs.compute;
    solution.dissipation_cost = costs.dissipation;
    solution.total_cost = totalCost;
    solution.partitioned_total_cost = costs.total;
    solution.bottleneck_delay = bottleneckDelay;
    solution.dissipation_flow = dissipationFlow;
    solution.max_flow = flowValue;
    solution.target_flow = targetFlow;
    solution.status = solverDiagnostics.status;
    if ~validation.is_valid
        solution.status = 'invalid_aggregation';
        solution.error_message = '达到伪聚合重建上限后仍存在无效聚合点。';
    elseif flowValue < targetFlow && isempty(solution.error_message)
        solution.error_message = sprintf( ...
            '仅找到 %g/%g 单位可行流。', flowValue, targetFlow);
    end
    solution.utility = calculate_utility(solution, config);
    solution.edge_flow = edgeFlow;
    solution.remaining_flow = remainingFlow;
    solution.network = network;
    solution.validation = validation;
    solverDiagnostics.total_constraint_resolve_count = ...
        totalConstraintResolveCount;
    solverDiagnostics.total_solve_calls = totalFlowSolveCalls;
    solution.solver_diagnostics = solverDiagnostics;
    solution.config = config;
catch exception
    solution.status = 'error';
    solution.error_message = exception.message;
    solution.error_identifier = exception.identifier;
    solution.utility = -inf;
end
solution.solve_time = toc(startTime);
end

function config = resolve_config(scenario, overrideConfig)
%RESOLVE_CONFIG 合并场景配置和本次求解的显式覆盖字段。
if isfield(scenario, 'config') && isstruct(scenario.config)
    config = scenario.config;
else
    config = struct();
end
if ~isempty(overrideConfig)
    if ~isstruct(overrideConfig) || ~isscalar(overrideConfig)
        error('topooptv2:solve_aggregation_scheme:InvalidConfig', ...
            'config 必须是标量结构体。');
    end
    overrideNames = fieldnames(overrideConfig);
    for fieldIndex = 1:numel(overrideNames)
        fieldName = overrideNames{fieldIndex};
        config.(fieldName) = overrideConfig.(fieldName);
    end
end
config = topooptv2.validate_config(config);
end

function aggregators = validate_aggregator_set(aggregatorSet, scenario)
%VALIDATE_AGGREGATOR_SET 校验、去重并确定性排序物理聚合节点。
requiredFields = {'physical_node_count', 'cloud_id', 'client_ids'};
if ~isstruct(scenario) || ~all(isfield(scenario, requiredFields))
    error('topooptv2:solve_aggregation_scheme:InvalidScenario', ...
        'scenario 缺少聚合求解所需字段。');
end
if isempty(aggregatorSet)
    aggregators = zeros(1, 0);
    return;
end
if ~isnumeric(aggregatorSet) || ~isreal(aggregatorSet) || ...
        ~isvector(aggregatorSet) || any(~isfinite(aggregatorSet)) || ...
        any(aggregatorSet ~= floor(aggregatorSet)) || ...
        any(aggregatorSet < 1) || ...
        any(aggregatorSet > scenario.physical_node_count)
    error('topooptv2:solve_aggregation_scheme:InvalidAggregatorSet', ...
        'aggregatorSet 必须是物理节点范围内的整数向量。');
end
aggregators = sort(unique(double(aggregatorSet(:).')));
if any(aggregators == scenario.cloud_id)
    error('topooptv2:solve_aggregation_scheme:CloudIsAggregator', ...
        '云节点不能作为聚合点。');
end
end

function [flowValue, totalCost, edgeFlow, diagnostics] = ...
        solve_network_flow(network, targetFlow, tolerance)
%SOLVE_NETWORK_FLOW 求最大流并为实际聚合阶段施加条件输出下界。
if targetFlow == 0
    flowValue = 0;
    totalCost = 0;
    edgeFlow = zeros(height(network.edges), 1);
    diagnostics = empty_solver_diagnostics();
    return;
end

[flowValue, ~, edgeFlow, initialDiagnostics] = ...
    topooptv2.min_cost_max_flow(network.node_count, network.edges, ...
    network.source_id, network.sink_id, targetFlow);
totalCost = sum(edgeFlow(:) .* network.edges.UnitCost(:));
diagnostics = initialDiagnostics;
constraintResolveCount = 0;
forcedStageRows = zeros(1, 0);
conditionalConstraintsFeasible = true;
lastLowerBoundDiagnostics = struct();

% 原始最小费用流可能因正常输出之后的路径昂贵，把 K 股输入全部送往耗散
% 节点。只要实际输入 K>=2，就必须强制该阶段普通输出至少 1 且耗散至少
% 1；普通输出容量本来就是 1，流守恒随后自动给出耗散量 K-1。
maximumConstraintRounds = height(network.compute_map);
for constraintRound = 1:maximumConstraintRounds
    newStageRows = find_conditional_stage_rows( ...
        network.compute_map, edgeFlow, forcedStageRows, tolerance);
    if isempty(newStageRows)
        break;
    end
    forcedStageRows = unique([forcedStageRows, newStageRows], 'stable');
    lowerBounds = build_conditional_lower_bounds( ...
        network, forcedStageRows);
    constraintResolveCount = constraintResolveCount + 1;
    [isFeasible, candidateCost, candidateFlow, lowerDiagnostics] = ...
        topooptv2.min_cost_flow_with_lower_bounds( ...
        network.node_count, network.edges, network.source_id, ...
        network.sink_id, flowValue, lowerBounds);
    lastLowerBoundDiagnostics = lowerDiagnostics;
    if ~isFeasible
        % 保留上一次真实流解交给伪聚合检查；不可满足条件下界的节点随后
        % 会按既有闭环规则被移除并重建，不把部分平衡流作为候选解。
        conditionalConstraintsFeasible = false;
        break;
    end
    edgeFlow = candidateFlow;
    totalCost = candidateCost;
    diagnostics = lowerDiagnostics;
end

% 带下界子问题只优化“已确认可发送的固定流量”，不能覆盖原始最大流求解
% 的 optimal/partial/infeasible 语义；对外请求量始终保持调用方目标流量。
diagnostics.status = initialDiagnostics.status;
diagnostics.message = initialDiagnostics.message;
diagnostics.requested_flow = targetFlow;
diagnostics.initial_solver_status = initialDiagnostics.status;
diagnostics.constrained_fixed_flow = flowValue;
diagnostics.constraint_resolve_count = constraintResolveCount;
diagnostics.solve_calls = 1 + constraintResolveCount;
diagnostics.forced_compute_stage_rows = forcedStageRows;
diagnostics.conditional_constraints_feasible = ...
    conditionalConstraintsFeasible;
diagnostics.last_lower_bound_diagnostics = lastLowerBoundDiagnostics;
diagnostics.reported_total_cost = totalCost;
if ~isfield(diagnostics, 'lower_bound_violation')
    diagnostics.lower_bound_violation = 0;
end
end

function stageRows = find_conditional_stage_rows( ...
        computeMap, edgeFlow, forcedStageRows, tolerance)
%FIND_CONDITIONAL_STAGE_ROWS 查找输入已聚合但输出分配不合法的计算阶段。
if isempty(computeMap)
    stageRows = zeros(1, 0);
    return;
end
inputFlow = edgeFlow(computeMap.ComputeInputEdgeIndex);
outputFlow = edgeFlow(computeMap.AggregateOutputEdgeIndex);
dissipationFlow = edgeFlow(computeMap.DissipationEdgeIndex);
needsConstraint = inputFlow >= 2 - tolerance & ...
    (abs(outputFlow - 1) > tolerance | ...
    dissipationFlow < 1 - tolerance);
needsConstraint(forcedStageRows) = false;
stageRows = find(needsConstraint).';
end

function lowerBounds = build_conditional_lower_bounds(network, stageRows)
%BUILD_CONDITIONAL_LOWER_BOUNDS 为聚合输出和耗散边成对设置单位下界。
lowerBounds = zeros(height(network.edges), 1);
if isempty(stageRows)
    return;
end
outputEdges = network.compute_map.AggregateOutputEdgeIndex(stageRows);
dissipationEdges = network.compute_map.DissipationEdgeIndex(stageRows);
lowerBounds(outputEdges) = 1;
lowerBounds(dissipationEdges) = 1;
end

function diagnostics = empty_solver_diagnostics()
%EMPTY_SOLVER_DIAGNOSTICS 构造零目标流量的标准诊断结构体。
diagnostics = struct('status', 'optimal', 'augmentations', 0, ...
    'message', '', 'elapsed_seconds', 0, 'requested_flow', 0, ...
    'constrained_fixed_flow', 0, 'conservation_violation', 0, ...
    'capacity_violation', 0, 'constraint_resolve_count', 0, ...
    'solve_calls', 0, 'forced_compute_stage_rows', zeros(1, 0), ...
    'conditional_constraints_feasible', true, ...
    'last_lower_bound_diagnostics', struct(), 'lower_bound_violation', 0, ...
    'reported_total_cost', 0);
end

function [pathRecords, successfulClientIds, aggregatedClientIds, ...
        groupMapping] = summarize_paths(network, edgeFlow, nodePaths, ...
        edgePaths, validation, tolerance)
%SUMMARIZE_PATHS 汇总单位流路径、客户端归属和有效聚合组映射。
pathCount = numel(edgePaths);
pathRecords = repmat(empty_path_record(), pathCount, 1);
pathClientIds = nan(pathCount, 1);
assignedAggregators = nan(pathCount, 1);
assignedLayers = nan(pathCount, 1);
roles = string(network.edges.Role);

for pathIndex = 1:pathCount
    pathEdgeIds = edgePaths{pathIndex};
    pathRecords(pathIndex).node_ids = nodePaths{pathIndex};
    pathRecords(pathIndex).edge_ids = pathEdgeIds;
    pathRecords(pathIndex).physical_nodes = ...
        extract_path_physical_nodes(network.edges, pathEdgeIds);
    pathRecords(pathIndex).path_delay = sum(network.edges.Delay(pathEdgeIds));
    pathRecords(pathIndex).path_cost = sum( ...
        network.edges.UnitCost(pathEdgeIds));

    sourceEdges = pathEdgeIds(roles(pathEdgeIds) == "source_input");
    if ~isempty(sourceEdges)
        pathClientIds(pathIndex) = network.edges.PhysicalTo(sourceEdges(1));
    end
    computeEdges = pathEdgeIds(roles(pathEdgeIds) == "compute_input");
    if ~isempty(computeEdges)
        % 客户端唯一归属于其路径上最先遇到的计算输入边。
        firstComputeEdge = computeEdges(1);
        assignedAggregators(pathIndex) = ...
            network.edges.AggregatorId(firstComputeEdge);
        assignedLayers(pathIndex) = network.edges.LayerFrom(firstComputeEdge);
    end
    pathRecords(pathIndex).client_id = pathClientIds(pathIndex);
    pathRecords(pathIndex).first_aggregator_id = ...
        assignedAggregators(pathIndex);
    pathRecords(pathIndex).first_aggregation_layer = ...
        assignedLayers(pathIndex);
    % 旧字段作为兼容别名保留，其语义同样是该客户端首次经过的聚合点。
    pathRecords(pathIndex).aggregator_id = assignedAggregators(pathIndex);
    pathRecords(pathIndex).aggregation_layer = assignedLayers(pathIndex);
    pathRecords(pathIndex).is_dissipated = any( ...
        roles(pathEdgeIds) == "dissipation");
end

validClientMask = isfinite(pathClientIds) & ...
    ismember(pathClientIds, network.client_ids);
successfulClientIds = sort(unique(pathClientIds(validClientMask))).';
aggregatedMask = validClientMask & isfinite(assignedAggregators) & ...
    ismember(assignedAggregators, validation.valid_aggregators);
aggregatedClientIds = sort(unique(pathClientIds(aggregatedMask))).';
groupMapping = build_group_mapping(network, validation, pathClientIds, ...
    assignedAggregators, assignedLayers, edgePaths, edgeFlow, tolerance);
end

function record = empty_path_record()
%EMPTY_PATH_RECORD 返回单位流路径报告的空结构模板。
record = struct('client_id', nan, 'node_ids', zeros(1, 0), ...
    'edge_ids', zeros(1, 0), 'physical_nodes', zeros(1, 0), ...
    'first_aggregator_id', nan, 'first_aggregation_layer', nan, ...
    'aggregator_id', nan, 'aggregation_layer', nan, ...
    'is_dissipated', false, 'path_delay', 0, 'path_cost', 0);
end

function physicalNodes = extract_path_physical_nodes(edges, pathEdgeIds)
%EXTRACT_PATH_PHYSICAL_NODES 从边路径中恢复连续的物理节点序列。
if isempty(pathEdgeIds)
    physicalNodes = zeros(1, 0);
    return;
end
physicalNodes = [edges.PhysicalFrom(pathEdgeIds(1)); ...
    edges.PhysicalTo(pathEdgeIds(:))];
physicalNodes = physicalNodes(physicalNodes > 0).';
if numel(physicalNodes) >= 2
    physicalNodes = physicalNodes([true, diff(physicalNodes) ~= 0]);
end
end

function groupMapping = build_group_mapping(network, validation, ...
        pathClientIds, assignedAggregators, assignedLayers, edgePaths, ...
        edgeFlow, tolerance)
%BUILD_GROUP_MAPPING 沿聚合代表流传播谱系并构造有效分组记录。
%   每条单位流最初只代表其源客户端。有效计算阶段把所有输入单位流当前
%   代表的客户端取并集，并把并集交给唯一普通输出路径继续向下游传播。
%   因此 CLIENT_IDS 表示完整谱系，而 FIRST_AGGREGATION_CLIENT_IDS 只
%   表示首次在当前聚合点和时间层进入聚合的客户端。

records = validation.records;
validRecordRows = find(records.IsUsed & records.IsValid);
recordCount = numel(validRecordRows);
groupMapping = repmat(empty_group_record(), recordCount, 1);

% EDGEFLOW 作为显式参数保留，确保调用处清楚映射来自本次流解。
if isempty(edgeFlow) && recordCount > 0
    error('topooptv2:solve_aggregation_scheme:MissingEdgeFlow', ...
        '存在有效聚合记录时 edgeFlow 不得为空。');
end
validate_lineage_mapping_contract(network, records);

pathLineages = initialize_path_lineages(pathClientIds, network.client_ids);
processingRows = order_compute_stage_rows(records, validRecordRows);
mappingIndexByRow = zeros(height(records), 1);
mappingIndexByRow(validRecordRows) = 1:recordCount;

for processingIndex = 1:numel(processingRows)
    recordRow = processingRows(processingIndex);
    recordIndex = mappingIndexByRow(recordRow);
    aggregatorId = records.AggregatorId(recordRow);
    layer = records.Layer(recordRow);
    inputEdgeIndex = read_compute_edge_index(network.compute_map, ...
        recordRow, 'ComputeInputEdgeIndex');
    outputEdgeIndex = read_compute_edge_index(network.compute_map, ...
        recordRow, 'AggregateOutputEdgeIndex');
    inputPathIndices = find(paths_containing_edge( ...
        edgePaths, inputEdgeIndex));
    outputPathIndices = find(paths_containing_edge( ...
        edgePaths, outputEdgeIndex));

    assert_stage_path_count(numel(inputPathIndices), ...
        records.InputFlow(recordRow), tolerance, '输入', ...
        aggregatorId, layer);
    assert_stage_path_count(numel(outputPathIndices), ...
        records.OutputFlow(recordRow), tolerance, '普通输出', ...
        aggregatorId, layer);
    clientIds = merge_path_lineages(pathLineages, inputPathIndices);
    firstClientMask = assignedAggregators == aggregatorId & ...
        assignedLayers == layer & isfinite(pathClientIds);
    firstClientIds = sort(unique(pathClientIds(firstClientMask))).';

    groupMapping(recordIndex).aggregator_id = aggregatorId;
    groupMapping(recordIndex).layer = layer;
    groupMapping(recordIndex).client_ids = clientIds;
    groupMapping(recordIndex).first_aggregation_client_ids = firstClientIds;
    groupMapping(recordIndex).input_flow = records.InputFlow(recordRow);
    groupMapping(recordIndex).output_flow = records.OutputFlow(recordRow);
    groupMapping(recordIndex).dissipation_flow = ...
        records.DissipationFlow(recordRow);
    groupMapping(recordIndex).flow_is_integral = all(abs([ ...
        groupMapping(recordIndex).input_flow, ...
        groupMapping(recordIndex).output_flow, ...
        groupMapping(recordIndex).dissipation_flow] - round([ ...
        groupMapping(recordIndex).input_flow, ...
        groupMapping(recordIndex).output_flow, ...
        groupMapping(recordIndex).dissipation_flow])) <= tolerance);

    % 普通输出容量固定为 1；其单位流携带整个输入组的客户端谱系。
    for outputListIndex = 1:numel(outputPathIndices)
        outputPathIndex = outputPathIndices(outputListIndex);
        pathLineages{outputPathIndex} = clientIds;
    end
end
end

function validate_lineage_mapping_contract(network, records)
%VALIDATE_LINEAGE_MAPPING_CONTRACT 校验计算映射和聚合记录可逐行对应。

requiredVariables = {'ComputeInputEdgeIndex', ...
    'AggregateOutputEdgeIndex'};
if ~isfield(network, 'compute_map') || ~istable(network.compute_map) || ...
        height(network.compute_map) ~= height(records) || ...
        ~all(ismember(requiredVariables, ...
        network.compute_map.Properties.VariableNames))
    error('topooptv2:solve_aggregation_scheme:InvalidLineageMap', ...
        'compute_map 无法与聚合校验记录逐行对应。');
end
end

function pathLineages = initialize_path_lineages(pathClientIds, clientIds)
%INITIALIZE_PATH_LINEAGES 用每条单位流的源客户端初始化谱系集合。

pathLineages = cell(numel(pathClientIds), 1);
for pathIndex = 1:numel(pathClientIds)
    if isfinite(pathClientIds(pathIndex)) && ...
            ismember(pathClientIds(pathIndex), clientIds)
        pathLineages{pathIndex} = pathClientIds(pathIndex);
    else
        pathLineages{pathIndex} = zeros(1, 0);
    end
end
end

function processingRows = order_compute_stage_rows(records, validRecordRows)
%ORDER_COMPUTE_STAGE_ROWS 按时间层和物理节点确定性排列有效计算阶段。

if isempty(validRecordRows)
    processingRows = zeros(0, 1);
    return;
end
sortKeys = [records.Layer(validRecordRows), ...
    records.AggregatorId(validRecordRows), validRecordRows];
[~, order] = sortrows(sortKeys, [1, 2, 3]);
processingRows = validRecordRows(order);
end

function edgeIndex = read_compute_edge_index(computeMap, rowIndex, variableName)
%READ_COMPUTE_EDGE_INDEX 读取并校验计算阶段对应的单条边索引。

edgeIndex = computeMap.(variableName)(rowIndex);
if ~isscalar(edgeIndex) || ~isfinite(edgeIndex) || ...
        edgeIndex < 1 || edgeIndex ~= floor(edgeIndex)
    error('topooptv2:solve_aggregation_scheme:InvalidComputeEdgeIndex', ...
        '计算映射第 %d 行的 %s 不是合法边索引。', ...
        rowIndex, variableName);
end
end

function pathMask = paths_containing_edge(edgePaths, edgeIndex)
%PATHS_CONTAINING_EDGE 返回包含指定边的所有单位流路径掩码。

pathMask = false(numel(edgePaths), 1);
for pathIndex = 1:numel(edgePaths)
    pathMask(pathIndex) = any(edgePaths{pathIndex} == edgeIndex);
end
end

function clientIds = merge_path_lineages(pathLineages, pathIndices)
%MERGE_PATH_LINEAGES 合并若干输入单位流当前代表的原始客户端集合。

clientIds = zeros(1, 0);
for listIndex = 1:numel(pathIndices)
    clientIds = sort(unique([clientIds, ...
        pathLineages{pathIndices(listIndex)}]));
end
end

function assert_stage_path_count(actualCount, expectedFlow, tolerance, ...
        roleLabel, aggregatorId, layer)
%ASSERT_STAGE_PATH_COUNT 校验流分解的单位路径数与阶段边流量一致。

if abs(actualCount - expectedFlow) > tolerance
    error('topooptv2:solve_aggregation_scheme:LineagePathMismatch', ...
        '聚合点 %d 第 %d 层%s路径数 %d 与边流量 %g 不一致。', ...
        aggregatorId, layer, roleLabel, actualCount, expectedFlow);
end
end

function record = empty_group_record()
%EMPTY_GROUP_RECORD 返回聚合组报告的空结构模板。
record = struct('aggregator_id', nan, 'layer', nan, ...
    'client_ids', zeros(1, 0), ...
    'first_aggregation_client_ids', zeros(1, 0), ...
    'input_flow', 0, 'output_flow', 0, ...
    'dissipation_flow', 0, 'flow_is_integral', true);
end

function costs = summarize_costs(edges, edgeFlow)
%SUMMARIZE_COSTS 按通信、存储、计算和耗散四类汇总实际成本。
roles = string(edges.Role);
edgeCosts = edgeFlow(:) .* edges.UnitCost(:);
storageMask = roles == "storage";
computeMask = roles == "compute_input" | roles == "aggregate_output";
dissipationMask = roles == "dissipation" | roles == "dissipation_sink";
communicationMask = ~(storageMask | computeMask | dissipationMask);

costs = struct();
costs.communication = sum(edgeCosts(communicationMask));
costs.storage = sum(edgeCosts(storageMask));
costs.compute = sum(edgeCosts(computeMask));
costs.dissipation = sum(edgeCosts(dissipationMask));
costs.total = sum(edgeCosts);
end

function bottleneckDelay = summarize_bottleneck_delay(edges, edgePaths)
%SUMMARIZE_BOTTLENECK_DELAY 计算所有单位流路径累计时延的最大值。
if isempty(edgePaths)
    bottleneckDelay = 0;
    return;
end
pathDelays = zeros(numel(edgePaths), 1);
for pathIndex = 1:numel(edgePaths)
    pathDelays(pathIndex) = sum(edges.Delay(edgePaths{pathIndex}));
end
bottleneckDelay = max(pathDelays);
end

function maximumLayer = summarize_maximum_layer(edges, edgeFlow, tolerance)
%SUMMARIZE_MAXIMUM_LAYER 返回承载流量边所到达的最大时间层。
usedMask = edgeFlow(:) > tolerance;
if ~any(usedMask)
    maximumLayer = 0;
    return;
end
layers = [edges.LayerFrom(usedMask); edges.LayerTo(usedMask)];
layers = layers(layers > 0);
if isempty(layers)
    maximumLayer = 0;
else
    maximumLayer = max(layers);
end
end

function utility = calculate_utility(solution, config)
%CALCULATE_UTILITY 按客户端奖励及成本、时延权重计算搜索效用。
utility = config.utility_client_weight * solution.successful_client_count + ...
    config.utility_cost_weight * solution.total_cost + ...
    config.utility_delay_weight * solution.bottleneck_delay;
end

function solution = empty_solution(aggregatorSet)
%EMPTY_SOLUTION 构造异常路径也具备完整字段的空解结构体。
solution = struct();
solution.schema_version = '1.0-solution';
solution.requested_aggregators = aggregatorSet;
solution.effective_aggregators = zeros(1, 0);
solution.removed_pseudo_aggregators = zeros(1, 0);
solution.pseudo_aggregation_removals = 0;
solution.resolve_count = 0;
solution.solve_attempts = 0;
solution.pseudo_rebuild_count = 0;
solution.constraint_resolve_count = 0;
solution.successful_client_count = 0;
solution.successful_client_ids = zeros(1, 0);
solution.aggregated_client_count = 0;
solution.aggregated_client_ids = zeros(1, 0);
solution.direct_client_count = 0;
solution.group_mapping = repmat(empty_group_record(), 0, 1);
solution.max_layer = 0;
solution.maximum_layer = 0;
solution.flow_paths = repmat(empty_path_record(), 0, 1);
solution.communication_cost = 0;
solution.storage_cost = 0;
solution.compute_cost = 0;
solution.dissipation_cost = 0;
solution.total_cost = 0;
solution.partitioned_total_cost = 0;
solution.bottleneck_delay = 0;
solution.dissipation_flow = 0;
solution.max_flow = 0;
solution.target_flow = 0;
solution.status = 'not_solved';
solution.error_message = '';
solution.error_identifier = '';
solution.utility = -inf;
solution.solve_time = 0;
solution.edge_flow = zeros(0, 1);
solution.remaining_flow = zeros(0, 1);
solution.network = struct();
solution.validation = struct();
solution.solver_diagnostics = struct();
solution.config = struct();
end

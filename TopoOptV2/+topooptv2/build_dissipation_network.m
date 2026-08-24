function network = build_dissipation_network(scenario, aggregatorSet, config)
%BUILD_DISSIPATION_NETWORK 为聚合点构造非对称耗散网络。
%   对每个聚合物理节点的相邻时间层，函数移除原普通存储边，并插入
%   “计算输入—聚合输出/耗散输出”三类边。所有计算节点共享一个全局
%   耗散节点，耗散节点再连接原场景的超级汇。

validate_scenario_contract(scenario);
if nargin < 3 || isempty(config)
    if isfield(scenario, 'config')
        config = scenario.config;
    else
        config = [];
    end
else
    % 显式配置只覆盖场景配置，避免意外丢失场景已经采用的参数。
    if isfield(scenario, 'config') && isstruct(scenario.config)
        config = merge_config_override(scenario.config, config);
    end
end
config = topooptv2.validate_config(config);
aggregatorSet = validate_aggregator_set( ...
    aggregatorSet, scenario.physical_node_count, scenario.cloud_id);

baseEdges = scenario.edges;
baseEdges.EdgeType = string(baseEdges.EdgeType);
baseEdges.Role = string(baseEdges.Role);

% 被选为聚合点的普通跨层存储边全部由计算模块替代。
removeMask = string(baseEdges.Role) == "storage" & ...
    ismember(baseEdges.PhysicalFrom, aggregatorSet) & ...
    baseEdges.PhysicalFrom == baseEdges.PhysicalTo & ...
    baseEdges.LayerTo == baseEdges.LayerFrom + 1;
removedStorageEdgeCount = nnz(removeMask);
baseEdges(removeMask, :) = [];

stageCount = numel(aggregatorSet) * max(scenario.layer_count - 1, 0);
firstComputeNode = scenario.node_count + 1;
computeNodeIds = (firstComputeNode:(firstComputeNode + stageCount - 1)).';
dissipationNode = scenario.node_count + stageCount + 1;

[computeEdges, computeMap] = create_compute_subnetwork( ...
    scenario, aggregatorSet, computeNodeIds, dissipationNode, ...
    height(baseEdges), config);

% 即使没有聚合点，也显式保留全局耗散节点与零容量终端边，稳定接口。
dissipationCapacity = stageCount * (config.compute_capacity - 1);
dissipationSinkEdge = make_edge_table( ...
    dissipationNode, scenario.sink_id, dissipationCapacity, 0, 0, ...
    "dissipation", 0, 0, 0, 0, 0, "dissipation_sink");

network = scenario;
network.schema_version = '1.0-dissipation-network';
network.edges = [baseEdges; computeEdges; dissipationSinkEdge];
network.node_count = dissipationNode;
network.dissipation_node = dissipationNode;
network.compute_node_ids = computeNodeIds;
network.compute_map = computeMap;
network.aggregator_set = aggregatorSet;
network.removed_storage_edge_count = removedStorageEdgeCount;
network.config = config;
end


function validate_scenario_contract(scenario)
%VALIDATE_SCENARIO_CONTRACT 校验输入场景是否满足统一接口约定。

requiredFields = {'edges', 'source_id', 'sink_id', 'node_count', ...
    'physical_node_count', 'layer_count', 'client_ids', 'cloud_id'};
if ~isstruct(scenario) || ~isscalar(scenario) || ...
        ~all(isfield(scenario, requiredFields))
    error('topooptv2:build_dissipation_network:InvalidScenario', ...
        'scenario 缺少统一场景接口要求的字段。');
end
if ~istable(scenario.edges)
    error('topooptv2:build_dissipation_network:InvalidEdgeTable', ...
        'scenario.edges 必须是 table。');
end
requiredVariables = {'From', 'To', 'Capacity', 'UnitCost', 'Delay', ...
    'EdgeType', 'PhysicalFrom', 'PhysicalTo', 'LayerFrom', 'LayerTo', ...
    'AggregatorId', 'Role'};
if ~all(ismember(requiredVariables, scenario.edges.Properties.VariableNames))
    error('topooptv2:build_dissipation_network:MissingEdgeVariable', ...
        'scenario.edges 缺少约定的边变量。');
end
end


function config = merge_config_override(baseConfig, overrideConfig)
%MERGE_CONFIG_OVERRIDE 用显式字段覆盖场景的既有配置。

if ~isstruct(overrideConfig) || ~isscalar(overrideConfig)
    error('topooptv2:build_dissipation_network:InvalidConfigOverride', ...
        '显式 config 必须是标量结构体。');
end
config = baseConfig;
overrideNames = fieldnames(overrideConfig);
for fieldIndex = 1:numel(overrideNames)
    fieldName = overrideNames{fieldIndex};
    config.(fieldName) = overrideConfig.(fieldName);
end
end


function aggregatorSet = validate_aggregator_set(aggregatorSet, nodeCount, cloudId)
%VALIDATE_AGGREGATOR_SET 校验并排序聚合物理节点集合。

if isempty(aggregatorSet)
    aggregatorSet = zeros(0, 1);
    return;
end
if ~isnumeric(aggregatorSet) || ~isreal(aggregatorSet) || ...
        ~isvector(aggregatorSet) || any(~isfinite(aggregatorSet)) || ...
        any(aggregatorSet ~= floor(aggregatorSet)) || ...
        any(aggregatorSet < 1) || any(aggregatorSet > nodeCount)
    error('topooptv2:build_dissipation_network:InvalidAggregatorSet', ...
        'aggregatorSet 必须是物理节点范围内的整数向量。');
end
aggregatorSet = sort(unique(double(aggregatorSet(:))));
if any(aggregatorSet == cloudId)
    error('topooptv2:build_dissipation_network:CloudIsAggregator', ...
        '云节点不能作为聚合节点。');
end
end


function [computeEdges, computeMap] = create_compute_subnetwork( ...
        scenario, aggregatorSet, computeNodeIds, dissipationNode, ...
        baseEdgeCount, config)
%CREATE_COMPUTE_SUBNETWORK 创建计算、聚合输出和耗散三类边及映射表。

stageCount = numel(computeNodeIds);
if stageCount == 0
    computeEdges = empty_edge_table();
    computeMap = empty_compute_map();
    return;
end

edgeCount = stageCount * 3;
fromNodes = zeros(edgeCount, 1);
toNodes = zeros(edgeCount, 1);
capacities = zeros(edgeCount, 1);
unitCosts = zeros(edgeCount, 1);
delays = zeros(edgeCount, 1);
edgeTypes = strings(edgeCount, 1);
physicalFrom = zeros(edgeCount, 1);
physicalTo = zeros(edgeCount, 1);
layerFrom = zeros(edgeCount, 1);
layerTo = zeros(edgeCount, 1);
aggregatorIds = zeros(edgeCount, 1);
roles = strings(edgeCount, 1);

mapAggregator = zeros(stageCount, 1);
mapLayerFrom = zeros(stageCount, 1);
mapLayerTo = zeros(stageCount, 1);
mapInputNode = zeros(stageCount, 1);
mapOutputNode = zeros(stageCount, 1);
mapInputEdge = zeros(stageCount, 1);
mapOutputEdge = zeros(stageCount, 1);
mapDissipationEdge = zeros(stageCount, 1);

stageIndex = 0;
for aggregatorIndex = 1:numel(aggregatorSet)
    aggregatorId = aggregatorSet(aggregatorIndex);
    for currentLayer = 1:max(scenario.layer_count - 1, 0)
        stageIndex = stageIndex + 1;
        computeNode = computeNodeIds(stageIndex);
        inputNode = expanded_node_id( ...
            aggregatorId, currentLayer, scenario.physical_node_count);
        outputNode = expanded_node_id( ...
            aggregatorId, currentLayer + 1, scenario.physical_node_count);
        firstLocalEdge = (stageIndex - 1) * 3 + 1;

        [fromNodes, toNodes, capacities, unitCosts, delays, edgeTypes, ...
            physicalFrom, physicalTo, layerFrom, layerTo, aggregatorIds, roles] = ...
            set_compute_stage_edges(firstLocalEdge, inputNode, outputNode, ...
            computeNode, dissipationNode, aggregatorId, currentLayer, config, ...
            fromNodes, toNodes, capacities, unitCosts, delays, edgeTypes, ...
            physicalFrom, physicalTo, layerFrom, layerTo, aggregatorIds, roles);

        mapAggregator(stageIndex) = aggregatorId;
        mapLayerFrom(stageIndex) = currentLayer;
        mapLayerTo(stageIndex) = currentLayer + 1;
        mapInputNode(stageIndex) = inputNode;
        mapOutputNode(stageIndex) = outputNode;
        mapInputEdge(stageIndex) = baseEdgeCount + firstLocalEdge;
        mapOutputEdge(stageIndex) = baseEdgeCount + firstLocalEdge + 1;
        mapDissipationEdge(stageIndex) = baseEdgeCount + firstLocalEdge + 2;
    end
end

computeEdges = table(fromNodes, toNodes, capacities, unitCosts, delays, ...
    edgeTypes, physicalFrom, physicalTo, layerFrom, layerTo, ...
    aggregatorIds, roles, 'VariableNames', edge_variable_names());
computeMap = table(mapAggregator, mapLayerFrom, mapLayerTo, computeNodeIds, ...
    mapInputNode, mapOutputNode, mapInputEdge, mapOutputEdge, ...
    mapDissipationEdge, repmat("compute_input", stageCount, 1), ...
    repmat("aggregate_output", stageCount, 1), ...
    repmat("dissipation", stageCount, 1), ...
    'VariableNames', compute_map_variable_names());
end


function [fromNodes, toNodes, capacities, unitCosts, delays, edgeTypes, ...
        physicalFrom, physicalTo, layerFrom, layerTo, aggregatorIds, roles] = ...
        set_compute_stage_edges(firstEdge, inputNode, outputNode, computeNode, ...
        dissipationNode, aggregatorId, currentLayer, config, fromNodes, ...
        toNodes, capacities, unitCosts, delays, edgeTypes, physicalFrom, ...
        physicalTo, layerFrom, layerTo, aggregatorIds, roles)
%SET_COMPUTE_STAGE_EDGES 填写一个聚合阶段对应的三条非对称边。

edgeRange = firstEdge:(firstEdge + 2);
fromNodes(edgeRange) = [inputNode; computeNode; computeNode];
toNodes(edgeRange) = [computeNode; outputNode; dissipationNode];
capacities(edgeRange) = [config.compute_capacity; 1; ...
    config.compute_capacity - 1];
unitCosts(edgeRange) = [config.compute_cost; ...
    config.aggregate_output_cost; config.dissipation_cost];
delays(edgeRange) = [config.compute_delay; ...
    config.aggregate_output_delay; config.dissipation_delay];
edgeTypes(edgeRange) = ["compute"; "aggregate"; "dissipation"];
physicalFrom(edgeRange) = aggregatorId;
physicalTo(edgeRange) = [aggregatorId; aggregatorId; 0];
layerFrom(edgeRange) = currentLayer;
layerTo(edgeRange) = currentLayer + 1;
aggregatorIds(edgeRange) = aggregatorId;
roles(edgeRange) = ["compute_input"; "aggregate_output"; "dissipation"];
end


function nodeId = expanded_node_id(physicalId, layerId, physicalNodeCount)
%EXPANDED_NODE_ID 按统一编号规则计算时间展开节点编号。

nodeId = (layerId - 1) * physicalNodeCount + physicalId;
end


function edges = make_edge_table(fromNode, toNode, capacity, unitCost, ...
        delay, edgeType, physicalFrom, physicalTo, layerFrom, layerTo, ...
        aggregatorId, role)
%MAKE_EDGE_TABLE 按统一接口创建单行边表。

edges = table(fromNode, toNode, capacity, unitCost, delay, string(edgeType), ...
    physicalFrom, physicalTo, layerFrom, layerTo, aggregatorId, string(role), ...
    'VariableNames', edge_variable_names());
end


function edges = empty_edge_table()
%EMPTY_EDGE_TABLE 创建与统一场景兼容的空边表。

edges = table(zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), strings(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), strings(0, 1), ...
    'VariableNames', edge_variable_names());
end


function computeMap = empty_compute_map()
%EMPTY_COMPUTE_MAP 创建具有稳定变量类型的空计算映射表。

computeMap = table(zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), strings(0, 1), strings(0, 1), ...
    strings(0, 1), 'VariableNames', compute_map_variable_names());
end


function names = edge_variable_names()
%EDGE_VARIABLE_NAMES 返回统一边表的固定变量顺序。

names = {'From', 'To', 'Capacity', 'UnitCost', 'Delay', 'EdgeType', ...
    'PhysicalFrom', 'PhysicalTo', 'LayerFrom', 'LayerTo', ...
    'AggregatorId', 'Role'};
end


function names = compute_map_variable_names()
%COMPUTE_MAP_VARIABLE_NAMES 返回计算映射表的固定变量顺序。

names = {'AggregatorId', 'LayerFrom', 'LayerTo', 'ComputeNode', ...
    'InputNode', 'OutputNode', 'ComputeInputEdgeIndex', ...
    'AggregateOutputEdgeIndex', 'DissipationEdgeIndex', ...
    'ComputeInputRole', 'AggregateOutputRole', 'DissipationRole'};
end

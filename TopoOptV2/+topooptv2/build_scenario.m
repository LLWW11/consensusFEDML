function scenario = build_scenario(TSML_BdwMat, clientIds, cloudId, config)
%BUILD_SCENARIO 将三维时空容量图转换为统一的有向边场景。
%   SCENARIO = BUILD_SCENARIO(TSML_BDWMAT, CLIENTIDS, CLOUDID, CONFIG)
%   按 nodeId=(layer-1)*N+physicalId 展开物理节点。正容量的三维图元素
%   形成同层通信边，相邻时间层的同一物理节点之间形成存储边。此外，
%   超级源连接第一层客户端，所有时间层的云节点均连接超级汇。

if nargin < 4
    config = [];
end
config = topooptv2.validate_config(config);
[capacityTensor, clientIds, cloudId, physicalNodeCount, layerCount] = ...
    validate_scenario_inputs(TSML_BdwMat, clientIds, cloudId, config);

communicationEdgeCount = nnz(capacityTensor > 0);
storageEdgeCount = physicalNodeCount * max(layerCount - 1, 0);
sourceEdgeCount = numel(clientIds);
cloudEdgeCount = layerCount;
edgeCount = communicationEdgeCount + storageEdgeCount + ...
    sourceEdgeCount + cloudEdgeCount;

% 一次性预分配所有列，避免逐行扩展 table。
edgeData = allocate_edge_data(edgeCount);
edgeIndex = 0;

for layerIndex = 1:layerCount
    [physicalFrom, physicalTo] = find(capacityTensor(:, :, layerIndex) > 0);
    for localEdgeIndex = 1:numel(physicalFrom)
        edgeIndex = edgeIndex + 1;
        fromPhysical = physicalFrom(localEdgeIndex);
        toPhysical = physicalTo(localEdgeIndex);
        edgeData = set_edge(edgeData, edgeIndex, ...
            expanded_node_id(fromPhysical, layerIndex, physicalNodeCount), ...
            expanded_node_id(toPhysical, layerIndex, physicalNodeCount), ...
            capacityTensor(fromPhysical, toPhysical, layerIndex), ...
            config.communication_cost, config.communication_delay, ...
            "communication", fromPhysical, toPhysical, ...
            layerIndex, layerIndex, 0, "communication");
    end
end

for layerIndex = 1:max(layerCount - 1, 0)
    for physicalId = 1:physicalNodeCount
        edgeIndex = edgeIndex + 1;
        edgeData = set_edge(edgeData, edgeIndex, ...
            expanded_node_id(physicalId, layerIndex, physicalNodeCount), ...
            expanded_node_id(physicalId, layerIndex + 1, physicalNodeCount), ...
            config.storage_capacity, config.storage_cost, ...
            config.storage_delay, "storage", physicalId, physicalId, ...
            layerIndex, layerIndex + 1, 0, "storage");
    end
end

sourceId = physicalNodeCount * layerCount + 1;
sinkId = sourceId + 1;
for clientIndex = 1:numel(clientIds)
    edgeIndex = edgeIndex + 1;
    clientId = clientIds(clientIndex);
    edgeData = set_edge(edgeData, edgeIndex, sourceId, ...
        expanded_node_id(clientId, 1, physicalNodeCount), ...
        1, 0, 0, "terminal", 0, clientId, 0, 1, 0, "source_input");
end

% 超级汇边需要容纳全部客户端；空客户端场景仍保留结构边。
cloudSinkCapacity = max(1, numel(clientIds));
for layerIndex = 1:layerCount
    edgeIndex = edgeIndex + 1;
    edgeData = set_edge(edgeData, edgeIndex, ...
        expanded_node_id(cloudId, layerIndex, physicalNodeCount), sinkId, ...
        cloudSinkCapacity, 0, 0, "terminal", cloudId, 0, ...
        layerIndex, 0, 0, "cloud_sink");
end

if edgeIndex ~= edgeCount
    error('topooptv2:build_scenario:InternalEdgeCountMismatch', ...
        '场景边计数与预分配数量不一致。');
end

scenario = struct();
scenario.schema_version = '1.0-scenario';
scenario.edges = edge_data_to_table(edgeData);
scenario.source_id = sourceId;
scenario.sink_id = sinkId;
scenario.node_count = sinkId;
scenario.physical_node_count = physicalNodeCount;
scenario.layer_count = layerCount;
scenario.client_ids = clientIds;
scenario.cloud_id = cloudId;
scenario.config = config;
end


function [capacityTensor, clientIds, cloudId, nodeCount, layerCount] = ...
        validate_scenario_inputs(TSML_BdwMat, clientIds, cloudId, config)
%VALIDATE_SCENARIO_INPUTS 校验容量图、客户端编号和云节点编号。

if ~isnumeric(TSML_BdwMat) || ~isreal(TSML_BdwMat) || isempty(TSML_BdwMat)
    error('topooptv2:build_scenario:InvalidCapacityTensor', ...
        'TSML_BdwMat 必须是非空实数数组。');
end
if ndims(TSML_BdwMat) > 3 || size(TSML_BdwMat, 1) ~= size(TSML_BdwMat, 2)
    error('topooptv2:build_scenario:InvalidCapacityShape', ...
        'TSML_BdwMat 的前两维必须是同阶方阵，且最多包含三维。');
end
capacityTensor = double(TSML_BdwMat);
if any(~isfinite(capacityTensor(:))) || any(capacityTensor(:) < 0)
    error('topooptv2:build_scenario:InvalidCapacityValue', ...
        'TSML_BdwMat 只能包含有限的非负容量。');
end
if any(abs(capacityTensor(:) - round(capacityTensor(:))) > config.tolerance)
    error('topooptv2:build_scenario:NonIntegerCapacity', ...
        'TSML_BdwMat 必须使用整数容量。');
end
capacityTensor = round(capacityTensor);
nodeCount = size(capacityTensor, 1);
layerCount = size(capacityTensor, 3);

if ~isnumeric(clientIds) || ~isreal(clientIds) || ~isvector(clientIds) || ...
        any(~isfinite(clientIds)) || any(clientIds ~= floor(clientIds)) || ...
        any(clientIds < 1) || any(clientIds > nodeCount)
    error('topooptv2:build_scenario:InvalidClientIds', ...
        'clientIds 必须是物理节点范围内的整数向量。');
end
clientIds = double(clientIds(:).');
if numel(unique(clientIds)) ~= numel(clientIds)
    error('topooptv2:build_scenario:DuplicateClientIds', ...
        'clientIds 不得包含重复节点。');
end

if ~isnumeric(cloudId) || ~isreal(cloudId) || ~isscalar(cloudId) || ...
        ~isfinite(cloudId) || cloudId ~= floor(cloudId) || ...
        cloudId < 1 || cloudId > nodeCount
    error('topooptv2:build_scenario:InvalidCloudId', ...
        'cloudId 必须是物理节点范围内的整数标量。');
end
cloudId = double(cloudId);
if any(clientIds == cloudId)
    error('topooptv2:build_scenario:CloudIsClient', ...
        '云节点不能同时作为客户端。');
end
end


function nodeId = expanded_node_id(physicalId, layerId, physicalNodeCount)
%EXPANDED_NODE_ID 按固定公式计算物理节点的时间展开编号。

nodeId = (layerId - 1) * physicalNodeCount + physicalId;
end


function edgeData = allocate_edge_data(edgeCount)
%ALLOCATE_EDGE_DATA 为统一边表的所有列预分配空间。

numericFields = {'From', 'To', 'Capacity', 'UnitCost', 'Delay', ...
    'PhysicalFrom', 'PhysicalTo', 'LayerFrom', 'LayerTo', 'AggregatorId'};
edgeData = struct();
for fieldIndex = 1:numel(numericFields)
    edgeData.(numericFields{fieldIndex}) = zeros(edgeCount, 1);
end
edgeData.EdgeType = strings(edgeCount, 1);
edgeData.Role = strings(edgeCount, 1);
end


function edgeData = set_edge(edgeData, edgeIndex, fromNode, toNode, ...
        capacity, unitCost, delay, edgeType, physicalFrom, physicalTo, ...
        layerFrom, layerTo, aggregatorId, role)
%SET_EDGE 将一条边写入预分配的统一边数据结构。

edgeData.From(edgeIndex) = fromNode;
edgeData.To(edgeIndex) = toNode;
edgeData.Capacity(edgeIndex) = capacity;
edgeData.UnitCost(edgeIndex) = unitCost;
edgeData.Delay(edgeIndex) = delay;
edgeData.EdgeType(edgeIndex) = edgeType;
edgeData.PhysicalFrom(edgeIndex) = physicalFrom;
edgeData.PhysicalTo(edgeIndex) = physicalTo;
edgeData.LayerFrom(edgeIndex) = layerFrom;
edgeData.LayerTo(edgeIndex) = layerTo;
edgeData.AggregatorId(edgeIndex) = aggregatorId;
edgeData.Role(edgeIndex) = role;
end


function edges = edge_data_to_table(edgeData)
%EDGE_DATA_TO_TABLE 按约定列顺序生成统一边表。

edges = table(edgeData.From, edgeData.To, edgeData.Capacity, ...
    edgeData.UnitCost, edgeData.Delay, edgeData.EdgeType, ...
    edgeData.PhysicalFrom, edgeData.PhysicalTo, edgeData.LayerFrom, ...
    edgeData.LayerTo, edgeData.AggregatorId, edgeData.Role, ...
    'VariableNames', {'From', 'To', 'Capacity', 'UnitCost', 'Delay', ...
    'EdgeType', 'PhysicalFrom', 'PhysicalTo', 'LayerFrom', 'LayerTo', ...
    'AggregatorId', 'Role'});
end

function validation = validate_aggregation(network, edgeFlow, selectedAggregators, tolerance)
%VALIDATE_AGGREGATION 检查耗散网络中的真实聚合与伪聚合节点。
%   VALIDATION = topooptv2.validate_aggregation(NETWORK, EDGEFLOW,
%   SELECTEDAGGREGATORS, TOLERANCE) 仅对实际承载流量的计算节点检查三项
%   条件：输入流不小于 2、普通输出流等于 1、耗散流不小于 1。未承载流量
%   的计算层不会单独报错，但被选中且所有层均未使用的物理聚合点会被判为
%   伪聚合点。

if nargin < 4 || isempty(tolerance)
    tolerance = infer_tolerance(network);
end
if nargin < 3 || isempty(selectedAggregators)
    selectedAggregators = infer_selected_aggregators(network);
end

validate_inputs(network, edgeFlow, selectedAggregators, tolerance);
selectedAggregators = sort(unique(selectedAggregators(:).'));
computeMap = network.compute_map;

recordCount = height(computeMap);
aggregatorIds = zeros(recordCount, 1);
layers = zeros(recordCount, 1);
inputFlows = zeros(recordCount, 1);
outputFlows = zeros(recordCount, 1);
dissipationFlows = zeros(recordCount, 1);
isUsed = false(recordCount, 1);
isValid = true(recordCount, 1);

for rowIndex = 1:recordCount
    aggregatorIds(rowIndex) = computeMap.AggregatorId(rowIndex);
    layers(rowIndex) = infer_compute_layer(computeMap, rowIndex);
    inputFlows(rowIndex) = read_role_flow(network, computeMap, edgeFlow, ...
        rowIndex, 'ComputeInputEdgeIndex', "compute_input");
    outputFlows(rowIndex) = read_role_flow(network, computeMap, edgeFlow, ...
        rowIndex, 'AggregateOutputEdgeIndex', "aggregate_output");
    dissipationFlows(rowIndex) = read_role_flow(network, computeMap, edgeFlow, ...
        rowIndex, 'DissipationEdgeIndex', "dissipation");

    isUsed(rowIndex) = inputFlows(rowIndex) > tolerance || ...
        outputFlows(rowIndex) > tolerance || ...
        dissipationFlows(rowIndex) > tolerance;
    if isUsed(rowIndex)
        isValid(rowIndex) = inputFlows(rowIndex) >= 2 - tolerance && ...
            abs(outputFlows(rowIndex) - 1) <= tolerance && ...
            dissipationFlows(rowIndex) >= 1 - tolerance;
    end
end

records = table(aggregatorIds, layers, inputFlows, outputFlows, ...
    dissipationFlows, isUsed, isValid, 'VariableNames', ...
    {'AggregatorId', 'Layer', 'InputFlow', 'OutputFlow', ...
    'DissipationFlow', 'IsUsed', 'IsValid'});

validAggregators = zeros(1, 0);
invalidAggregators = zeros(1, 0);
unusedAggregators = zeros(1, 0);
for aggregatorIndex = 1:numel(selectedAggregators)
    aggregatorId = selectedAggregators(aggregatorIndex);
    aggregatorRows = aggregatorIds == aggregatorId;
    usedRows = aggregatorRows & isUsed;
    if ~any(usedRows)
        unusedAggregators(end + 1) = aggregatorId; %#ok<AGROW>
        invalidAggregators(end + 1) = aggregatorId; %#ok<AGROW>
    elseif all(isValid(usedRows))
        validAggregators(end + 1) = aggregatorId; %#ok<AGROW>
    else
        invalidAggregators(end + 1) = aggregatorId; %#ok<AGROW>
    end
end

validation = struct();
validation.is_valid = isempty(invalidAggregators);
validation.valid_aggregators = sort(unique(validAggregators));
validation.invalid_aggregators = sort(unique(invalidAggregators));
validation.pseudo_aggregators = validation.invalid_aggregators;
validation.unused_aggregators = sort(unique(unusedAggregators));
validation.active_compute_count = nnz(isUsed);
validation.invalid_compute_count = nnz(isUsed & ~isValid);
validation.records = records;
validation.tolerance = tolerance;
end

function tolerance = infer_tolerance(network)
%INFER_TOLERANCE 从网络配置读取容差，并在缺失时使用默认值。
tolerance = 1e-9;
if isfield(network, 'config') && isstruct(network.config) && ...
        isfield(network.config, 'tolerance') && ...
        ~isempty(network.config.tolerance)
    tolerance = network.config.tolerance;
end
end

function aggregators = infer_selected_aggregators(network)
%INFER_SELECTED_AGGREGATORS 从计算节点映射推断所选物理聚合点。
if ~isfield(network, 'compute_map') || isempty(network.compute_map)
    aggregators = zeros(1, 0);
else
    aggregators = unique(network.compute_map.AggregatorId(:)).';
end
end

function validate_inputs(network, edgeFlow, selectedAggregators, tolerance)
%VALIDATE_INPUTS 校验聚合检查的网络、流量和容差参数。
requiredFields = {'edges', 'compute_map'};
for fieldIndex = 1:numel(requiredFields)
    if ~isfield(network, requiredFields{fieldIndex})
        error('topooptv2:validate_aggregation:MissingNetworkField', ...
            '网络缺少字段 %s。', requiredFields{fieldIndex});
    end
end
if ~istable(network.edges) || ~istable(network.compute_map)
    error('topooptv2:validate_aggregation:InvalidTable', ...
        'network.edges 和 network.compute_map 必须是 table。');
end
if ~isnumeric(edgeFlow) || ~isvector(edgeFlow) || ...
        numel(edgeFlow) ~= height(network.edges) || ...
        any(~isfinite(edgeFlow)) || any(edgeFlow < -tolerance)
    error('topooptv2:validate_aggregation:InvalidEdgeFlow', ...
        'edgeFlow 必须是与边表等长的有限非负向量。');
end
if ~isnumeric(selectedAggregators) || any(~isfinite(selectedAggregators))
    error('topooptv2:validate_aggregation:InvalidAggregators', ...
        'selectedAggregators 必须是有限数值向量。');
end
if ~isscalar(tolerance) || ~isfinite(tolerance) || tolerance < 0
    error('topooptv2:validate_aggregation:InvalidTolerance', ...
        'tolerance 必须是有限非负标量。');
end
requiredMapVariables = {'AggregatorId'};
if ~all(ismember(requiredMapVariables, ...
        network.compute_map.Properties.VariableNames))
    error('topooptv2:validate_aggregation:InvalidComputeMap', ...
        'compute_map 至少需要 AggregatorId 变量。');
end
requiredEdgeVariables = {'Role', 'AggregatorId', 'LayerFrom', 'LayerTo'};
if ~all(ismember(requiredEdgeVariables, network.edges.Properties.VariableNames))
    error('topooptv2:validate_aggregation:InvalidEdgeTable', ...
        '边表缺少聚合角色定位所需变量。');
end
end

function layer = infer_compute_layer(computeMap, rowIndex)
%INFER_COMPUTE_LAYER 从计算映射行推断用于报告的时间层。
if ismember('LayerFrom', computeMap.Properties.VariableNames)
    layer = computeMap.LayerFrom(rowIndex);
elseif ismember('Layer', computeMap.Properties.VariableNames)
    layer = computeMap.Layer(rowIndex);
elseif ismember('LayerTo', computeMap.Properties.VariableNames)
    layer = computeMap.LayerTo(rowIndex);
else
    layer = 0;
end
end

function roleFlow = read_role_flow(network, computeMap, edgeFlow, ...
        rowIndex, indexVariable, roleName)
%READ_ROLE_FLOW 优先按映射边索引读取流量，缺失时按角色和层定位。
if ismember(indexVariable, computeMap.Properties.VariableNames)
    edgeIndex = computeMap.(indexVariable)(rowIndex);
    edgeIndex = edgeIndex(isfinite(edgeIndex) & edgeIndex >= 1 & ...
        edgeIndex <= numel(edgeFlow) & edgeIndex == floor(edgeIndex));
    if ~isempty(edgeIndex)
        roleFlow = sum(edgeFlow(edgeIndex));
        return;
    end
end

aggregatorId = computeMap.AggregatorId(rowIndex);
layerFrom = read_optional_map_value(computeMap, 'LayerFrom', rowIndex, nan);
layerTo = read_optional_map_value(computeMap, 'LayerTo', rowIndex, nan);
edgeMask = string(network.edges.Role) == roleName & ...
    network.edges.AggregatorId == aggregatorId;
if isfinite(layerFrom)
    edgeMask = edgeMask & (network.edges.LayerFrom == layerFrom | ...
        network.edges.LayerTo == layerFrom);
end
if isfinite(layerTo)
    edgeMask = edgeMask & (network.edges.LayerFrom == layerTo | ...
        network.edges.LayerTo == layerTo);
end
roleFlow = sum(edgeFlow(edgeMask));
end

function value = read_optional_map_value(computeMap, variableName, rowIndex, defaultValue)
%READ_OPTIONAL_MAP_VALUE 安全读取计算映射中的可选标量字段。
if ismember(variableName, computeMap.Properties.VariableNames)
    value = computeMap.(variableName)(rowIndex);
else
    value = defaultValue;
end
end

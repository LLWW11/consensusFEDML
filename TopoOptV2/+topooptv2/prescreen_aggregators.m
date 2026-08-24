function screening = prescreen_aggregators(scenario, config)
%PRESCREEN_AGGREGATORS 根据基础时空流路径预筛选聚合节点。
%   SCREENING = topooptv2.prescreen_aggregators(SCENARIO, CONFIG) 将基础图
%   的单位费用临时置零，计算从超级源点到超级汇点的最大流并分解为单位
%   客户端路径。多条物理路径的交汇点被划分为初始集合和候选集合；没有
%   交汇点的独立路径使用向上取整的空间中点作为初始聚合点。
%
%   本函数不修改输入 SCENARIO，也不会保存结果或绘制图形。CONFIG 当前仅
%   预留给后续筛选参数扩展，可以省略。

if nargin < 2 || isempty(config)
    config = struct(); %#ok<NASGU>
end

validate_scenario(scenario);
baseEdges = scenario.edges;
baseEdges.UnitCost = zeros(height(baseEdges), 1);
targetFlow = numel(scenario.client_ids);

if targetFlow == 0
    flowValue = 0;
    totalCost = 0;
    edgeFlow = zeros(height(baseEdges), 1);
    solverDiagnostics = empty_solver_diagnostics();
    nodePaths = cell(0, 1);
    edgePaths = cell(0, 1);
    remainingFlow = edgeFlow;
else
    [flowValue, totalCost, edgeFlow, solverDiagnostics] = ...
        topooptv2.min_cost_max_flow(scenario.node_count, baseEdges, ...
        scenario.source_id, scenario.sink_id, targetFlow);
    [nodePaths, edgePaths, remainingFlow] = ...
        topooptv2.decompose_flow_paths(scenario.node_count, baseEdges, ...
        edgeFlow, scenario.source_id, scenario.sink_id);
end

[physicalPaths, pathClientIds] = extract_physical_paths( ...
    baseEdges, edgePaths, scenario.client_ids, scenario.cloud_id, ...
    scenario.physical_node_count);
[initialSet, candidateSet, intersectionSet] = classify_path_nodes( ...
    physicalPaths, scenario.cloud_id);

allPhysicalNodes = 1:scenario.physical_node_count;
excludedNodes = unique([scenario.cloud_id; initialSet(:); candidateSet(:)]).';
otherSet = setdiff(allPhysicalNodes, excludedNodes, 'stable');

% 所有集合按物理节点编号排序，保证重复运行和平局处理完全确定。
initialSet = sort(unique(initialSet));
candidateSet = sort(unique(candidateSet));
intersectionSet = sort(unique(intersectionSet));
otherSet = sort(unique(otherSet));

screening = struct();
screening.initial = initialSet;
screening.candidate = candidateSet;
screening.others = otherSet;
screening.intersections = intersectionSet;
screening.max_flow = flowValue;
screening.total_cost = totalCost;
screening.edge_flow = edgeFlow;
screening.node_paths = nodePaths;
screening.edge_paths = edgePaths;
screening.physical_paths = physicalPaths;
screening.path_client_ids = pathClientIds;
screening.remaining_flow = remainingFlow;
screening.status = solverDiagnostics.status;
screening.solver_diagnostics = solverDiagnostics;
end

function validate_scenario(scenario)
%VALIDATE_SCENARIO 校验预筛选所需的场景字段和边表字段。
requiredFields = {'edges', 'node_count', 'source_id', 'sink_id', ...
    'physical_node_count', 'client_ids', 'cloud_id'};
for fieldIndex = 1:numel(requiredFields)
    if ~isfield(scenario, requiredFields{fieldIndex})
        error('topooptv2:prescreen_aggregators:MissingScenarioField', ...
            '场景缺少字段 %s。', requiredFields{fieldIndex});
    end
end
if ~istable(scenario.edges)
    error('topooptv2:prescreen_aggregators:InvalidEdges', ...
        'scenario.edges 必须是 table。');
end
requiredVariables = {'From', 'To', 'Capacity', 'UnitCost', 'Delay', ...
    'PhysicalFrom', 'PhysicalTo', 'Role'};
missingVariables = setdiff(requiredVariables, ...
    scenario.edges.Properties.VariableNames);
if ~isempty(missingVariables)
    error('topooptv2:prescreen_aggregators:MissingEdgeVariable', ...
        '边表缺少变量：%s。', strjoin(missingVariables, ', '));
end
end

function diagnostics = empty_solver_diagnostics()
%EMPTY_SOLVER_DIAGNOSTICS 构造零目标流量时的求解诊断信息。
diagnostics = struct();
diagnostics.status = 'optimal';
diagnostics.augmentations = 0;
diagnostics.elapsed_seconds = 0;
diagnostics.conservation_violation = 0;
diagnostics.capacity_violation = 0;
end

function [physicalPaths, pathClientIds] = extract_physical_paths( ...
        edges, edgePaths, clientIds, cloudId, physicalNodeCount)
%EXTRACT_PHYSICAL_PATHS 将单位流边路径转换为去除虚拟点的物理路径。
pathCount = numel(edgePaths);
physicalPaths = cell(pathCount, 1);
pathClientIds = nan(pathCount, 1);
roles = string(edges.Role);

for pathIndex = 1:pathCount
    pathEdgeIds = edgePaths{pathIndex};
    if isempty(pathEdgeIds)
        physicalPaths{pathIndex} = zeros(1, 0);
        continue;
    end

    firstEdge = pathEdgeIds(1);
    rawPhysicalPath = [edges.PhysicalFrom(firstEdge); ...
        edges.PhysicalTo(pathEdgeIds(:))];
    rawPhysicalPath = rawPhysicalPath(isfinite(rawPhysicalPath) & ...
        rawPhysicalPath >= 1 & rawPhysicalPath <= physicalNodeCount);
    rawPhysicalPath = remove_consecutive_duplicates(rawPhysicalPath(:).');
    physicalPaths{pathIndex} = rawPhysicalPath;

    sourceEdges = pathEdgeIds(roles(pathEdgeIds) == "source_input");
    if ~isempty(sourceEdges)
        candidateClient = edges.PhysicalTo(sourceEdges(1));
        if ismember(candidateClient, clientIds)
            pathClientIds(pathIndex) = candidateClient;
            continue;
        end
    end

    % 兼容没有显式 source_input 角色的人工小图，以路径首个客户端为准。
    clientPosition = find(ismember(rawPhysicalPath, clientIds), 1, 'first');
    if ~isempty(clientPosition)
        pathClientIds(pathIndex) = rawPhysicalPath(clientPosition);
    elseif ~isempty(rawPhysicalPath) && rawPhysicalPath(1) ~= cloudId
        pathClientIds(pathIndex) = rawPhysicalPath(1);
    end
end
end

function values = remove_consecutive_duplicates(values)
%REMOVE_CONSECUTIVE_DUPLICATES 删除物理路径中相邻的重复节点。
if numel(values) < 2
    return;
end
values = values([true, diff(values) ~= 0]);
end

function [initialSet, candidateSet, intersectionSet] = classify_path_nodes( ...
        physicalPaths, cloudId)
%CLASSIFY_PATH_NODES 按交汇关系划分初始、候选和独立路径中点。
pathCount = numel(physicalPaths);
eligiblePaths = cell(pathCount, 1);
allEligible = zeros(1, 0);

for pathIndex = 1:pathCount
    pathNodes = physicalPaths{pathIndex};
    % Metro 场景允许客户端物理节点同时承担聚合功能，因此仅排除云节点。
    pathNodes = pathNodes(pathNodes ~= cloudId);
    pathNodes = unique(pathNodes, 'stable');
    eligiblePaths{pathIndex} = pathNodes;
    allEligible = [allEligible, pathNodes]; %#ok<AGROW>
end

if isempty(allEligible)
    initialSet = zeros(1, 0);
    candidateSet = zeros(1, 0);
    intersectionSet = zeros(1, 0);
    return;
end

uniqueNodes = unique(allEligible);
pathOccurrence = zeros(size(uniqueNodes));
for nodeIndex = 1:numel(uniqueNodes)
    for pathIndex = 1:pathCount
        pathOccurrence(nodeIndex) = pathOccurrence(nodeIndex) + ...
            ismember(uniqueNodes(nodeIndex), eligiblePaths{pathIndex});
    end
end
intersectionSet = uniqueNodes(pathOccurrence >= 2);

initialSet = zeros(1, 0);
for pathIndex = 1:pathCount
    pathNodes = eligiblePaths{pathIndex};
    pathIntersections = pathNodes(ismember(pathNodes, intersectionSet));
    if ~isempty(pathIntersections)
        % 路径顺序由客户端指向云端，因此首个交汇点离客户端最近。
        initialSet(end + 1) = pathIntersections(1); %#ok<AGROW>
    elseif ~isempty(pathNodes)
        initialSet(end + 1) = choose_spatial_midpoint(pathNodes); %#ok<AGROW>
    end
end

initialSet = unique(initialSet);
candidateSet = setdiff(intersectionSet, initialSet, 'stable');
end

function midpointNode = choose_spatial_midpoint(pathNodes)
%CHOOSE_SPATIAL_MIDPOINT 选择路径中点，并按物理节点编号打破偶数平局。
nodeCount = numel(pathNodes);
leftIndex = floor((nodeCount + 1) / 2);
rightIndex = ceil((nodeCount + 1) / 2);
midpointNode = min(pathNodes([leftIndex, rightIndex]));
end

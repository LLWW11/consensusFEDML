function [flowValue, totalCost, edgeFlow, diagnostics] = min_cost_max_flow(nodeCount, edges, sourceId, sinkId, maxFlow)
%MIN_COST_MAX_FLOW 使用逐次最短增广路计算整数最小费用最大流。
%   [FLOWVALUE, TOTALCOST, EDGEFLOW, DIAGNOSTICS] =
%   MIN_COST_MAX_FLOW(NODECOUNT, EDGES, SOURCEID, SINKID, MAXFLOW) 在由
%   EDGES 描述的有向网络中，最多发送 MAXFLOW 单位整数流。EDGES 必须为
%   table，并包含 From、To、Capacity、UnitCost 和 Delay 五列。前向边费用
%   必须非负；算法在残量网络中显式使用反向边，因此不依赖 Optimization
%   Toolbox。
%
%   EDGEFLOW 与 EDGES 行一一对应。DIAGNOSTICS 返回求解状态、增广次数、
%   耗时、流守恒误差和容量约束误差。

    startTimer = tic;
    [fromNode, toNode, capacity, unitCost] = validate_inputs( ...
        nodeCount, edges, sourceId, sinkId, maxFlow);

    edgeCount = numel(fromNode);
    edgeFlow = zeros(edgeCount, 1);
    flowValue = 0;
    augmentationCount = 0;

    % 每轮在当前残量网络上寻找费用最小的可增广路径。
    while flowValue < maxFlow
        [pathFound, predecessorEdge, predecessorDirection] = ...
            find_shortest_residual_path(nodeCount, fromNode, toNode, ...
            capacity, unitCost, edgeFlow, sourceId, sinkId);

        if ~pathFound
            break;
        end

        [pathEdges, pathDirections, residualBottleneck] = ...
            reconstruct_path(nodeCount, fromNode, toNode, capacity, ...
            edgeFlow, predecessorEdge, predecessorDirection, ...
            sourceId, sinkId);

        augmentation = min(maxFlow - flowValue, residualBottleneck);
        if augmentation <= 0
            error('topooptv2:min_cost_max_flow:InvalidAugmentation', ...
                '内部错误：最短残量路径没有正的可增广容量。');
        end

        % 正向残量边增加原边流量，反向残量边撤销原边流量。
        for pathIndex = 1:numel(pathEdges)
            edgeIndex = pathEdges(pathIndex);
            edgeFlow(edgeIndex) = edgeFlow(edgeIndex) + ...
                pathDirections(pathIndex) * augmentation;
        end

        flowValue = flowValue + augmentation;
        augmentationCount = augmentationCount + 1;
    end

    % 用最终边流重新计算总费用，避免增量累计产生不必要的数值漂移。
    totalCost = sum(edgeFlow .* unitCost);
    [conservationViolation, capacityViolation] = calculate_violations( ...
        nodeCount, fromNode, toNode, capacity, edgeFlow, ...
        sourceId, sinkId, flowValue);

    if flowValue == maxFlow
        status = 'optimal';
        message = '已达到请求流量，并获得该流量下的最小费用解。';
    elseif flowValue == 0
        status = 'infeasible';
        message = '源点到汇点之间不存在正容量可行路径。';
    else
        status = 'partial';
        message = '网络容量不足，只得到小于请求值的最大可行流。';
    end

    diagnostics = struct( ...
        'status', status, ...
        'message', message, ...
        'augmentations', augmentationCount, ...
        'elapsed_seconds', toc(startTimer), ...
        'conservation_violation', conservationViolation, ...
        'capacity_violation', capacityViolation, ...
        'requested_flow', maxFlow);
end

function [fromNode, toNode, capacity, unitCost] = validate_inputs( ...
        nodeCount, edges, sourceId, sinkId, maxFlow)
%VALIDATE_INPUTS 严格校验网络、节点编号和整数容量输入。

    validateattributes(nodeCount, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', 'positive'}, ...
        mfilename, 'nodeCount');
    validateattributes(sourceId, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', '>=', 1, '<=', nodeCount}, ...
        mfilename, 'sourceId');
    validateattributes(sinkId, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', '>=', 1, '<=', nodeCount}, ...
        mfilename, 'sinkId');
    validateattributes(maxFlow, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', 'nonnegative'}, ...
        mfilename, 'maxFlow');

    if sourceId == sinkId
        error('topooptv2:min_cost_max_flow:IdenticalTerminals', ...
            'sourceId 与 sinkId 必须是不同节点。');
    end
    if ~istable(edges)
        error('topooptv2:min_cost_max_flow:InvalidEdgesType', ...
            'edges 必须是 table。');
    end

    requiredVariables = {'From', 'To', 'Capacity', 'UnitCost', 'Delay'};
    missingVariables = setdiff(requiredVariables, ...
        edges.Properties.VariableNames, 'stable');
    if ~isempty(missingVariables)
        error('topooptv2:min_cost_max_flow:MissingEdgeVariables', ...
            'edges 缺少必需列：%s。', strjoin(missingVariables, ', '));
    end

    % 每个表变量必须恰好为一列，以保证一行只描述一条有向边。
    for variableIndex = 1:numel(requiredVariables)
        variableName = requiredVariables{variableIndex};
        variableValue = edges.(variableName);
        if ~isnumeric(variableValue) || size(variableValue, 2) ~= 1
            error('topooptv2:min_cost_max_flow:InvalidEdgeVariable', ...
                'edges.%s 必须是数值列向量。', variableName);
        end
    end

    fromNode = double(edges.From);
    toNode = double(edges.To);
    capacity = double(edges.Capacity);
    unitCost = double(edges.UnitCost);
    delay = double(edges.Delay);

    validateattributes(fromNode, {'numeric'}, ...
        {'real', 'finite', 'integer', '>=', 1, '<=', nodeCount}, ...
        mfilename, 'edges.From');
    validateattributes(toNode, {'numeric'}, ...
        {'real', 'finite', 'integer', '>=', 1, '<=', nodeCount}, ...
        mfilename, 'edges.To');
    validateattributes(capacity, {'numeric'}, ...
        {'real', 'finite', 'integer', 'nonnegative'}, ...
        mfilename, 'edges.Capacity');
    validateattributes(unitCost, {'numeric'}, ...
        {'real', 'finite', 'nonnegative'}, ...
        mfilename, 'edges.UnitCost');
    validateattributes(delay, {'numeric'}, ...
        {'real', 'finite', 'nonnegative'}, ...
        mfilename, 'edges.Delay');
end

function [pathFound, predecessorEdge, predecessorDirection] = ...
        find_shortest_residual_path(nodeCount, fromNode, toNode, ...
        capacity, unitCost, edgeFlow, sourceId, sinkId)
%FIND_SHORTEST_RESIDUAL_PATH 用 Bellman-Ford 搜索最小费用残量路径。
%   原始边按表中行号顺序扫描，并且同一行总是先扫描正向残量边，再扫描
%   反向残量边。只接受严格更短的标签，因而平行边和平费用路径的选择是
%   确定性的。

    distance = inf(nodeCount, 1);
    distance(sourceId) = 0;
    predecessorEdge = zeros(nodeCount, 1);
    predecessorDirection = zeros(nodeCount, 1);
    edgeCount = numel(fromNode);

    % 无负费用环时，至多 nodeCount-1 轮松弛即可得到最短简单路径。
    for iteration = 1:(nodeCount - 1)
        changed = false;

        for edgeIndex = 1:edgeCount
            tail = fromNode(edgeIndex);
            head = toNode(edgeIndex);

            % 扫描原边对应的正向残量边。
            if edgeFlow(edgeIndex) < capacity(edgeIndex) && ...
                    isfinite(distance(tail))
                candidateDistance = distance(tail) + unitCost(edgeIndex);
                if candidateDistance < distance(head)
                    distance(head) = candidateDistance;
                    predecessorEdge(head) = edgeIndex;
                    predecessorDirection(head) = 1;
                    changed = true;
                end
            end

            % 扫描用于撤销既有流量的反向残量边。
            if edgeFlow(edgeIndex) > 0 && isfinite(distance(head))
                candidateDistance = distance(head) - unitCost(edgeIndex);
                if candidateDistance < distance(tail)
                    distance(tail) = candidateDistance;
                    predecessorEdge(tail) = edgeIndex;
                    predecessorDirection(tail) = -1;
                    changed = true;
                end
            end
        end

        if ~changed
            break;
        end
    end

    pathFound = isfinite(distance(sinkId));
end

function [pathEdges, pathDirections, residualBottleneck] = ...
        reconstruct_path(nodeCount, fromNode, toNode, capacity, edgeFlow, ...
        predecessorEdge, predecessorDirection, sourceId, sinkId)
%RECONSTRUCT_PATH 回溯最短路径并计算路径残量瓶颈。

    pathEdgesReversed = zeros(1, nodeCount - 1);
    pathDirectionsReversed = zeros(1, nodeCount - 1);
    pathLength = 0;
    currentNode = sinkId;
    residualBottleneck = inf;

    while currentNode ~= sourceId
        pathLength = pathLength + 1;
        if pathLength > nodeCount - 1
            error('topooptv2:min_cost_max_flow:PredecessorCycle', ...
                '内部错误：最短路径前驱链中出现环。');
        end

        edgeIndex = predecessorEdge(currentNode);
        direction = predecessorDirection(currentNode);
        if edgeIndex == 0 || direction == 0
            error('topooptv2:min_cost_max_flow:BrokenPredecessorChain', ...
                '内部错误：无法从汇点完整回溯到源点。');
        end

        pathEdgesReversed(pathLength) = edgeIndex;
        pathDirectionsReversed(pathLength) = direction;

        if direction == 1
            residualCapacity = capacity(edgeIndex) - edgeFlow(edgeIndex);
            currentNode = fromNode(edgeIndex);
        else
            residualCapacity = edgeFlow(edgeIndex);
            currentNode = toNode(edgeIndex);
        end
        residualBottleneck = min(residualBottleneck, residualCapacity);
    end

    pathEdges = fliplr(pathEdgesReversed(1:pathLength));
    pathDirections = fliplr(pathDirectionsReversed(1:pathLength));
end

function [conservationViolation, capacityViolation] = calculate_violations( ...
        nodeCount, fromNode, toNode, capacity, edgeFlow, ...
        sourceId, sinkId, flowValue)
%CALCULATE_VIOLATIONS 计算流守恒与容量约束的最大绝对违反量。

    nodeBalance = zeros(nodeCount, 1);
    for edgeIndex = 1:numel(edgeFlow)
        nodeBalance(fromNode(edgeIndex)) = ...
            nodeBalance(fromNode(edgeIndex)) - edgeFlow(edgeIndex);
        nodeBalance(toNode(edgeIndex)) = ...
            nodeBalance(toNode(edgeIndex)) + edgeFlow(edgeIndex);
    end

    expectedBalance = zeros(nodeCount, 1);
    expectedBalance(sourceId) = -flowValue;
    expectedBalance(sinkId) = flowValue;
    conservationViolation = max(abs(nodeBalance - expectedBalance));
    capacityViolation = max([0; -edgeFlow; edgeFlow - capacity]);
end

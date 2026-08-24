function [nodePaths, edgePaths, remainingFlow] = decompose_flow_paths( ...
        nodeCount, edges, edgeFlow, sourceId, sinkId)
%DECOMPOSE_FLOW_PATHS 将整数源汇流确定性地分解为单位流路径。
%   [NODEPATHS, EDGEPATHS, REMAININGFLOW] = DECOMPOSE_FLOW_PATHS(...)
%   按 EDGES 的原始行号顺序，在正流量子图中反复寻找从 SOURCEID 到
%   SINKID 的路径，每次扣除一个单位流。NODEPATHS 和 EDGEPATHS 为列
%   cell，每个元素分别保存一条单位流的节点序列和边行号序列。
%   REMAININGFLOW 保存无法归入源汇路径的剩余流量，例如独立环流。

    [fromNode, toNode, remainingFlow] = validate_inputs( ...
        nodeCount, edges, edgeFlow, sourceId, sinkId);

    nodePaths = cell(0, 1);
    edgePaths = cell(0, 1);

    while true
        [pathFound, predecessorEdge] = find_path_by_edge_order( ...
            nodeCount, fromNode, toNode, remainingFlow, sourceId, sinkId);
        if ~pathFound
            break;
        end

        [nodePath, edgePath] = reconstruct_path( ...
            nodeCount, fromNode, toNode, predecessorEdge, sourceId, sinkId);

        % 按定义每个 cell 只代表一个单位流，容量大于一的路径会重复出现。
        remainingFlow(edgePath) = remainingFlow(edgePath) - 1;
        nodePaths{end + 1, 1} = nodePath; %#ok<AGROW>
        edgePaths{end + 1, 1} = edgePath; %#ok<AGROW>
    end
end

function [fromNode, toNode, normalizedFlow] = validate_inputs( ...
        nodeCount, edges, edgeFlow, sourceId, sinkId)
%VALIDATE_INPUTS 校验流分解所需的图和非负整数边流。

    validateattributes(nodeCount, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', 'positive'}, ...
        mfilename, 'nodeCount');
    validateattributes(sourceId, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', '>=', 1, '<=', nodeCount}, ...
        mfilename, 'sourceId');
    validateattributes(sinkId, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', '>=', 1, '<=', nodeCount}, ...
        mfilename, 'sinkId');

    if sourceId == sinkId
        error('topooptv2:decompose_flow_paths:IdenticalTerminals', ...
            'sourceId 与 sinkId 必须是不同节点。');
    end
    if ~istable(edges)
        error('topooptv2:decompose_flow_paths:InvalidEdgesType', ...
            'edges 必须是 table。');
    end

    requiredVariables = {'From', 'To'};
    missingVariables = setdiff(requiredVariables, ...
        edges.Properties.VariableNames, 'stable');
    if ~isempty(missingVariables)
        error('topooptv2:decompose_flow_paths:MissingEdgeVariables', ...
            'edges 缺少必需列：%s。', strjoin(missingVariables, ', '));
    end

    if ~isnumeric(edges.From) || size(edges.From, 2) ~= 1 || ...
            ~isnumeric(edges.To) || size(edges.To, 2) ~= 1
        error('topooptv2:decompose_flow_paths:InvalidEndpointColumns', ...
            'edges.From 和 edges.To 必须是数值列向量。');
    end

    fromNode = double(edges.From);
    toNode = double(edges.To);
    validateattributes(fromNode, {'numeric'}, ...
        {'real', 'finite', 'integer', '>=', 1, '<=', nodeCount}, ...
        mfilename, 'edges.From');
    validateattributes(toNode, {'numeric'}, ...
        {'real', 'finite', 'integer', '>=', 1, '<=', nodeCount}, ...
        mfilename, 'edges.To');

    if ~isnumeric(edgeFlow) || (~isvector(edgeFlow) && ~isempty(edgeFlow)) || ...
            numel(edgeFlow) ~= height(edges)
        error('topooptv2:decompose_flow_paths:InvalidEdgeFlowSize', ...
            'edgeFlow 必须是与 edges 行数相同的数值向量。');
    end
    normalizedFlow = double(edgeFlow(:));
    validateattributes(normalizedFlow, {'numeric'}, ...
        {'real', 'finite', 'integer', 'nonnegative'}, ...
        mfilename, 'edgeFlow');
end

function [pathFound, predecessorEdge] = find_path_by_edge_order( ...
        nodeCount, fromNode, toNode, remainingFlow, sourceId, sinkId)
%FIND_PATH_BY_EDGE_ORDER 按边行号升序执行确定性广度优先搜索。

    visited = false(nodeCount, 1);
    predecessorEdge = zeros(nodeCount, 1);
    queue = zeros(nodeCount, 1);
    queueHead = 1;
    queueTail = 1;
    queue(queueTail) = sourceId;
    visited(sourceId) = true;
    pathFound = false;

    while queueHead <= queueTail && ~pathFound
        currentNode = queue(queueHead);
        queueHead = queueHead + 1;

        % FIND 天然返回升序行号，因此平行边和等长路径选择可重复。
        outgoingEdges = find(fromNode == currentNode & remainingFlow > 0);
        for listIndex = 1:numel(outgoingEdges)
            edgeIndex = outgoingEdges(listIndex);
            nextNode = toNode(edgeIndex);
            if visited(nextNode)
                continue;
            end

            visited(nextNode) = true;
            predecessorEdge(nextNode) = edgeIndex;
            if nextNode == sinkId
                pathFound = true;
                break;
            end

            queueTail = queueTail + 1;
            queue(queueTail) = nextNode;
        end
    end
end

function [nodePath, edgePath] = reconstruct_path( ...
        nodeCount, fromNode, toNode, predecessorEdge, sourceId, sinkId)
%RECONSTRUCT_PATH 根据广度优先搜索前驱恢复节点和边序列。

    reversedEdges = zeros(1, nodeCount - 1);
    pathLength = 0;
    currentNode = sinkId;

    while currentNode ~= sourceId
        pathLength = pathLength + 1;
        if pathLength > nodeCount - 1
            error('topooptv2:decompose_flow_paths:PredecessorCycle', ...
                '内部错误：单位流路径的前驱链中出现环。');
        end

        edgeIndex = predecessorEdge(currentNode);
        if edgeIndex == 0
            error('topooptv2:decompose_flow_paths:BrokenPredecessorChain', ...
                '内部错误：无法恢复完整的单位流路径。');
        end
        if toNode(edgeIndex) ~= currentNode
            error('topooptv2:decompose_flow_paths:InvalidPredecessorEdge', ...
                '内部错误：前驱边终点与当前节点不一致。');
        end
        reversedEdges(pathLength) = edgeIndex;
        currentNode = fromNode(edgeIndex);
    end

    edgePath = fliplr(reversedEdges(1:pathLength));
    nodePath = zeros(1, pathLength + 1);
    nodePath(1) = sourceId;
    for pathIndex = 1:pathLength
        nodePath(pathIndex + 1) = toNode(edgePath(pathIndex));
    end
end

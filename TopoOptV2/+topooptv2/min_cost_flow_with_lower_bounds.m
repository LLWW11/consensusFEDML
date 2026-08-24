function [isFeasible, totalCost, edgeFlow, diagnostics] = ...
        min_cost_flow_with_lower_bounds(nodeCount, edges, sourceId, ...
        sinkId, fixedFlow, lowerBounds)
%MIN_COST_FLOW_WITH_LOWER_BOUNDS 求解带边流量下界的固定流量最小费用流。
%   [ISFEASIBLE, TOTALCOST, EDGEFLOW, DIAGNOSTICS] =
%   topooptv2.min_cost_flow_with_lower_bounds(NODECOUNT, EDGES, SOURCEID,
%   SINKID, FIXEDFLOW, LOWERBOUNDS) 要求从源点到汇点恰好发送
%   FIXEDFLOW 单位整数流，并要求每条边的流量不小于对应下界。
%
%   实现采用标准下界循环流变换：加入一条“汇点到源点”的固定流量边，
%   扣除所有边的下界后，根据节点不平衡量连接超级源和超级汇，再复用
%   topooptv2.min_cost_max_flow 求出最小费用平衡流。函数不依赖
%   Optimization Toolbox。

startTimer = tic;
[edgeCount, lowerBounds] = validate_inputs(nodeCount, edges, sourceId, ...
    sinkId, fixedFlow, lowerBounds);

% 把固定的源汇流量表示为汇点到源点的固定回边，随后统一按循环流处理。
allFrom = [double(edges.From); sinkId];
allTo = [double(edges.To); sourceId];
allCapacity = [double(edges.Capacity); fixedFlow];
allCost = [double(edges.UnitCost); 0];
allDelay = [double(edges.Delay); 0];
allLower = [lowerBounds; fixedFlow];
residualCapacity = allCapacity - allLower;

% balance(v)=流入下界-流出下界。正值节点需要从超级源获得同量流入，
% 负值节点需要向超级汇发送同量流量。
balance = accumarray(allTo, allLower, [nodeCount, 1], @sum, 0) - ...
    accumarray(allFrom, allLower, [nodeCount, 1], @sum, 0);
positiveNodes = find(balance > 0);
negativeNodes = find(balance < 0);
requiredBalanceFlow = sum(balance(positiveNodes));

superSource = nodeCount + 1;
superSink = nodeCount + 2;
auxFrom = [allFrom; repmat(superSource, numel(positiveNodes), 1); ...
    negativeNodes];
auxTo = [allTo; positiveNodes; ...
    repmat(superSink, numel(negativeNodes), 1)];
auxCapacity = [residualCapacity; balance(positiveNodes); ...
    -balance(negativeNodes)];
auxCost = [allCost; zeros(numel(positiveNodes) + numel(negativeNodes), 1)];
auxDelay = [allDelay; zeros(numel(positiveNodes) + numel(negativeNodes), 1)];
auxEdges = table(auxFrom, auxTo, auxCapacity, auxCost, auxDelay, ...
    'VariableNames', {'From', 'To', 'Capacity', 'UnitCost', 'Delay'});

[balancedFlow, ~, auxiliaryFlow, auxiliaryDiagnostics] = ...
    topooptv2.min_cost_max_flow(nodeCount + 2, auxEdges, ...
    superSource, superSink, requiredBalanceFlow);
isFeasible = balancedFlow == requiredBalanceFlow;

if isFeasible
    % 原图流量等于预先扣除的下界与辅助网络剩余容量流量之和。
    edgeFlow = lowerBounds + auxiliaryFlow(1:edgeCount);
    totalCost = sum(edgeFlow .* double(edges.UnitCost));
    [conservationViolation, capacityViolation, lowerBoundViolation] = ...
        calculate_violations(nodeCount, edges, sourceId, sinkId, ...
        fixedFlow, edgeFlow, lowerBounds);
    status = 'optimal';
    message = '';
else
    % 不可行时不返回部分流，避免调用方把未满足下界的流误当成合法解。
    edgeFlow = zeros(edgeCount, 1);
    totalCost = inf;
    conservationViolation = inf;
    capacityViolation = inf;
    lowerBoundViolation = inf;
    status = 'infeasible';
    message = sprintf('下界平衡流仅完成 %g/%g 单位。', ...
        balancedFlow, requiredBalanceFlow);
end

diagnostics = struct();
diagnostics.status = status;
diagnostics.message = message;
diagnostics.elapsed_seconds = toc(startTimer);
diagnostics.requested_flow = fixedFlow;
diagnostics.required_balance_flow = requiredBalanceFlow;
diagnostics.balanced_flow = balancedFlow;
diagnostics.augmentations = auxiliaryDiagnostics.augmentations;
diagnostics.conservation_violation = conservationViolation;
diagnostics.capacity_violation = capacityViolation;
diagnostics.lower_bound_violation = lowerBoundViolation;
diagnostics.auxiliary_diagnostics = auxiliaryDiagnostics;
end


function [edgeCount, lowerBounds] = validate_inputs(nodeCount, edges, ...
        sourceId, sinkId, fixedFlow, lowerBounds)
%VALIDATE_INPUTS 校验固定流量、边表和整数下界。

validateattributes(nodeCount, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', 'positive'}, ...
    mfilename, 'nodeCount');
validateattributes(sourceId, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', '>=', 1, '<=', nodeCount}, ...
    mfilename, 'sourceId');
validateattributes(sinkId, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', '>=', 1, '<=', nodeCount}, ...
    mfilename, 'sinkId');
validateattributes(fixedFlow, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', 'nonnegative'}, ...
    mfilename, 'fixedFlow');
if sourceId == sinkId
    error('topooptv2:min_cost_flow_with_lower_bounds:IdenticalTerminals', ...
        'sourceId 与 sinkId 必须不同。');
end
if ~istable(edges)
    error('topooptv2:min_cost_flow_with_lower_bounds:InvalidEdgesType', ...
        'edges 必须是 table。');
end
requiredVariables = {'From', 'To', 'Capacity', 'UnitCost', 'Delay'};
if ~all(ismember(requiredVariables, edges.Properties.VariableNames))
    error('topooptv2:min_cost_flow_with_lower_bounds:MissingVariables', ...
        'edges 缺少固定流量下界求解所需变量。');
end
edgeCount = height(edges);
if ~isnumeric(lowerBounds) || ~isreal(lowerBounds) || ...
        ~isvector(lowerBounds) || numel(lowerBounds) ~= edgeCount || ...
        any(~isfinite(lowerBounds)) || any(lowerBounds < 0) || ...
        any(lowerBounds ~= floor(lowerBounds))
    error('topooptv2:min_cost_flow_with_lower_bounds:InvalidLowerBounds', ...
        'lowerBounds 必须是与边数相同的非负整数向量。');
end
lowerBounds = double(lowerBounds(:));
if any(lowerBounds > double(edges.Capacity))
    error('topooptv2:min_cost_flow_with_lower_bounds:LowerExceedsCapacity', ...
        '边流量下界不能超过对应容量。');
end
end


function [conservationViolation, capacityViolation, lowerViolation] = ...
        calculate_violations(nodeCount, edges, sourceId, sinkId, ...
        fixedFlow, edgeFlow, lowerBounds)
%CALCULATE_VIOLATIONS 计算流守恒、容量和下界约束的最大违反量。

nodeBalance = accumarray(double(edges.To), edgeFlow, ...
    [nodeCount, 1], @sum, 0) - accumarray(double(edges.From), edgeFlow, ...
    [nodeCount, 1], @sum, 0);
expectedBalance = zeros(nodeCount, 1);
expectedBalance(sourceId) = -fixedFlow;
expectedBalance(sinkId) = fixedFlow;
conservationViolation = max(abs(nodeBalance - expectedBalance));
capacityViolation = max([0; edgeFlow - double(edges.Capacity); -edgeFlow]);
lowerViolation = max([0; lowerBounds - edgeFlow]);
end

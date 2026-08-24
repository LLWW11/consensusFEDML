function tests = test_min_cost_max_flow
%TEST_MIN_COST_MAX_FLOW 返回最小费用最大流与流分解的单元测试集合。

    tests = functiontests(localfunctions);
end

function testSinglePath(testCase)
%TESTSINGLEPATH 验证单路径按请求流量增广并正确累计费用。

    edges = make_edges([1; 2], [2; 3], [3; 3], [2; 1], [1; 1]);
    [flowValue, totalCost, edgeFlow, diagnostics] = ...
        topooptv2.min_cost_max_flow(3, edges, 1, 3, 2);

    verifyEqual(testCase, flowValue, 2);
    verifyEqual(testCase, totalCost, 6);
    verifyEqual(testCase, edgeFlow, [2; 2]);
    verifyEqual(testCase, diagnostics.status, 'optimal');
    verifyEqual(testCase, diagnostics.conservation_violation, 0);
    verifyEqual(testCase, diagnostics.capacity_violation, 0);
end

function testParallelEdges(testCase)
%TESTPARALLELEDGES 验证平行边可分别承载流量且输出顺序与输入表一致。

    edges = make_edges([1; 1; 2], [2; 2; 3], [1; 1; 2], ...
        [2; 1; 0], [1; 1; 1]);
    [flowValue, totalCost, edgeFlow, diagnostics] = ...
        topooptv2.min_cost_max_flow(3, edges, 1, 3, 2);

    verifyEqual(testCase, flowValue, 2);
    verifyEqual(testCase, totalCost, 3);
    verifyEqual(testCase, edgeFlow, [1; 1; 2]);
    verifyEqual(testCase, diagnostics.augmentations, 2);
end

function testCapacityBottleneck(testCase)
%TESTCAPACITYBOTTLENECK 验证容量不足时返回最大部分流而不是抛错。

    edges = make_edges([1; 2], [2; 3], [3; 2], [1; 1], [1; 1]);
    [flowValue, totalCost, edgeFlow, diagnostics] = ...
        topooptv2.min_cost_max_flow(3, edges, 1, 3, 3);

    verifyEqual(testCase, flowValue, 2);
    verifyEqual(testCase, totalCost, 4);
    verifyEqual(testCase, edgeFlow, [2; 2]);
    verifyEqual(testCase, diagnostics.status, 'partial');
end

function testNoFeasibleFlow(testCase)
%TESTNOFEASIBLEFLOW 验证源汇不连通时返回零流和不可行状态。

    edges = make_edges(1, 2, 1, 1, 1);
    [flowValue, totalCost, edgeFlow, diagnostics] = ...
        topooptv2.min_cost_max_flow(3, edges, 1, 3, 1);

    verifyEqual(testCase, flowValue, 0);
    verifyEqual(testCase, totalCost, 0);
    verifyEqual(testCase, edgeFlow, 0);
    verifyEqual(testCase, diagnostics.status, 'infeasible');
end

function testZeroRequestedFlow(testCase)
%TESTZEROREQUESTEDFLOW 验证请求零流时直接返回合法最优空解。

    edges = make_edges(1, 2, 1, 1, 1);
    [flowValue, totalCost, edgeFlow, diagnostics] = ...
        topooptv2.min_cost_max_flow(2, edges, 1, 2, 0);

    verifyEqual(testCase, flowValue, 0);
    verifyEqual(testCase, totalCost, 0);
    verifyEqual(testCase, edgeFlow, 0);
    verifyEqual(testCase, diagnostics.status, 'optimal');
end

function testCheapestPathIsSelected(testCase)
%TESTCHEAPESTPATHISSELECTED 验证算法优先选择总单位费用较低的路径。

    edges = make_edges([1; 2; 1; 3], [2; 4; 3; 4], ones(4, 1), ...
        [1; 1; 3; 3], ones(4, 1));
    [flowValue, totalCost, edgeFlow] = ...
        topooptv2.min_cost_max_flow(4, edges, 1, 4, 1);

    verifyEqual(testCase, flowValue, 1);
    verifyEqual(testCase, totalCost, 2);
    verifyEqual(testCase, edgeFlow, [1; 1; 0; 0]);
end

function testResidualRerouting(testCase)
%TESTRESIDUALREROUTING 验证反向残量边能够撤销早期路径并恢复全局最优。

    edges = make_edges( ...
        [1; 1; 2; 2; 3; 3; 4; 5], ...
        [2; 3; 4; 5; 4; 5; 6; 6], ...
        ones(8, 1), ...
        [0; 0; 0; 1; 0; 100; 0; 0], ...
        ones(8, 1));
    [flowValue, totalCost, edgeFlow, diagnostics] = ...
        topooptv2.min_cost_max_flow(6, edges, 1, 6, 2);

    verifyEqual(testCase, flowValue, 2);
    verifyEqual(testCase, totalCost, 1);
    verifyEqual(testCase, edgeFlow, [1; 1; 0; 1; 1; 0; 1; 1]);
    verifyEqual(testCase, diagnostics.status, 'optimal');
end

function testFlowDecomposition(testCase)
%TESTFLOWDECOMPOSITION 验证平行边流按边行号稳定分解为单位路径。

    edges = make_edges([1; 1; 2], [2; 2; 3], [1; 1; 2], ...
        [2; 1; 0], [1; 1; 1]);
    [~, ~, edgeFlow] = topooptv2.min_cost_max_flow(3, edges, 1, 3, 2);
    [nodePaths, edgePaths, remainingFlow] = ...
        topooptv2.decompose_flow_paths(3, edges, edgeFlow, 1, 3);

    verifyEqual(testCase, numel(nodePaths), 2);
    verifyEqual(testCase, nodePaths{1}, [1, 2, 3]);
    verifyEqual(testCase, nodePaths{2}, [1, 2, 3]);
    verifyEqual(testCase, edgePaths{1}, [1, 3]);
    verifyEqual(testCase, edgePaths{2}, [2, 3]);
    verifyEqual(testCase, remainingFlow, zeros(3, 1));
end

function edges = make_edges(fromNode, toNode, capacity, unitCost, delay)
%MAKE_EDGES 创建测试使用的标准有向边表。

    edges = table(fromNode(:), toNode(:), capacity(:), unitCost(:), ...
        delay(:), 'VariableNames', ...
        {'From', 'To', 'Capacity', 'UnitCost', 'Delay'});
end

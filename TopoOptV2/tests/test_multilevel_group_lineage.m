function summary = test_multilevel_group_lineage()
%TEST_MULTILEVEL_GROUP_LINEAGE 验证串联聚合会传播完整客户端谱系。
%   人工图先在节点 4 的第一层聚合客户端 [1,2]，再在第二层把该上游
%   代表流与新到达的客户端 3 聚合。测试确认下游组保存三个原始客户端，
%   同时每个客户端仍只有一个明确的首次聚合点。

config = topooptv2.default_config();
config.compute_capacity = 4;
config = topooptv2.validate_config(config);

scenario = topooptv2.build_scenario( ...
    build_serial_aggregation_tensor(), [1, 2, 3], 5, config);
solution = topooptv2.solve_aggregation_scheme( ...
    scenario, 4, config);

assert(strcmp(solution.status, 'optimal'), ...
    '串联聚合人工图没有返回最优状态：%s', solution.error_message);
assert(solution.successful_client_count == 3 && ...
    solution.aggregated_client_count == 3, ...
    '串联聚合人工图没有完整服务三个客户端。');
assert(isequal(solution.effective_aggregators, 4), ...
    '串联聚合人工图没有保留有效聚合物理节点 4。');

groupAtLayerOne = find_group(solution.group_mapping, 4, 1);
groupAtLayerTwo = find_group(solution.group_mapping, 4, 2);
assert(isequal(groupAtLayerOne.client_ids, [1, 2]), ...
    '节点 4 第一层聚合组的客户端谱系不正确。');
assert(isequal(groupAtLayerTwo.client_ids, [1, 2, 3]), ...
    '下游串联聚合组没有继承上游组并加入客户端 3 的完整谱系。');

assert(isequal(groupAtLayerOne.first_aggregation_client_ids, [1, 2]), ...
    '第一层聚合组没有记录在该处首次聚合的客户端。');
assert(isequal(groupAtLayerTwo.first_aggregation_client_ids, 3), ...
    '第二层聚合组没有只记录新加入聚合的客户端 3。');
verify_unique_first_aggregation(solution.flow_paths);

summary = struct();
summary.schema_version = '1.0-test';
summary.downstream_aggregator_id = groupAtLayerTwo.aggregator_id;
summary.downstream_layer = groupAtLayerTwo.layer;
summary.downstream_client_ids = groupAtLayerTwo.client_ids;
summary.passed = true;
fprintf('多层谱系测试通过：下游组恢复 %d 个原始客户端。\n', ...
    numel(summary.downstream_client_ids));
end


function capacityTensor = build_serial_aggregation_tensor()
%BUILD_SERIAL_AGGREGATION_TENSOR 构造上游代表流与新客户端再聚合的三层图。

capacityTensor = zeros(5, 5, 3);

% 第一层由客户端 1 和 2 形成首个聚合组。
capacityTensor(1, 4, 1) = 1;
capacityTensor(2, 4, 1) = 1;

% 客户端 3 经自身存储边到达第二层后加入上游代表流。
capacityTensor(3, 4, 2) = 1;

% 下游聚合结果在第三层进入云节点。
capacityTensor(4, 5, 3) = 1;
end


function group = find_group(groupMapping, aggregatorId, layer)
%FIND_GROUP 按物理聚合点和时间层唯一定位分组记录。

matches = find([groupMapping.aggregator_id] == aggregatorId & ...
    [groupMapping.layer] == layer);
assert(isscalar(matches), ...
    '未能唯一定位聚合点 %d 的第 %d 层记录。', aggregatorId, layer);
group = groupMapping(matches);
end


function verify_unique_first_aggregation(flowPaths)
%VERIFY_UNIQUE_FIRST_AGGREGATION 校验每个客户端的首次聚合字段及兼容别名。

assert(all(isfield(flowPaths, {'first_aggregator_id', ...
    'first_aggregation_layer'})), '单位流路径缺少显式首次聚合字段。');
clientIds = [flowPaths.client_id];
assert(isequal(sort(clientIds), [1, 2, 3]), ...
    '单位流路径没有与三个原始客户端一一对应。');

for pathIndex = 1:numel(flowPaths)
    path = flowPaths(pathIndex);
    assert(path.aggregator_id == path.first_aggregator_id && ...
        path.aggregation_layer == path.first_aggregation_layer, ...
        '兼容字段与显式首次聚合字段语义不一致。');
    if ismember(path.client_id, [1, 2])
        expectedLayer = 1;
    else
        expectedLayer = 2;
    end
    assert(path.first_aggregator_id == 4 && ...
        path.first_aggregation_layer == expectedLayer, ...
        '客户端 %d 的唯一首次聚合映射不正确。', path.client_id);
end
end

function summary = test_client_coverage_postprocess()
%TEST_CLIENT_COVERAGE_POSTPROCESS 验证客户端覆盖交换的核心约束。
%   使用合成六方法数据检查默认保留模式、硬覆盖、HFL 组规模不变、后续轮次
%   不变、结果可复现，以及总参与名额不足时达到理论最大覆盖数。

postprocess_paths();
base_data = build_synthetic_data(8, 6, 2);

% 默认保留模式只审计，不应修改任何映射。
[preserved, preserve_audit] = enforce_client_coverage(base_data, 'preserve', 4);
assert(isequaln(preserved, base_data), 'preserve 模式修改了输入映射。');
assert(preserve_audit.total_swaps == 0, 'preserve 模式产生了交换记录。');

% 硬覆盖应只替换前四轮中的客户端编号，并保持所有人数与分组规模。
[controlled, hard_audit] = enforce_client_coverage(base_data, 'hard', 4);
assert(hard_audit.total_swaps > 0, '合成数据未触发硬覆盖交换。');
assert(hard_audit.all_feasible_columns_covered, ...
    '具备足够名额的合成列未实现全覆盖。');
assert_count_matrices_unchanged(base_data, controlled);
assert_group_shapes_unchanged(base_data, controlled);
assert_round_tail_unchanged(base_data, controlled, 5);
assert_all_method_columns_covered(controlled, 4);

% 硬覆盖排序必须稳定，相同输入应产生逐元素一致的结果和交换记录。
[controlled_again, hard_audit_again] = ...
    enforce_client_coverage(base_data, 'hard', 4);
assert(isequaln(controlled_again, controlled), '硬覆盖结果不可复现。');
assert(isequaln(hard_audit_again.methods, hard_audit.methods), ...
    '硬覆盖审计中的交换记录不可复现。');

% 一轮只有两个名额，无法覆盖五个客户端；结果必须达到理论最大覆盖数 2。
infeasible_data = build_synthetic_data(3, 6, 2);
[~, infeasible_audit] = enforce_client_coverage( ...
    infeasible_data, 'hard', 1);
method_names = fieldnames(infeasible_audit.methods);
for method_index = 1:numel(method_names)
    column = infeasible_audit.methods.(method_names{method_index}).columns(1);
    assert(~column.is_feasible, '不可行测试列被误判为可行。');
    assert(column.covered_client_count_after == ...
        column.theoretical_max_covered_clients, ...
        '不可行测试列未达到理论最大覆盖数。');
end

summary = struct();
summary.schema_version = '1.0-test';
summary.hard_swap_count = hard_audit.total_swaps;
summary.infeasible_warning_policy_checked = true;
summary.passed = true;
fprintf('客户端覆盖合成测试通过：硬覆盖交换 %d 次。\n', ...
    hard_audit.total_swaps);
end


function data = build_synthetic_data(epoch_count, node_count, per_round_count)
%BUILD_SYNTHETIC_DATA 构造六方法共享尺寸的合成人数和嵌套映射数据。
data = struct();
data.num_of_nodes = node_count;
data.Cloud = node_count;
util_count = 2;
counts = per_round_count .* ones(epoch_count, util_count);
client_fields = {'client_num_HFLSnF_fix', 'client_num_HFLSnF_los', ...
    'client_num_HFLnoSnF_fix', 'client_num_HFLnoSnF_los', ...
    'client_num_FLSnF', 'client_num_FLnoSnF'};
for field_index = 1:numel(client_fields)
    data.(client_fields{field_index}) = counts;
end

data.actual_c2e_map_HFLSnF = cell(epoch_count, util_count);
data.actual_c2e_map_HFLnoSnF = cell(epoch_count, util_count);
data.c2cmap_FLSnF_all = cell(epoch_count, util_count);
data.c2cmap_FLnoSnF_all = cell(epoch_count, util_count);
for util_index = 1:util_count
    for round_index = 1:epoch_count
        flat_mapping = 1:per_round_count;
        policy_mapping = cell(1, 3);
        policy_mapping{1} = {flat_mapping(1), flat_mapping(2:end)};
        policy_mapping{2} = [];
        policy_mapping{3} = {flat_mapping(1), flat_mapping(2:end)};
        data.actual_c2e_map_HFLSnF{round_index, util_index} = policy_mapping;
        data.actual_c2e_map_HFLnoSnF{round_index, util_index} = policy_mapping;
        data.c2cmap_FLSnF_all{round_index, util_index} = flat_mapping;
        data.c2cmap_FLnoSnF_all{round_index, util_index} = flat_mapping;
    end
end
end


function assert_count_matrices_unchanged(before, after)
%ASSERT_COUNT_MATRICES_UNCHANGED 校验覆盖处理未修改六种逐轮人数矩阵。
fields = {'client_num_HFLSnF_fix', 'client_num_HFLSnF_los', ...
    'client_num_HFLnoSnF_fix', 'client_num_HFLnoSnF_los', ...
    'client_num_FLSnF', 'client_num_FLnoSnF'};
for field_index = 1:numel(fields)
    assert(isequal(before.(fields{field_index}), after.(fields{field_index})), ...
        '覆盖处理修改了人数矩阵：%s', fields{field_index});
end
end


function assert_group_shapes_unchanged(before, after)
%ASSERT_GROUP_SHAPES_UNCHANGED 校验 HFL 每个策略的分组数和组内人数不变。
mapping_fields = {'actual_c2e_map_HFLSnF', 'actual_c2e_map_HFLnoSnF'};
policy_indices = [1, 3];
for field_index = 1:numel(mapping_fields)
    field_name = mapping_fields{field_index};
    for snapshot_index = 1:numel(before.(field_name))
        for policy_index = policy_indices
            before_groups = before.(field_name){snapshot_index}{policy_index};
            after_groups = after.(field_name){snapshot_index}{policy_index};
            assert(isequal(size(before_groups), size(after_groups)), ...
                '%s 的分组边界发生变化。', field_name);
            assert(isequal(cellfun(@numel, before_groups), ...
                cellfun(@numel, after_groups)), ...
                '%s 的组内人数发生变化。', field_name);
        end
    end
end
end


function assert_round_tail_unchanged(before, after, first_tail_round)
%ASSERT_ROUND_TAIL_UNCHANGED 校验覆盖窗口之后的六种映射完全不变。
fields = {'actual_c2e_map_HFLSnF', 'actual_c2e_map_HFLnoSnF', ...
    'c2cmap_FLSnF_all', 'c2cmap_FLnoSnF_all'};
for field_index = 1:numel(fields)
    field_name = fields{field_index};
    assert(isequaln(before.(field_name)(first_tail_round:end, :), ...
        after.(field_name)(first_tail_round:end, :)), ...
        '覆盖窗口之后的映射发生变化：%s', field_name);
end
end


function assert_all_method_columns_covered(data, coverage_horizon)
%ASSERT_ALL_METHOD_COLUMNS_COVERED 独立检查六方法各利用率均覆盖全部客户端。
[~, audit] = enforce_client_coverage(data, 'preserve', coverage_horizon);
method_names = fieldnames(audit.methods);
for method_index = 1:numel(method_names)
    columns = audit.methods.(method_names{method_index}).columns;
    assert(all([columns.all_clients_covered_after]), ...
        '%s 存在未覆盖客户端。', method_names{method_index});
end
end

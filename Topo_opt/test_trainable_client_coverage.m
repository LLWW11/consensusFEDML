function summary = test_trainable_client_coverage()
%TEST_TRAINABLE_CLIENT_COVERAGE 验证现有 MAT 的前 150 轮硬覆盖结果。
%   分别生成 preserve 和 hard 临时训练文件，检查六种方法全部利用率的覆盖、
%   人数和方差不变、HFL 组规模不变、聚合时间不变以及后 50 轮映射不变。

script_directory = fileparts(mfilename('fullpath'));
input_file = fullfile(script_directory, 'result-U-6fixedge_epoch200.mat');
if ~isfile(input_file)
    error('test_trainable_client_coverage:SourceNotFound', ...
        '测试所需原始结果不存在：%s', input_file);
end

preserve_file = [tempname, '_preserve.mat'];
hard_file = [tempname, '_hard.mat'];
cleanup_guard = onCleanup(@() cleanup_test_files(preserve_file, hard_file));
preserve_audit = build_trainable_varalpha_mat( ...
    input_file, preserve_file, 0.1, 'preserve', 150);
hard_audit = build_trainable_varalpha_mat( ...
    input_file, hard_file, 0.1, 'hard', 150);
preserved = load(preserve_file);
controlled = load(hard_file);

assert(hard_audit.coverage.all_feasible_columns_covered, ...
    '现有 MAT 中具备足够名额的列未实现全覆盖。');
assert(hard_audit.coverage.total_swaps > 0, ...
    '现有 MAT 未产生覆盖交换，测试未覆盖问题场景。');
assert(strcmp(preserve_audit.coverage.mode, 'preserve') && ...
    preserve_audit.coverage.total_swaps == 0, ...
    'preserve 模式审计不正确。');

client_fields = {'client_num_HFLSnF_fix', 'client_num_HFLSnF_los', ...
    'client_num_HFLnoSnF_fix', 'client_num_HFLnoSnF_los', ...
    'client_num_FLSnF', 'client_num_FLnoSnF'};
for field_index = 1:numel(client_fields)
    field_name = client_fields{field_index};
    assert(isequal(preserved.(field_name), controlled.(field_name)), ...
        '硬覆盖修改了人数矩阵：%s', field_name);
    assert(isequal(mean(preserved.(field_name), 1), ...
        mean(controlled.(field_name), 1)), ...
        '硬覆盖修改了人数均值：%s', field_name);
    assert(isequal(var(preserved.(field_name), 0, 1), ...
        var(controlled.(field_name), 0, 1)), ...
        '硬覆盖修改了人数方差：%s', field_name);
end

time_fields = {'time_agg_HFLSnF_fix', 'time_agg_HFLSnF_los', ...
    'time_agg_HFLnoSnF_fix', 'time_agg_HFLnoSnF_los', ...
    'time_agg_FLSnF', 'time_agg_FLnoSnF'};
for field_index = 1:numel(time_fields)
    field_name = time_fields{field_index};
    assert(isequaln(preserved.(field_name), controlled.(field_name)), ...
        '硬覆盖修改了聚合时间：%s', field_name);
end

assert_hfl_group_sizes_equal(preserved, controlled);
assert_mapping_tail_equal(preserved, controlled, 151);
assert_coverage_audit_complete(hard_audit.coverage);

summary = struct();
summary.schema_version = '1.0-test';
summary.varAlpha = 0.1;
summary.coverage_horizon = 150;
summary.total_swaps = hard_audit.coverage.total_swaps;
summary.passed = true;
fprintf('现有 MAT 客户端覆盖测试通过：前 150 轮交换 %d 次。\n', ...
    summary.total_swaps);
clear cleanup_guard;
end


function assert_hfl_group_sizes_equal(before, after)
%ASSERT_HFL_GROUP_SIZES_EQUAL 校验四种 HFL 策略的组边界和组内人数未改变。
mapping_fields = {'actual_c2e_map_HFLSnF', 'actual_c2e_map_HFLnoSnF'};
policy_indices = [1, 3];
for field_index = 1:numel(mapping_fields)
    field_name = mapping_fields{field_index};
    for snapshot_index = 1:numel(before.(field_name))
        for policy_index = policy_indices
            before_groups = before.(field_name){snapshot_index}{policy_index};
            after_groups = after.(field_name){snapshot_index}{policy_index};
            assert(isequal(size(before_groups), size(after_groups)), ...
                '%s 第 %d 个快照的分组边界发生变化。', ...
                field_name, snapshot_index);
            if iscell(before_groups)
                assert(isequal(cellfun(@numel, before_groups), ...
                    cellfun(@numel, after_groups)), ...
                    '%s 第 %d 个快照的组内人数发生变化。', ...
                    field_name, snapshot_index);
            else
                assert(numel(before_groups) == numel(after_groups), ...
                    '%s 第 %d 个快照的人数发生变化。', ...
                    field_name, snapshot_index);
            end
        end
    end
end
end


function assert_mapping_tail_equal(before, after, first_tail_round)
%ASSERT_MAPPING_TAIL_EQUAL 校验覆盖窗口后的 HFL 和 FL 映射完全不变。
mapping_fields = {'actual_c2e_map_HFLSnF', 'actual_c2e_map_HFLnoSnF', ...
    'c2cmap_FLSnF_all', 'c2cmap_FLnoSnF_all'};
for field_index = 1:numel(mapping_fields)
    field_name = mapping_fields{field_index};
    assert(isequaln(before.(field_name)(first_tail_round:end, :), ...
        after.(field_name)(first_tail_round:end, :)), ...
        '覆盖窗口之后的映射发生变化：%s', field_name);
end
end


function assert_coverage_audit_complete(coverage_audit)
%ASSERT_COVERAGE_AUDIT_COMPLETE 校验六方法全部利用率的审计均达到全覆盖。
method_names = fieldnames(coverage_audit.methods);
assert(numel(method_names) == 6, '覆盖审计未包含六种方法。');
for method_index = 1:numel(method_names)
    method = coverage_audit.methods.(method_names{method_index});
    columns = method.columns;
    assert(all([columns.is_feasible]), ...
        '%s 的现有 MAT 列被意外判定为不可行。', method_names{method_index});
    assert(all([columns.all_clients_covered_after]), ...
        '%s 仍存在永久缺席客户端。', method_names{method_index});
    assert(all(cellfun(@isempty, {columns.missing_client_ids_after})), ...
        '%s 的审计仍记录了缺席客户端。', method_names{method_index});
end
end


function cleanup_test_files(varargin)
%CLEANUP_TEST_FILES 删除端到端测试生成的临时 MAT 文件。
for file_index = 1:nargin
    if isfile(varargin{file_index})
        delete(varargin{file_index});
    end
end
end

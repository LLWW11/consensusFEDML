function summary = validate_corrected_result(result_file)
%VALIDATE_CORRECTED_RESULT 独立复算并校验修正后的历史拓扑结果。
%   summary = VALIDATE_CORRECTED_RESULT() 校验默认 corrected MAT 的策略位置、
%   客户端人数、映射、聚合时间、汇总值以及历史映射缺失标记。

paths = postprocess_paths();
if nargin < 1 || isempty(result_file)
    result_file = fullfile(paths.output_directory, ...
        'result-U-6fixedge_epoch200_corrected.mat');
end
result_file = char(result_file);
if ~isfile(result_file)
    error('validate_corrected_result:FileNotFound', '找不到修正文件：%s', result_file);
end

data = load(result_file);
expected_size = [double(data.epoch_num), numel(data.total_util)];
if ~isequal(expected_size, [200, 8])
    error('validate_corrected_result:UnexpectedExperimentSize', ...
        '修正文件尺寸为 %s，期望 [200 8]。', mat2str(expected_size));
end

valid_client_ids = setdiff(1:data.num_of_nodes, data.Cloud);
for index = 1:prod(expected_size)
    group_snf = data.group_num_HFLSnF{index};
    client_snf = data.client_num_HFLSnF{index};
    group_no_snf = data.group_num_HFLnoSnF{index};
    client_no_snf = data.client_num_HFLnoSnF{index};

    % 第 1 项是动态策略，第 3 项是固定策略。
    assert_scalar_equal(data.group_num_HFLSnF_los(index), group_snf{1,1}, ...
        'group_num_HFLSnF_los', index);
    assert_scalar_equal(data.group_num_HFLSnF_fix(index), group_snf{1,3}, ...
        'group_num_HFLSnF_fix', index);
    assert_scalar_equal(data.group_num_HFLnoSnF_los(index), group_no_snf{1,1}, ...
        'group_num_HFLnoSnF_los', index);
    assert_scalar_equal(data.group_num_HFLnoSnF_fix(index), group_no_snf{1,3}, ...
        'group_num_HFLnoSnF_fix', index);

    assert_scalar_equal(data.client_num_HFLSnF_los(index), sum(client_snf{1,1}), ...
        'client_num_HFLSnF_los', index);
    assert_scalar_equal(data.client_num_HFLSnF_fix(index), sum(client_snf{1,3}), ...
        'client_num_HFLSnF_fix', index);
    assert_scalar_equal(data.client_num_HFLnoSnF_los(index), sum(client_no_snf{1,1}), ...
        'client_num_HFLnoSnF_los', index);
    assert_scalar_equal(data.client_num_HFLnoSnF_fix(index), sum(client_no_snf{1,3}), ...
        'client_num_HFLnoSnF_fix', index);

    assert_scalar_equal(data.time_agg_HFLSnF_los(index), calc_hfl_agg_time( ...
        client_snf{1,1}, group_snf{1,1}, data.agg_time_per_clients), ...
        'time_agg_HFLSnF_los', index);
    assert_scalar_equal(data.time_agg_HFLSnF_fix(index), calc_hfl_agg_time( ...
        client_snf{1,3}, group_snf{1,3}, data.agg_time_per_clients), ...
        'time_agg_HFLSnF_fix', index);
    assert_scalar_equal(data.time_agg_HFLnoSnF_los(index), calc_hfl_agg_time( ...
        client_no_snf{1,1}, group_no_snf{1,1}, data.agg_time_per_clients), ...
        'time_agg_HFLnoSnF_los', index);
    assert_scalar_equal(data.time_agg_HFLnoSnF_fix(index), calc_hfl_agg_time( ...
        client_no_snf{1,3}, group_no_snf{1,3}, data.agg_time_per_clients), ...
        'time_agg_HFLnoSnF_fix', index);

    maps_snf = data.actual_c2e_map_HFLSnF{index};
    validate_client_mapping(maps_snf{1,1}, data.client_num_HFLSnF_los(index), ...
        valid_client_ids, sprintf('corrected HFL-SnF-dynamic cell=%d', index));
    validate_client_mapping(maps_snf{1,3}, data.client_num_HFLSnF_fix(index), ...
        valid_client_ids, sprintf('corrected HFL-SnF-fixed cell=%d', index));
end

if any(data.group_num_HFLSnF_fix(:) > numel(data.EdgeSet)) || ...
        any(data.group_num_HFLnoSnF_fix(:) > numel(data.EdgeSet))
    error('validate_corrected_result:FixedGroupLimit', '固定边缘组数超过 EdgeSet 大小。');
end

assert_matrix_equal(data.time_agg_FLnoSnF, ...
    data.client_num_FLnoSnF .* data.agg_time_per_clients, 'time_agg_FLnoSnF');
assert_matrix_equal(data.time_agg_FLSnF, ...
    data.client_num_FLSnF .* data.agg_time_per_clients, 'time_agg_FLSnF');

% 汇总字段必须等于底层 200 次结果的列均值。
assert_matrix_equal(data.enum_HFLSnF_fix, mean(data.group_num_HFLSnF_fix, 1), 'enum_HFLSnF_fix');
assert_matrix_equal(data.enum_HFLSnF_los, mean(data.group_num_HFLSnF_los, 1), 'enum_HFLSnF_los');
assert_matrix_equal(data.enum_HFLnoSnF_fix, mean(data.group_num_HFLnoSnF_fix, 1), 'enum_HFLnoSnF_fix');
assert_matrix_equal(data.enum_HFLnoSnF_los, mean(data.group_num_HFLnoSnF_los, 1), 'enum_HFLnoSnF_los');
assert_matrix_equal(data.cnum_HFLSnF_fix, mean(data.client_num_HFLSnF_fix, 1), 'cnum_HFLSnF_fix');
assert_matrix_equal(data.cnum_HFLSnF_los, mean(data.client_num_HFLSnF_los, 1), 'cnum_HFLSnF_los');
assert_matrix_equal(data.cnum_HFLnoSnF_fix, mean(data.client_num_HFLnoSnF_fix, 1), 'cnum_HFLnoSnF_fix');
assert_matrix_equal(data.cnum_HFLnoSnF_los, mean(data.client_num_HFLnoSnF_los, 1), 'cnum_HFLnoSnF_los');
assert_matrix_equal(data.cnum_FLnoSnF, mean(data.client_num_FLnoSnF, 1), 'cnum_FLnoSnF');
assert_matrix_equal(data.cnum_FLSnF, mean(data.client_num_FLSnF, 1), 'cnum_FLSnF');
assert_matrix_equal(data.aver_time_agg_HFLSnF_fix, mean(data.time_agg_HFLSnF_fix, 1), ...
    'aver_time_agg_HFLSnF_fix');
assert_matrix_equal(data.aver_time_agg_HFLSnF_los, mean(data.time_agg_HFLSnF_los, 1), ...
    'aver_time_agg_HFLSnF_los');
assert_matrix_equal(data.aver_time_agg_HFLnoSnF_fix, mean(data.time_agg_HFLnoSnF_fix, 1), ...
    'aver_time_agg_HFLnoSnF_fix');
assert_matrix_equal(data.aver_time_agg_HFLnoSnF_los, mean(data.time_agg_HFLnoSnF_los, 1), ...
    'aver_time_agg_HFLnoSnF_los');
assert_matrix_equal(data.aver_time_agg_FLnoSnF, mean(data.time_agg_FLnoSnF, 1), ...
    'aver_time_agg_FLnoSnF');
assert_matrix_equal(data.aver_time_agg_FLSnF, mean(data.time_agg_FLSnF, 1), ...
    'aver_time_agg_FLSnF');

assert_matrix_equal(data.toall_time_HFLSnF_fix, ...
    data.time_HFLSnF_fix + data.aver_time_agg_HFLSnF_fix, 'toall_time_HFLSnF_fix');
assert_matrix_equal(data.toall_time_HFLSnF_los, ...
    data.time_HFLSnF_los + data.aver_time_agg_HFLSnF_los, 'toall_time_HFLSnF_los');
assert_matrix_equal(data.toall_time_HFLnoSnF_fix, ...
    data.time_HFLnoSnF_fix + data.aver_time_agg_HFLnoSnF_fix, 'toall_time_HFLnoSnF_fix');
assert_matrix_equal(data.toall_time_HFLnoSnF_los, ...
    data.time_HFLnoSnF_los + data.aver_time_agg_HFLnoSnF_los, 'toall_time_HFLnoSnF_los');
assert_matrix_equal(data.toall_time_FLnoSnF, ...
    data.time_FLnoSnF + data.aver_time_agg_FLnoSnF, 'toall_time_FLnoSnF');
assert_matrix_equal(data.toall_time_FLSnF, ...
    data.time_FLSnF + data.aver_time_agg_FLSnF, 'toall_time_FLSnF');

% 旧文件缺失的四类映射必须明确标记，且不能在修正文件中伪造。
if isfield(data, 'actual_c2e_map_HFLnoSnF') || isfield(data, 'c2cmap_FLnoSnF_all') || ...
        isfield(data, 'c2cmap_FLSnF_all')
    error('validate_corrected_result:FabricatedMappings', '修正文件中出现了无法恢复的历史映射。');
end
if data.mapping_availability.HFL_noSnF_dynamic_and_fixed || ...
        data.mapping_availability.FL_noSnF || data.mapping_availability.FL_SnF
    error('validate_corrected_result:WrongMappingAvailability', '历史映射可用性标记不正确。');
end

audit = data.repair_audit;
assert_scalar_equal(audit.hfl_no_snf_group_cells_rebuilt, 1600, ...
    'hfl_no_snf_group_cells_rebuilt', 1);
assert_scalar_equal(audit.hfl_no_snf_group_pair_cells_changed, 1133, ...
    'hfl_no_snf_group_pair_cells_changed', 1);
assert_scalar_equal(audit.hfl_no_snf_time_pair_cells_changed, 1133, ...
    'hfl_no_snf_time_pair_cells_changed', 1);
assert_scalar_equal(audit.hfl_no_snf_time_values_changed, 2266, ...
    'hfl_no_snf_time_values_changed', 1);
assert_scalar_equal(audit.hfl_snf_time_values_changed, 32, ...
    'hfl_snf_time_values_changed', 1);

summary = struct();
summary.result_file = result_file;
summary.checked_snapshot_count = prod(expected_size);
summary.fixed_group_limit = numel(data.EdgeSet);
summary.no_snf_group_cells_rebuilt = audit.hfl_no_snf_group_cells_rebuilt;
summary.no_snf_time_pair_cells_changed = audit.hfl_no_snf_time_pair_cells_changed;
summary.hfl_snf_time_values_changed = audit.hfl_snf_time_values_changed;
summary.passed = true;
fprintf('修正结果验证通过：%d 个快照，noSnF 错误时间单元 %d。\n', ...
    summary.checked_snapshot_count, summary.no_snf_time_pair_cells_changed);
end


function assert_scalar_equal(actual, expected, field_name, index)
%ASSERT_SCALAR_EQUAL 校验指定单元的标量是否完全一致。

if ~isequaln(actual, expected)
    error('validate_corrected_result:ScalarMismatch', ...
        '%s 第 %d 个单元不一致：实际 %g，期望 %g。', field_name, index, actual, expected);
end
end


function assert_matrix_equal(actual, expected, field_name)
%ASSERT_MATRIX_EQUAL 校验矩阵尺寸和值是否逐单元完全一致。

if ~isequaln(actual, expected)
    error('validate_corrected_result:MatrixMismatch', '%s 与独立复算结果不一致。', field_name);
end
end

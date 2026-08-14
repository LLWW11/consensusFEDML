function audit = repair_result_U_6fixedge_epoch200(input_file, output_file)
%REPAIR_RESULT_U_6FIXEDGE_EPOCH200 修复旧版六固定边缘实验结果中的字段与时间。
%   audit = REPAIR_RESULT_U_6FIXEDGE_EPOCH200() 读取文件名带扩展名前空格的
%   原始 MAT，保留原文件不变，并生成 result-U-6fixedge_epoch200_corrected.mat。
%   也可以通过两个输入参数指定原始文件和修正文件。

paths = postprocess_paths();
if nargin < 1 || isempty(input_file)
    input_file = fullfile(paths.topology_directory, ...
        'result-U-6fixedge_epoch200 .mat');
end
if nargin < 2 || isempty(output_file)
    output_file = fullfile(paths.output_directory, ...
        'result-U-6fixedge_epoch200_corrected.mat');
end

input_file = char(input_file);
output_file = char(output_file);
if ~isfile(input_file)
    error('repair_result:SourceNotFound', '找不到原始结果文件：%s', input_file);
end
if strcmp(input_file, output_file)
    error('repair_result:SameFile', '修正文件不能覆盖原始结果文件。');
end

data = load(input_file);
required_fields = {'group_num_HFLSnF', 'client_num_HFLSnF', 'max_layer_HFLSnF', ...
    'group_num_HFLnoSnF', 'client_num_HFLnoSnF', 'max_layer_HFLnoSnF', ...
    'client_num_FLnoSnF', 'max_layer_FLnoSnF', ...
    'client_num_FLSnF', 'max_layer_FLSnF', ...
    'total_util', 'epoch_num', 'EdgeSet', 'agg_time_per_clients'};
assert_required_fields(data, required_fields);

expected_size = [double(data.epoch_num), numel(data.total_util)];
assert_result_size(data.group_num_HFLSnF, expected_size, 'group_num_HFLSnF');
assert_result_size(data.group_num_HFLnoSnF, expected_size, 'group_num_HFLnoSnF');
assert_result_size(data.client_num_FLnoSnF, expected_size, 'client_num_FLnoSnF');
assert_result_size(data.client_num_FLSnF, expected_size, 'client_num_FLSnF');

% 先保存旧字段快照，修正文件中可直接审计修改前后的差异。
original_field_names = {'group_num_HFLnoSnF_fix', 'group_num_HFLnoSnF_los', ...
    'time_agg_HFLnoSnF_fix', 'time_agg_HFLnoSnF_los', ...
    'time_agg_HFLSnF_fix', 'time_agg_HFLSnF_los', ...
    'time_agg_FLnoSnF', 'time_agg_FLSnF', ...
    'aver_time_agg_HFLnoSnF_fix', 'aver_time_agg_HFLnoSnF_los', ...
    'aver_time_agg_HFLSnF_fix', 'aver_time_agg_HFLSnF_los'};
original_values = copy_existing_fields(data, original_field_names);

% 固定策略始终读取 cell 第 3 项，动态策略始终读取 cell 第 1 项。
hfl_snf = rebuild_hfl_fields(data.group_num_HFLSnF, data.client_num_HFLSnF, ...
    data.max_layer_HFLSnF, data.agg_time_per_clients, 'HFL-SnF');
hfl_no_snf = rebuild_hfl_fields(data.group_num_HFLnoSnF, data.client_num_HFLnoSnF, ...
    data.max_layer_HFLnoSnF, data.agg_time_per_clients, 'HFL-noSnF');

% 原始客户端数量字段应当与底层分组 cell 完全一致，否则停止修复以免掩盖新问题。
assert_matrix_equal(data.client_num_HFLSnF_fix, hfl_snf.client_fix, ...
    'client_num_HFLSnF_fix');
assert_matrix_equal(data.client_num_HFLSnF_los, hfl_snf.client_dynamic, ...
    'client_num_HFLSnF_los');
assert_matrix_equal(data.client_num_HFLnoSnF_fix, hfl_no_snf.client_fix, ...
    'client_num_HFLnoSnF_fix');
assert_matrix_equal(data.client_num_HFLnoSnF_los, hfl_no_snf.client_dynamic, ...
    'client_num_HFLnoSnF_los');

source_cnum_error = struct();
source_cnum_error.HFLSnF_fix = assert_summary_mean( ...
    data.cnum_HFLSnF_fix, hfl_snf.client_fix, 'cnum_HFLSnF_fix');
source_cnum_error.HFLSnF_dynamic = assert_summary_mean( ...
    data.cnum_HFLSnF_los, hfl_snf.client_dynamic, 'cnum_HFLSnF_los');
source_cnum_error.HFLnoSnF_fix = assert_summary_mean( ...
    data.cnum_HFLnoSnF_fix, hfl_no_snf.client_fix, 'cnum_HFLnoSnF_fix');
source_cnum_error.HFLnoSnF_dynamic = assert_summary_mean( ...
    data.cnum_HFLnoSnF_los, hfl_no_snf.client_dynamic, 'cnum_HFLnoSnF_los');
source_cnum_error.FLnoSnF = assert_summary_mean( ...
    data.cnum_FLnoSnF, data.client_num_FLnoSnF, 'cnum_FLnoSnF');
source_cnum_error.FLSnF = assert_summary_mean( ...
    data.cnum_FLSnF, data.client_num_FLSnF, 'cnum_FLSnF');

% 重建四种 HFL 的组数、参与人数、最大层和聚合时间。
data.group_num_HFLSnF_fix = hfl_snf.group_fix;
data.group_num_HFLSnF_los = hfl_snf.group_dynamic;
data.client_num_HFLSnF_fix = hfl_snf.client_fix;
data.client_num_HFLSnF_los = hfl_snf.client_dynamic;
data.max_layer_HFLSnF_fix = hfl_snf.layer_fix;
data.max_layer_HFLSnF_los = hfl_snf.layer_dynamic;
data.time_agg_HFLSnF_fix = hfl_snf.time_fix;
data.time_agg_HFLSnF_los = hfl_snf.time_dynamic;

data.group_num_HFLnoSnF_fix = hfl_no_snf.group_fix;
data.group_num_HFLnoSnF_los = hfl_no_snf.group_dynamic;
data.client_num_HFLnoSnF_fix = hfl_no_snf.client_fix;
data.client_num_HFLnoSnF_los = hfl_no_snf.client_dynamic;
data.max_layer_HFLnoSnF_fix = hfl_no_snf.layer_fix;
data.max_layer_HFLnoSnF_los = hfl_no_snf.layer_dynamic;
data.time_agg_HFLnoSnF_fix = hfl_no_snf.time_fix;
data.time_agg_HFLnoSnF_los = hfl_no_snf.time_dynamic;

% 两种普通 FL 的聚合时间仅由参与客户端数量决定：T_agg = 2N。
data.time_agg_FLnoSnF = data.client_num_FLnoSnF .* data.agg_time_per_clients;
data.time_agg_FLSnF = data.client_num_FLSnF .* data.agg_time_per_clients;

% 所有均值均从 200 次底层数据重新计算，不沿用旧汇总字段。
data.enum_HFLSnF_fix = mean(data.group_num_HFLSnF_fix, 1);
data.enum_HFLSnF_los = mean(data.group_num_HFLSnF_los, 1);
data.cnum_HFLSnF_fix = mean(data.client_num_HFLSnF_fix, 1);
data.cnum_HFLSnF_los = mean(data.client_num_HFLSnF_los, 1);
data.mlayer_HFLSnF_fix = mean(data.max_layer_HFLSnF_fix, 1);
data.mlayer_HFLSnF_los = mean(data.max_layer_HFLSnF_los, 1);
data.aver_time_agg_HFLSnF_fix = mean(data.time_agg_HFLSnF_fix, 1);
data.aver_time_agg_HFLSnF_los = mean(data.time_agg_HFLSnF_los, 1);

data.enum_HFLnoSnF_fix = mean(data.group_num_HFLnoSnF_fix, 1);
data.enum_HFLnoSnF_los = mean(data.group_num_HFLnoSnF_los, 1);
data.cnum_HFLnoSnF_fix = mean(data.client_num_HFLnoSnF_fix, 1);
data.cnum_HFLnoSnF_los = mean(data.client_num_HFLnoSnF_los, 1);
data.mlayer_HFLnoSnF_fix = mean(data.max_layer_HFLnoSnF_fix, 1);
data.mlayer_HFLnoSnF_los = mean(data.max_layer_HFLnoSnF_los, 1);
data.aver_time_agg_HFLnoSnF_fix = mean(data.time_agg_HFLnoSnF_fix, 1);
data.aver_time_agg_HFLnoSnF_los = mean(data.time_agg_HFLnoSnF_los, 1);

data.cnum_FLnoSnF = mean(data.client_num_FLnoSnF, 1);
data.mlayer_FLnoSnF = mean(data.max_layer_FLnoSnF, 1);
data.aver_time_agg_FLnoSnF = mean(data.time_agg_FLnoSnF, 1);
data.cnum_FLSnF = mean(data.client_num_FLSnF, 1);
data.mlayer_FLSnF = mean(data.max_layer_FLSnF, 1);
data.aver_time_agg_FLSnF = mean(data.time_agg_FLSnF, 1);

% 传输时间公式沿用旧实验：16 秒/层，并加 100 秒固定时间。
data.transport_time_per_layer = 16;
data.base_round_time = 100;
data.time_HFLSnF_fix = data.mlayer_HFLSnF_fix .* data.transport_time_per_layer + data.base_round_time;
data.time_HFLSnF_los = data.mlayer_HFLSnF_los .* data.transport_time_per_layer + data.base_round_time;
data.time_HFLnoSnF_fix = data.mlayer_HFLnoSnF_fix .* data.transport_time_per_layer + data.base_round_time;
data.time_HFLnoSnF_los = data.mlayer_HFLnoSnF_los .* data.transport_time_per_layer + data.base_round_time;
data.time_FLnoSnF = data.mlayer_FLnoSnF .* data.transport_time_per_layer + data.base_round_time;
data.time_FLSnF = data.mlayer_FLSnF .* data.transport_time_per_layer + data.base_round_time;

data.toall_time_HFLSnF_fix = data.time_HFLSnF_fix + data.aver_time_agg_HFLSnF_fix;
data.toall_time_HFLSnF_los = data.time_HFLSnF_los + data.aver_time_agg_HFLSnF_los;
data.toall_time_HFLnoSnF_fix = data.time_HFLnoSnF_fix + data.aver_time_agg_HFLnoSnF_fix;
data.toall_time_HFLnoSnF_los = data.time_HFLnoSnF_los + data.aver_time_agg_HFLnoSnF_los;
data.toall_time_FLnoSnF = data.time_FLnoSnF + data.aver_time_agg_FLnoSnF;
data.toall_time_FLSnF = data.time_FLSnF + data.aver_time_agg_FLSnF;

% 固定边缘策略不能生成超过六个固定边缘节点的有效组。
if any(data.group_num_HFLSnF_fix(:) > numel(data.EdgeSet)) || ...
        any(data.group_num_HFLnoSnF_fix(:) > numel(data.EdgeSet))
    error('repair_result:TooManyFixedGroups', '固定边缘组数超过 EdgeSet 大小。');
end

audit = build_repair_audit(original_values, data, source_cnum_error, expected_size);
data.repair_audit = audit;
data.repair_original_values = original_values;
data.repair_schema_version = '2.0-corrected';
data.schema_version = data.repair_schema_version;
data.repair_source_file = input_file;
data.repair_created_at = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
data.repair_note = ['从原始策略 cell 重新提取 dynamic(第1项) 与 fixed(第3项)，' ...
    '并独立重算四种 HFL 及两种 FL 的聚合时间、均值和总时间。原始 MAT 未被覆盖。'];
data.historical_mapping_limitations = ['原文件仅保存 HFL-SnF 映射；FL-noSnF、FL-SnF、' ...
    'HFL-noSnF-dynamic 和 HFL-noSnF-fixed 的历史参与客户端 ID 无法从人数反推，修正文件不伪造编号。'];
data.mapping_availability = struct( ...
    'HFL_SnF_dynamic_and_fixed', isfield(data, 'actual_c2e_map_HFLSnF'), ...
    'HFL_noSnF_dynamic_and_fixed', false, ...
    'FL_noSnF', false, ...
    'FL_SnF', false);
data.unrecoverable_participant_id_fields = {'actual_c2e_map_HFLnoSnF', ...
    'c2cmap_FLnoSnF_all', 'c2cmap_FLSnF_all'};
data.repair_original_filename_field = get_optional_field(data, 'filename', '');
[~, corrected_name, corrected_extension] = fileparts(output_file);
data.filename = [corrected_name, corrected_extension];

save(output_file, '-struct', 'data');
fprintf('修正文件已生成：%s\n', output_file);
fprintf('重建 noSnF 组数单元：%d；重算 noSnF 时间值：%d；重算 SnF 时间值：%d。\n', ...
    audit.hfl_no_snf_group_cells_rebuilt, audit.hfl_no_snf_time_values_changed, ...
    audit.hfl_snf_time_values_changed);
end


function rebuilt = rebuild_hfl_fields(group_cells, client_cells, layer_cells, per_model_time, context)
%REBUILD_HFL_FIELDS 从 HFL 底层策略 cell 重建固定与动态结果矩阵。

if ~isequal(size(group_cells), size(client_cells), size(layer_cells))
    error('repair_result:RawCellSizeMismatch', '%s 的组数、客户端数和层数 cell 尺寸不一致。', context);
end

result_size = size(group_cells);
rebuilt.group_fix = zeros(result_size);
rebuilt.group_dynamic = zeros(result_size);
rebuilt.client_fix = zeros(result_size);
rebuilt.client_dynamic = zeros(result_size);
rebuilt.layer_fix = zeros(result_size);
rebuilt.layer_dynamic = zeros(result_size);
rebuilt.time_fix = zeros(result_size);
rebuilt.time_dynamic = zeros(result_size);

for index = 1:numel(group_cells)
    group_dynamic = extract_policy_scalar(group_cells{index}, 1, context, 'group');
    group_fix = extract_policy_scalar(group_cells{index}, 3, context, 'group');
    client_dynamic = extract_policy_vector(client_cells{index}, 1, context, 'client');
    client_fix = extract_policy_vector(client_cells{index}, 3, context, 'client');
    layer_dynamic = extract_policy_scalar(layer_cells{index}, 1, context, 'layer');
    layer_fix = extract_policy_scalar(layer_cells{index}, 3, context, 'layer');

    if group_dynamic ~= nnz(client_dynamic) || group_fix ~= nnz(client_fix)
        error('repair_result:GroupClientMismatch', ...
            '%s 第 %d 个快照的有效组数与非空客户端组数不一致。', context, index);
    end

    rebuilt.group_dynamic(index) = group_dynamic;
    rebuilt.group_fix(index) = group_fix;
    rebuilt.client_dynamic(index) = sum(client_dynamic(:));
    rebuilt.client_fix(index) = sum(client_fix(:));
    rebuilt.layer_dynamic(index) = layer_dynamic;
    rebuilt.layer_fix(index) = layer_fix;
    rebuilt.time_dynamic(index) = calc_hfl_agg_time( ...
        client_dynamic, group_dynamic, per_model_time);
    rebuilt.time_fix(index) = calc_hfl_agg_time(client_fix, group_fix, per_model_time);
end
end


function value = extract_policy_scalar(policy_cell, policy_index, context, value_name)
%EXTRACT_POLICY_SCALAR 提取策略 cell 中指定位置的有限非负标量。

value = extract_policy_vector(policy_cell, policy_index, context, value_name);
if ~isscalar(value)
    error('repair_result:ExpectedScalar', '%s 的 %s 策略 %d 不是标量。', ...
        context, value_name, policy_index);
end
end


function value = extract_policy_vector(policy_cell, policy_index, context, value_name)
%EXTRACT_POLICY_VECTOR 提取策略 cell 中指定位置的有限非负数值向量。

if ~iscell(policy_cell) || numel(policy_cell) < policy_index
    error('repair_result:MissingPolicy', '%s 的 %s 缺少策略 %d。', ...
        context, value_name, policy_index);
end
value = policy_cell{policy_index};
if ~isnumeric(value) || any(~isfinite(value(:))) || any(value(:) < 0)
    error('repair_result:InvalidPolicyValue', '%s 的 %s 策略 %d 包含非法值。', ...
        context, value_name, policy_index);
end
end


function assert_required_fields(data, required_fields)
%ASSERT_REQUIRED_FIELDS 校验修复所需的变量是否都存在于原始 MAT 中。

missing_fields = required_fields(~isfield(data, required_fields));
if ~isempty(missing_fields)
    error('repair_result:MissingFields', '原始 MAT 缺少字段：%s', strjoin(missing_fields, ', '));
end
end


function assert_result_size(value, expected_size, field_name)
%ASSERT_RESULT_SIZE 校验底层结果矩阵是否符合 epoch×util 尺寸。

if ~isequal(size(value), expected_size)
    error('repair_result:UnexpectedSize', '%s 尺寸为 %s，期望 %s。', ...
        field_name, mat2str(size(value)), mat2str(expected_size));
end
end


function assert_matrix_equal(actual, expected, field_name)
%ASSERT_MATRIX_EQUAL 校验两个底层结果矩阵逐单元完全一致。

if ~isequaln(actual, expected)
    mismatch_count = nnz(actual ~= expected);
    error('repair_result:MatrixMismatch', '%s 有 %d 个单元与底层 cell 不一致。', ...
        field_name, mismatch_count);
end
end


function max_error = assert_summary_mean(source_summary, raw_matrix, field_name)
%ASSERT_SUMMARY_MEAN 校验旧汇总值是否等于底层轮次数据的列均值。

expected_summary = mean(raw_matrix, 1);
max_error = max(abs(source_summary(:) - expected_summary(:)));
if max_error > 1e-12
    error('repair_result:SummaryMismatch', '%s 与底层轮次均值不一致，最大误差 %.3g。', ...
        field_name, max_error);
end
end


function copied = copy_existing_fields(data, field_names)
%COPY_EXISTING_FIELDS 复制原始 MAT 中存在的指定字段用于修复审计。

copied = struct();
for index = 1:numel(field_names)
    field_name = field_names{index};
    if isfield(data, field_name)
        copied.(field_name) = data.(field_name);
    end
end
end


function value = get_optional_field(data, field_name, default_value)
%GET_OPTIONAL_FIELD 读取可选字段，不存在时返回给定默认值。

if isfield(data, field_name)
    value = data.(field_name);
else
    value = default_value;
end
end


function audit = build_repair_audit(original_values, corrected, source_cnum_error, expected_size)
%BUILD_REPAIR_AUDIT 汇总修复覆盖范围、实际变更数量和一致性误差。

audit = struct();
audit.epoch_util_size = expected_size;
audit.hfl_no_snf_group_cells_rebuilt = prod(expected_size);
audit.hfl_no_snf_group_pair_cells_changed = count_pair_changes( ...
    original_values.group_num_HFLnoSnF_fix, corrected.group_num_HFLnoSnF_fix, ...
    original_values.group_num_HFLnoSnF_los, corrected.group_num_HFLnoSnF_los);
audit.hfl_no_snf_group_values_changed = ...
    nnz(original_values.group_num_HFLnoSnF_fix ~= corrected.group_num_HFLnoSnF_fix) + ...
    nnz(original_values.group_num_HFLnoSnF_los ~= corrected.group_num_HFLnoSnF_los);
audit.hfl_no_snf_time_pair_cells_changed = count_pair_changes( ...
    original_values.time_agg_HFLnoSnF_fix, corrected.time_agg_HFLnoSnF_fix, ...
    original_values.time_agg_HFLnoSnF_los, corrected.time_agg_HFLnoSnF_los);
audit.hfl_no_snf_time_values_changed = ...
    nnz(original_values.time_agg_HFLnoSnF_fix ~= corrected.time_agg_HFLnoSnF_fix) + ...
    nnz(original_values.time_agg_HFLnoSnF_los ~= corrected.time_agg_HFLnoSnF_los);
audit.hfl_snf_time_pair_cells_changed = count_pair_changes( ...
    original_values.time_agg_HFLSnF_fix, corrected.time_agg_HFLSnF_fix, ...
    original_values.time_agg_HFLSnF_los, corrected.time_agg_HFLSnF_los);
audit.hfl_snf_time_values_changed = ...
    nnz(original_values.time_agg_HFLSnF_fix ~= corrected.time_agg_HFLSnF_fix) + ...
    nnz(original_values.time_agg_HFLSnF_los ~= corrected.time_agg_HFLSnF_los);
audit.source_cnum_max_abs_error = source_cnum_error;
audit.fixed_group_limit = numel(corrected.EdgeSet);
audit.fixed_group_limit_passed = all(corrected.group_num_HFLSnF_fix(:) <= numel(corrected.EdgeSet)) && ...
    all(corrected.group_num_HFLnoSnF_fix(:) <= numel(corrected.EdgeSet));
end


function changed_count = count_pair_changes(original_fix, corrected_fix, original_dynamic, corrected_dynamic)
%COUNT_PAIR_CHANGES 统计固定或动态任一字段发生变化的 epoch×util 单元数。

changed_count = nnz((original_fix ~= corrected_fix) | ...
    (original_dynamic ~= corrected_dynamic));
end

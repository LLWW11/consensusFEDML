function audit = control_client_variance(input_file, output_file, varAlpha)
%CONTROL_CLIENT_VARIANCE 在保持均值不变的前提下控制客户端数量方差。
%   audit = CONTROL_CLIENT_VARIANCE(input_file, output_file, varAlpha)
%   对六种方法的客户端数量矩阵逐列执行整数方差收缩。varAlpha 表示
%   “目标方差/原方差”，取值范围为 [0,1]。原始字段保持不变，处理后的
%   字段使用 _varctrl 后缀保存，并同时写入完整审计信息。

script_directory = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(input_file)
    input_file = fullfile(script_directory, 'result-U-6fixedge_epoch200.mat');
end
if nargin < 3 || isempty(varAlpha)
    varAlpha = 0.5;
end
if nargin < 2 || isempty(output_file)
    alpha_token = format_alpha_token(varAlpha);
    output_file = fullfile(script_directory, ...
        ['result-U-6fixedge_epoch200_varAlpha_', alpha_token, '.mat']);
end

input_file = char(input_file);
output_file = char(output_file);
validate_inputs(input_file, output_file, varAlpha);

data = load(input_file);
client_fields = { ...
    'client_num_HFLSnF_fix', ...
    'client_num_HFLSnF_los', ...
    'client_num_HFLnoSnF_fix', ...
    'client_num_HFLnoSnF_los', ...
    'client_num_FLnoSnF', ...
    'client_num_FLSnF'};
summary_fields = { ...
    'cnum_HFLSnF_fix', ...
    'cnum_HFLSnF_los', ...
    'cnum_HFLnoSnF_fix', ...
    'cnum_HFLnoSnF_los', ...
    'cnum_FLnoSnF', ...
    'cnum_FLSnF'};

assert_required_fields(data, client_fields);
max_client_count = infer_max_client_count(data, client_fields);

audit = struct();
audit.schema_version = '1.0';
audit.varAlpha = varAlpha;
audit.source_file = input_file;
audit.output_file = output_file;
audit.max_client_count = max_client_count;
audit.client_fields = client_fields;
audit.created_at = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
audit.methods = struct();

for field_index = 1:numel(client_fields)
    field_name = client_fields{field_index};
    summary_name = summary_fields{field_index};
    original = double(data.(field_name));
    validate_client_matrix(original, max_client_count, field_name);

    [controlled, method_audit] = control_matrix( ...
        original, varAlpha, max_client_count, field_name);
    controlled_field = [field_name, '_varctrl'];
    controlled_summary = [summary_name, '_varctrl'];
    data.(controlled_field) = controlled;
    data.(controlled_summary) = mean(controlled, 1);
    audit.methods.(field_name) = method_audit;
end

data.variance_control_schema_version = '1.0';
data.variance_control_varAlpha = varAlpha;
data.variance_control_source_file = input_file;
data.variance_control_created_at = audit.created_at;
data.variance_control_note = [ ...
    '原始客户端数量、分组和映射字段未修改；_varctrl 字段仅用于' ...
    '明确标记的方差控制分析，不代表重新运行得到的原始实验样本。'];
data.variance_control_audit = audit;

save(output_file, '-struct', 'data');
fprintf('客户端数量方差控制文件已生成：%s\n', output_file);
fprintf('varAlpha=%.6g，目标方差为原方差的 %.2f%%。\n', ...
    varAlpha, varAlpha * 100);
end


function validate_inputs(input_file, output_file, varAlpha)
%VALIDATE_INPUTS 校验输入文件、输出文件和方差比例参数。

if ~isfile(input_file)
    error('control_client_variance:SourceNotFound', ...
        '找不到输入结果文件：%s', input_file);
end
if strcmpi(input_file, output_file)
    error('control_client_variance:SameFile', ...
        '输出文件不能覆盖原始结果文件。');
end
if ~isnumeric(varAlpha) || ~isscalar(varAlpha) || ~isfinite(varAlpha) || ...
        varAlpha < 0 || varAlpha > 1
    error('control_client_variance:InvalidVarAlpha', ...
        'varAlpha 必须是 [0,1] 范围内的有限数值标量。');
end
end


function token = format_alpha_token(varAlpha)
%FORMAT_ALPHA_TOKEN 将 varAlpha 转换为适合文件名的字符串。

token = sprintf('%.6g', varAlpha);
token = strrep(token, '.', 'p');
token = strrep(token, '-', 'm');
end


function assert_required_fields(data, required_fields)
%ASSERT_REQUIRED_FIELDS 校验 MAT 文件是否包含所有客户端数量字段。

for field_index = 1:numel(required_fields)
    field_name = required_fields{field_index};
    if ~isfield(data, field_name)
        error('control_client_variance:MissingField', ...
            '输入结果缺少字段：%s', field_name);
    end
end
end


function max_client_count = infer_max_client_count(data, client_fields)
%INFER_MAX_CLIENT_COUNT 从实验配置推断客户端数量上限。

if isfield(data, 'num_of_nodes') && isfield(data, 'Cloud')
    max_client_count = double(data.num_of_nodes) - 1;
else
    max_client_count = 0;
    for field_index = 1:numel(client_fields)
        max_client_count = max(max_client_count, ...
            max(double(data.(client_fields{field_index})(:))));
    end
end
if ~isscalar(max_client_count) || max_client_count < 0 || ...
        max_client_count ~= floor(max_client_count)
    error('control_client_variance:InvalidClientLimit', ...
        '无法从结果文件推断合法的客户端数量上限。');
end
end


function validate_client_matrix(values, max_client_count, field_name)
%VALIDATE_CLIENT_MATRIX 校验客户端数量矩阵的尺寸、范围和整数性。

if ~isnumeric(values) || ~ismatrix(values) || isempty(values)
    error('control_client_variance:InvalidMatrix', ...
        '%s 必须是非空二维数值矩阵。', field_name);
end
if any(~isfinite(values(:))) || any(values(:) < 0) || ...
        any(values(:) > max_client_count)
    error('control_client_variance:OutOfRange', ...
        '%s 包含超出 [0,%d] 的客户端数量。', field_name, max_client_count);
end
if any(values(:) ~= round(values(:)))
    error('control_client_variance:NonIntegerClientCount', ...
        '%s 包含非整数客户端数量。', field_name);
end
end


function [controlled, method_audit] = control_matrix( ...
        original, varAlpha, max_client_count, field_name)
%CONTROL_MATRIX 对一个 epoch×util 客户端数量矩阵逐列控制方差。

[epoch_count, util_count] = size(original);
controlled = zeros(epoch_count, util_count);
method_audit = initialize_method_audit(util_count, field_name);

for util_index = 1:util_count
    original_column = original(:, util_index);
    [controlled_column, column_audit] = shrink_integer_column( ...
        original_column, varAlpha, max_client_count);
    controlled(:, util_index) = controlled_column;

    audit_fields = fieldnames(column_audit);
    for audit_index = 1:numel(audit_fields)
        audit_name = audit_fields{audit_index};
        method_audit.(audit_name)(util_index) = column_audit.(audit_name);
    end
end
end


function method_audit = initialize_method_audit(util_count, field_name)
%INITIALIZE_METHOD_AUDIT 初始化单个方法的逐利用率审计结构。

method_audit = struct();
method_audit.field_name = field_name;
numeric_fields = { ...
    'original_mean', 'controlled_mean', 'mean_error', ...
    'original_variance', 'requested_variance', 'effective_target_variance', ...
    'minimum_integer_variance', 'achieved_variance', 'achieved_alpha', ...
    'selected_shrink_factor', 'minimum_value', 'maximum_value'};
logical_fields = {'requested_target_above_integer_floor', ...
    'target_reached_within_tolerance', 'sum_preserved', ...
    'all_integer', 'all_in_range'};

for field_index = 1:numel(numeric_fields)
    method_audit.(numeric_fields{field_index}) = zeros(1, util_count);
end
for field_index = 1:numel(logical_fields)
    method_audit.(logical_fields{field_index}) = false(1, util_count);
end
end


function [controlled, column_audit] = shrink_integer_column( ...
        original, varAlpha, max_client_count)
%SHRINK_INTEGER_COLUMN 搜索最接近目标方差的整数收缩结果。

original = double(original(:));
epoch_count = numel(original);
original_sum = sum(original);
original_mean = mean(original);
original_variance = var(original, 0);

minimum_column = minimum_variance_integer_column( ...
    original_sum, epoch_count, original);
minimum_variance = var(minimum_column, 0);
requested_variance = varAlpha * original_variance;
effective_target = max(requested_variance, minimum_variance);

if varAlpha == 1 || original_variance == 0
    controlled = original;
    selected_factor = 1;
else
    [controlled, selected_factor] = search_shrink_factor( ...
        original, original_sum, effective_target, max_client_count);
end

achieved_variance = var(controlled, 0);
if original_variance == 0
    achieved_alpha = 0;
else
    achieved_alpha = achieved_variance / original_variance;
end
integer_floor_tolerance = 1e-12;
target_tolerance = max(1e-12, 0.05 * max(original_variance, effective_target));

column_audit = struct();
column_audit.original_mean = original_mean;
column_audit.controlled_mean = mean(controlled);
column_audit.mean_error = mean(controlled) - original_mean;
column_audit.original_variance = original_variance;
column_audit.requested_variance = requested_variance;
column_audit.effective_target_variance = effective_target;
column_audit.minimum_integer_variance = minimum_variance;
column_audit.achieved_variance = achieved_variance;
column_audit.achieved_alpha = achieved_alpha;
column_audit.selected_shrink_factor = selected_factor;
column_audit.minimum_value = min(controlled);
column_audit.maximum_value = max(controlled);
column_audit.requested_target_above_integer_floor = ...
    requested_variance >= minimum_variance - integer_floor_tolerance;
column_audit.target_reached_within_tolerance = ...
    abs(achieved_variance - effective_target) <= target_tolerance;
column_audit.sum_preserved = sum(controlled) == original_sum;
column_audit.all_integer = all(controlled == round(controlled));
column_audit.all_in_range = all(controlled >= 0 & controlled <= max_client_count);
end


function minimum_column = minimum_variance_integer_column( ...
        total_sum, epoch_count, original)
%MINIMUM_VARIANCE_INTEGER_COLUMN 构造固定整数总和下的最小方差列。

lower_value = floor(total_sum / epoch_count);
upper_count = round(total_sum - lower_value * epoch_count);
minimum_column = lower_value * ones(epoch_count, 1);

% 将较大的取值分配给原始值较大的位置，尽量保留样本次序关系。
if upper_count > 0
    [~, order] = sort(original, 'descend');
    minimum_column(order(1:upper_count)) = lower_value + 1;
end
end


function [best_column, best_factor] = search_shrink_factor( ...
        original, original_sum, target_variance, max_client_count)
%SEARCH_SHRINK_FACTOR 使用二分搜索寻找最接近目标方差的收缩系数。

lower_factor = 0;
upper_factor = 1;
[best_column, best_variance] = evaluate_factor( ...
    original, original_sum, lower_factor, max_client_count);
best_factor = lower_factor;
best_error = abs(best_variance - target_variance);

[candidate, candidate_variance] = evaluate_factor( ...
    original, original_sum, upper_factor, max_client_count);
[best_column, best_factor, best_error] = update_best_candidate( ...
    best_column, best_factor, best_error, candidate, upper_factor, ...
    candidate_variance, target_variance);

for iteration = 1:64
    middle_factor = (lower_factor + upper_factor) / 2;
    [candidate, candidate_variance] = evaluate_factor( ...
        original, original_sum, middle_factor, max_client_count);
    [best_column, best_factor, best_error] = update_best_candidate( ...
        best_column, best_factor, best_error, candidate, middle_factor, ...
        candidate_variance, target_variance);

    if candidate_variance < target_variance
        lower_factor = middle_factor;
    else
        upper_factor = middle_factor;
    end
end
end


function [candidate, candidate_variance] = evaluate_factor( ...
        original, original_sum, shrink_factor, max_client_count)
%EVALUATE_FACTOR 生成指定收缩系数对应的平衡整数结果和方差。

original_mean = mean(original);
continuous = original_mean + shrink_factor .* (original - original_mean);
continuous = min(max(continuous, 0), max_client_count);
candidate = balanced_integer_round(continuous, original_sum, max_client_count);
candidate_variance = var(candidate, 0);
end


function rounded = balanced_integer_round(values, target_sum, max_client_count)
%BALANCED_INTEGER_ROUND 在保持总和的条件下执行平衡整数舍入。

rounded = floor(values);
remaining = round(target_sum - sum(rounded));
fractional = values - rounded;

if remaining > 0
    eligible = find(rounded < max_client_count);
    [~, local_order] = sort(fractional(eligible), 'descend');
    selected = eligible(local_order(1:remaining));
    rounded(selected) = rounded(selected) + 1;
elseif remaining < 0
    eligible = find(rounded > 0);
    [~, local_order] = sort(fractional(eligible), 'ascend');
    selected = eligible(local_order(1:(-remaining)));
    rounded(selected) = rounded(selected) - 1;
end

if sum(rounded) ~= target_sum
    error('control_client_variance:SumNotPreserved', ...
        '平衡整数舍入未能保持客户端数量总和。');
end
end


function [best_column, best_factor, best_error] = update_best_candidate( ...
        best_column, best_factor, best_error, candidate, candidate_factor, ...
        candidate_variance, target_variance)
%UPDATE_BEST_CANDIDATE 更新当前最接近目标方差的候选结果。

candidate_error = abs(candidate_variance - target_variance);
if candidate_error < best_error
    best_column = candidate;
    best_factor = candidate_factor;
    best_error = candidate_error;
end
end

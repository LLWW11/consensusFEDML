function summary = build_epoch_variance_summary(result_matrices, total_util)
%BUILD_EPOCH_VARIANCE_SUMMARY 汇总各利用率下跨 epoch 的结果方差。
%   summary = BUILD_EPOCH_VARIANCE_SUMMARY(result_matrices, total_util)
%   接收字段值均为 epoch×util 数值矩阵的结构体，返回每个字段的均值、
%   标准差、样本方差和变异系数。该函数只统计，不修改输入结果。

if ~isstruct(result_matrices) || ~isscalar(result_matrices)
    error('build_epoch_variance_summary:InvalidInput', ...
        'result_matrices 必须是标量结构体。');
end
if ~isnumeric(total_util) || ~isvector(total_util) || isempty(total_util) || ...
        any(~isfinite(total_util(:)))
    error('build_epoch_variance_summary:InvalidUtilization', ...
        'total_util 必须是非空有限数值向量。');
end

util_count = numel(total_util);
field_names = fieldnames(result_matrices);
summary = struct();
summary.schema_version = '1.0';
summary.total_util = reshape(double(total_util), 1, []);
summary.epoch_count = [];
summary.metrics = struct();

for field_index = 1:numel(field_names)
    field_name = field_names{field_index};
    values = double(result_matrices.(field_name));
    validate_result_matrix(values, util_count, field_name);
    if isempty(summary.epoch_count)
        summary.epoch_count = size(values, 1);
    elseif size(values, 1) ~= summary.epoch_count
        error('build_epoch_variance_summary:EpochCountMismatch', ...
            '%s 的 epoch 数与其他结果矩阵不一致。', field_name);
    end

    metric = struct();
    metric.mean = mean(values, 1);
    metric.standard_deviation = std(values, 0, 1);
    metric.variance = var(values, 0, 1);
    metric.coefficient_of_variation = zeros(1, util_count);
    nonzero_mean = metric.mean ~= 0;
    metric.coefficient_of_variation(nonzero_mean) = ...
        metric.standard_deviation(nonzero_mean) ./ abs(metric.mean(nonzero_mean));
    metric.coefficient_of_variation(~nonzero_mean & ...
        metric.standard_deviation ~= 0) = Inf;
    summary.metrics.(field_name) = metric;
end
end


function validate_result_matrix(values, util_count, field_name)
%VALIDATE_RESULT_MATRIX 校验一个待统计的 epoch×util 数值矩阵。

if ~isnumeric(values) || ~ismatrix(values) || isempty(values) || ...
        size(values, 2) ~= util_count
    error('build_epoch_variance_summary:InvalidMatrix', ...
        '%s 必须是列数与 total_util 一致的非空二维数值矩阵。', field_name);
end
if any(~isfinite(values(:)))
    error('build_epoch_variance_summary:NonFiniteValue', ...
        '%s 包含非有限数值。', field_name);
end
end

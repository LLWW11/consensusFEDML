function aggregation_time = calc_hfl_agg_time(client_counts, group_count, per_model_time)
%CALC_HFL_AGG_TIME 计算一次层次聚合所需时间。
%   aggregation_time = CALC_HFL_AGG_TIME(client_counts, group_count,
%   per_model_time) 按“最大组内客户端数 + 有效边缘组数”计算聚合时间。
%   空客户端向量按最大组内客户端数为 0 处理。

if nargin ~= 3
    error('calc_hfl_agg_time:InvalidInputCount', ...
        '需要 client_counts、group_count 和 per_model_time 三个输入。');
end
if ~isnumeric(client_counts) || ~isnumeric(group_count) || ~isnumeric(per_model_time)
    error('calc_hfl_agg_time:InvalidInputType', '所有输入必须为数值。');
end
if ~isscalar(group_count) || group_count < 0 || group_count ~= floor(group_count)
    error('calc_hfl_agg_time:InvalidGroupCount', 'group_count 必须为非负整数。');
end
if ~isscalar(per_model_time) || per_model_time < 0
    error('calc_hfl_agg_time:InvalidPerModelTime', 'per_model_time 必须为非负标量。');
end
if any(~isfinite(client_counts(:))) || any(client_counts(:) < 0)
    error('calc_hfl_agg_time:InvalidClientCounts', 'client_counts 必须为有限非负数。');
end

if isempty(client_counts)
    max_client_count = 0;
else
    max_client_count = max(client_counts(:));
end

aggregation_time = per_model_time * (max_client_count + group_count);
end

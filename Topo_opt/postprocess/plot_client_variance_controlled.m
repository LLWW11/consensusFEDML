function figure_handle = plot_client_variance_controlled(result_file, data_mode)
%PLOT_CLIENT_VARIANCE_CONTROLLED 绘制六种方法的客户端数量均值和标准差。
%   figure_handle = PLOT_CLIENT_VARIANCE_CONTROLLED(result_file, data_mode)
%   data_mode 可取 'original' 或 'varctrl'。'varctrl' 使用明确标记的
%   方差控制字段，'original' 使用原始客户端数量字段。

paths = postprocess_paths();
if nargin < 1 || isempty(result_file)
    result_file = fullfile(paths.output_directory, ...
        'result-U-6fixedge_epoch200_varAlpha_0p5.mat');
end
if nargin < 2 || isempty(data_mode)
    data_mode = 'varctrl';
end

result_file = char(result_file);
data_mode = validatestring(data_mode, {'original', 'varctrl'}, ...
    mfilename, 'data_mode');
if ~isfile(result_file)
    error('plot_client_variance_controlled:FileNotFound', ...
        '找不到结果文件：%s', result_file);
end

data = load(result_file);
if ~isfield(data, 'total_util')
    error('plot_client_variance_controlled:MissingUtilization', ...
        '结果文件缺少 total_util 字段。');
end

base_fields = { ...
    'client_num_HFLSnF_fix', ...
    'client_num_HFLSnF_los', ...
    'client_num_HFLnoSnF_fix', ...
    'client_num_HFLnoSnF_los', ...
    'client_num_FLSnF', ...
    'client_num_FLnoSnF'};
legend_labels = { ...
    'HFL-SnF-fixed', ...
    'HFL-SnF-dynamic', ...
    'HFL-noSnF-fixed', ...
    'HFL-noSnF-dynamic', ...
    'FL-SnF', ...
    'FL-noSnF'};

figure_handle = figure('Name', '客户端数量方差对比');
axes_handle = axes(figure_handle);
hold(axes_handle, 'on');

for field_index = 1:numel(base_fields)
    field_name = select_field_name(base_fields{field_index}, data_mode);
    if ~isfield(data, field_name)
        error('plot_client_variance_controlled:MissingField', ...
            '结果文件缺少绘图字段：%s', field_name);
    end
    values = double(data.(field_name));
    errorbar(axes_handle, data.total_util, mean(values, 1), std(values, 0, 1), ...
        '-o', 'LineWidth', 1.2, 'MarkerSize', 5, ...
        'DisplayName', legend_labels{field_index});
end

hold(axes_handle, 'off');
grid(axes_handle, 'on');
xlabel(axes_handle, '利用率');
ylabel(axes_handle, '客户端数量（均值 ± 标准差）');
if strcmp(data_mode, 'varctrl') && isfield(data, 'variance_control_varAlpha')
    title(axes_handle, sprintf('客户端数量方差控制，varAlpha=%.4g', ...
        data.variance_control_varAlpha));
else
    title(axes_handle, '原始客户端数量方差');
end
legend(axes_handle, 'Location', 'best');
end


function field_name = select_field_name(base_field, data_mode)
%SELECT_FIELD_NAME 根据绘图模式选择原始或方差控制字段。

if strcmp(data_mode, 'varctrl')
    field_name = [base_field, '_varctrl'];
else
    field_name = base_field;
end
end

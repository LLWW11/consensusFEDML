function summary = test_control_client_variance()
%TEST_CONTROL_CLIENT_VARIANCE 验证 varAlpha 客户端数量方差控制功能。
%   测试原字段不变、列总和不变、整数范围合法、varAlpha=1 恒等，以及
%   varAlpha 下降时各列实际方差不增加。

paths = postprocess_paths();
input_file = fullfile(paths.topology_directory, ...
    'result-U-6fixedge_epoch200.mat');
if ~isfile(input_file)
    error('test_control_client_variance:SourceNotFound', ...
        '测试所需原始结果不存在：%s', input_file);
end

client_fields = { ...
    'client_num_HFLSnF_fix', ...
    'client_num_HFLSnF_los', ...
    'client_num_HFLnoSnF_fix', ...
    'client_num_HFLnoSnF_los', ...
    'client_num_FLnoSnF', ...
    'client_num_FLSnF'};
alpha_values = [1, 0.75, 0.5, 0.25, 0];
output_files = cell(size(alpha_values));
for alpha_index = 1:numel(alpha_values)
    output_files{alpha_index} = [tempname, '.mat'];
end
cleanup_guard = onCleanup(@() cleanup_output_files(output_files));
source_info_before = dir(input_file);
source_data = load(input_file, client_fields{:});
previous_variances = [];
first_matrix = source_data.(client_fields{1});
current_variance_table = zeros(numel(client_fields), size(first_matrix, 2));

for alpha_index = 1:numel(alpha_values)
    varAlpha = alpha_values(alpha_index);
    audit = control_client_variance( ...
        input_file, output_files{alpha_index}, varAlpha);
    result = load(output_files{alpha_index});

    for field_index = 1:numel(client_fields)
        field_name = client_fields{field_index};
        controlled_name = [field_name, '_varctrl'];
        original = double(source_data.(field_name));
        controlled = double(result.(controlled_name));

        assert(isequal(result.(field_name), source_data.(field_name)), ...
            '后处理文件中的原始客户端字段发生变化：%s', field_name);
        assert(isequal(sum(controlled, 1), sum(original, 1)), ...
            '客户端数量列总和未保持：%s', field_name);
        assert(all(controlled(:) == round(controlled(:))), ...
            '方差控制结果包含非整数：%s', field_name);
        assert(all(controlled(:) >= 0 & controlled(:) <= 37), ...
            '方差控制结果超出 [0,37]：%s', field_name);

        current_variances = var(controlled, 0, 1);
        if varAlpha == 1
            assert(isequal(controlled, original), ...
                'varAlpha=1 时结果未与原矩阵完全一致：%s', field_name);
        end
        if ~isempty(previous_variances)
            assert(all(current_variances <= ...
                previous_variances(field_index, :) + 1e-12), ...
                'varAlpha 下降时方差出现增加：%s', field_name);
        end

        current_variance_table(field_index, :) = current_variances; %#ok<AGROW>
        method_audit = audit.methods.(field_name);
        assert(all(method_audit.sum_preserved), ...
            '审计发现列总和未保持：%s', field_name);
        assert(all(method_audit.all_integer & method_audit.all_in_range), ...
            '审计发现整数性或范围约束失败：%s', field_name);
    end
    previous_variances = current_variance_table;
end

source_info_after = dir(input_file);
assert(source_info_before.bytes == source_info_after.bytes && ...
    source_info_before.datenum == source_info_after.datenum, ...
    '原始 MAT 文件在测试过程中发生变化。');

summary = struct();
summary.schema_version = '1.0-test';
summary.alpha_values = alpha_values;
summary.client_fields = client_fields;
summary.original_file_unchanged = true;
summary.passed = true;
fprintf('varAlpha 方差控制测试通过：%d 个参数值，%d 种方法。\n', ...
    numel(alpha_values), numel(client_fields));
clear cleanup_guard;
end


function cleanup_output_files(output_files)
%CLEANUP_OUTPUT_FILES 删除测试过程中生成的临时 MAT 文件。

for file_index = 1:numel(output_files)
    output_file = output_files{file_index};
    if ~isempty(output_file) && isfile(output_file)
        delete(output_file);
    end
end
end

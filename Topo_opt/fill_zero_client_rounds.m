function audit = fill_zero_client_rounds( ...
        input_file, output_file, coverage_mode, coverage_horizon)
%FILL_ZERO_CLIENT_ROUNDS 用相邻有效轮次替换客户端数量为零的方法快照。
%   audit = FILL_ZERO_CLIENT_ROUNDS(input_file, output_file,
%   coverage_mode, coverage_horizon) 读取指定 MAT
%   文件，分别检查六种方法在每个利用率下的客户端数量。如果某轮人数为 0，
%   则用该方法、同一利用率下已经修复的上一轮整套数据覆盖当前轮。
%
%   第一轮没有上一轮可用；若第一轮为 0，本函数会使用同列首个非零轮次
%   回填第一轮，后续连续零值再沿用修复后的上一轮。如果某方法某个利用率整列
%   都为 0，则停止并报错，避免伪造无法恢复的数据。
%
%   本函数不会覆盖输入文件。省略 output_file 时，输出文件名自动增加
%   “_zeroFilled”后缀。coverage_mode 默认为 'preserve'；设为 'hard' 时，
%   还会在 coverage_horizon 指定的前若干轮内尽量覆盖全部合法客户端。

script_directory = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(input_file)
    input_file = fullfile(script_directory, 'result-U-6fixedge_epoch200.mat');
end
if nargin < 2 || isempty(output_file)
    [input_directory, input_name, input_extension] = fileparts(char(input_file));
    output_file = fullfile(input_directory, ...
        [input_name, '_zeroFilled', input_extension]);
end
if nargin < 3 || isempty(coverage_mode)
    coverage_mode = 'preserve';
end
if nargin < 4 || isempty(coverage_horizon)
    coverage_horizon = 150;
end

input_file = char(input_file);
output_file = char(output_file);
validate_file_paths(input_file, output_file);

data = load(input_file);
methods = build_method_descriptors();
validate_source_data(data, methods);

audit = initialize_audit(input_file, output_file);
for method_index = 1:numel(methods)
    method = methods(method_index);
    [data, method_audit] = fill_one_method(data, method);
    audit.methods.(method.name) = method_audit;
    audit.total_replacements = audit.total_replacements + ...
        method_audit.replacement_count;
end

% 零轮次修复完成后执行可选的跨轮覆盖修复，且不改变任何人数矩阵。
[data, coverage_audit] = enforce_client_coverage( ...
    data, coverage_mode, coverage_horizon);
audit.coverage = coverage_audit;

% 方法级底层数据替换完成后，重新计算所有依赖 epoch 数据的汇总字段。
data = rebuild_summary_fields(data);
audit.remaining_zero_count = count_all_zero_clients(data, methods);
audit.all_zero_removed = audit.remaining_zero_count == 0;
if ~audit.all_zero_removed
    error('fill_zero_client_rounds:RemainingZero', ...
        '修复完成后仍存在 %d 个客户端数量为零的单元。', ...
        audit.remaining_zero_count);
end

audit.created_at = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
data.zero_client_fill_schema_version = '1.0';
data.zero_client_fill_source_file = input_file;
data.zero_client_fill_created_at = audit.created_at;
data.zero_client_fill_note = [ ...
    '六种方法分别按利用率检查；零客户端轮次由同方法同利用率的上一有效轮次', ...
    '整套覆盖。第一轮为零时使用该列首个非零轮次回填。客户端覆盖模式为 ', ...
    coverage_audit.mode, '，覆盖窗口为前 ', ...
    num2str(coverage_audit.effective_horizon), ' 轮。源 MAT 未被修改。'];
data.client_coverage_schema_version = coverage_audit.schema_version;
data.client_coverage_mode = coverage_audit.mode;
data.client_coverage_horizon = coverage_audit.effective_horizon;
data.client_coverage_audit = coverage_audit;
data.zero_client_fill_audit = audit;

output_directory = fileparts(output_file);
if ~isempty(output_directory) && ~isfolder(output_directory)
    mkdir(output_directory);
end
save(output_file, '-struct', 'data');

fprintf('零客户端轮次修复文件已生成：%s\n', output_file);
fprintf('共替换 %d 个方法-轮次-利用率单元，修复后零值数量为 %d。\n', ...
    audit.total_replacements, audit.remaining_zero_count);
fprintf('客户端覆盖模式=%s，窗口=%d，映射交换=%d 次。\n', ...
    coverage_audit.mode, coverage_audit.effective_horizon, ...
    coverage_audit.total_swaps);
end


function validate_file_paths(input_file, output_file)
%VALIDATE_FILE_PATHS 检查输入文件存在且输出文件不会覆盖输入文件。
if ~isfile(input_file)
    error('fill_zero_client_rounds:SourceNotFound', ...
        '找不到输入 MAT 文件：%s', input_file);
end
if strcmpi(normalize_path(input_file), normalize_path(output_file))
    error('fill_zero_client_rounds:SameFile', ...
        '输出文件不能覆盖输入 MAT 文件。');
end
end


function normalized = normalize_path(file_path)
%NORMALIZE_PATH 将相对路径转换为便于比较的绝对路径文本。
file_path = char(file_path);
is_windows_absolute = ~isempty(regexp(file_path, '^[A-Za-z]:[\\/]', 'once'));
is_unc_path = startsWith(file_path, '\\');
is_unix_absolute = startsWith(file_path, '/');
if is_windows_absolute || is_unc_path || is_unix_absolute
    normalized = file_path;
else
    normalized = fullfile(pwd, file_path);
end
normalized = strrep(normalized, '/', filesep);
end


function methods = build_method_descriptors()
%BUILD_METHOD_DESCRIPTORS 定义六种方法的客户端字段及其关联快照字段。
methods = repmat(struct( ...
    'name', '', ...
    'client_field', '', ...
    'kind', '', ...
    'policy_index', [], ...
    'numeric_fields', {{}}, ...
    'policy_fields', {{}}, ...
    'cell_fields', {{}}), 1, 6);

methods(1) = make_hfl_descriptor('HFLSnF_fix', 'HFLSnF', 'fix', 3);
methods(2) = make_hfl_descriptor('HFLSnF_los', 'HFLSnF', 'los', 1);
methods(3) = make_hfl_descriptor('HFLnoSnF_fix', 'HFLnoSnF', 'fix', 3);
methods(4) = make_hfl_descriptor('HFLnoSnF_los', 'HFLnoSnF', 'los', 1);
methods(5) = make_fl_descriptor('FLSnF');
methods(6) = make_fl_descriptor('FLnoSnF');
end


function method = make_hfl_descriptor(name, prefix, suffix, policy_index)
%MAKE_HFL_DESCRIPTOR 构造一种 HFL 策略的关联字段说明。
method = struct();
method.name = name;
method.client_field = ['client_num_', prefix, '_', suffix];
method.kind = 'hfl';
method.policy_index = policy_index;
method.numeric_fields = { ...
    ['group_num_', prefix, '_', suffix], ...
    ['client_num_', prefix, '_', suffix], ...
    ['max_layer_', prefix, '_', suffix], ...
    ['time_agg_', prefix, '_', suffix]};
method.policy_fields = { ...
    ['group_num_', prefix], ...
    ['client_num_', prefix], ...
    ['max_layer_', prefix], ...
    ['actual_c2e_map_', prefix], ...
    ['DynEdgeSet_', prefix]};
method.cell_fields = {};
end


function method = make_fl_descriptor(prefix)
%MAKE_FL_DESCRIPTOR 构造一种普通 FL 方法的关联字段说明。
method = struct();
method.name = prefix;
method.client_field = ['client_num_', prefix];
method.kind = 'fl';
method.policy_index = [];
method.numeric_fields = { ...
    ['client_num_', prefix], ...
    ['max_layer_', prefix], ...
    ['time_agg_', prefix]};
method.policy_fields = {};
method.cell_fields = {['c2cmap_', prefix, '_all']};
end


function validate_source_data(data, methods)
%VALIDATE_SOURCE_DATA 检查六种方法的必需字段、尺寸与客户端数量合法性。
for method_index = 1:numel(methods)
    method = methods(method_index);
    required_fields = [method.numeric_fields, method.policy_fields, method.cell_fields];
    for field_index = 1:numel(required_fields)
        field_name = required_fields{field_index};
        if ~isfield(data, field_name)
            error('fill_zero_client_rounds:MissingField', ...
                '输入 MAT 缺少 %s 方法所需字段：%s', method.name, field_name);
        end
    end

    counts = data.(method.client_field);
    if ~isnumeric(counts) || ~ismatrix(counts) || isempty(counts) || ...
            any(~isfinite(counts(:))) || any(counts(:) < 0) || ...
            any(counts(:) ~= round(counts(:)))
        error('fill_zero_client_rounds:InvalidClientMatrix', ...
            '%s 必须是非空、有限、非负的整数矩阵。', method.client_field);
    end
    expected_size = size(counts);
    validate_associated_sizes(data, method, expected_size);
end
end


function validate_associated_sizes(data, method, expected_size)
%VALIDATE_ASSOCIATED_SIZES 检查方法关联字段与客户端矩阵尺寸一致。
associated_fields = [method.numeric_fields, method.policy_fields, method.cell_fields];
for field_index = 1:numel(associated_fields)
    field_name = associated_fields{field_index};
    if ~isequal(size(data.(field_name)), expected_size)
        error('fill_zero_client_rounds:SizeMismatch', ...
            '%s 的尺寸为 %s，但 %s 的尺寸为 %s。', ...
            field_name, mat2str(size(data.(field_name))), ...
            method.client_field, mat2str(expected_size));
    end
end
end


function audit = initialize_audit(input_file, output_file)
%INITIALIZE_AUDIT 创建零客户端轮次替换的顶层审计结构。
audit = struct();
audit.schema_version = '1.0';
audit.source_file = input_file;
audit.output_file = output_file;
audit.rule = [ ...
    '每种方法、每个利用率独立处理；零值使用修复后的上一轮。', ...
    '第一轮为零时使用该列首个非零轮次。'];
audit.total_replacements = 0;
audit.remaining_zero_count = NaN;
audit.all_zero_removed = false;
audit.methods = struct();
end


function [data, audit] = fill_one_method(data, method)
%FILL_ONE_METHOD 修复一种方法在全部利用率下的零客户端轮次。
counts = data.(method.client_field);
[epoch_count, util_count] = size(counts);
zero_positions = find(counts == 0);
[zero_rounds, zero_utils] = ind2sub(size(counts), zero_positions);

audit = struct();
audit.client_field = method.client_field;
audit.original_zero_count = numel(zero_positions);
audit.replacement_count = 0;
audit.positions = zeros(numel(zero_positions), 2);
audit.source_rounds = zeros(numel(zero_positions), 1);
audit.first_round_forward_fill_count = 0;
audit_index = 0;

for util_index = 1:util_count
    first_valid_round = find(counts(:, util_index) > 0, 1, 'first');
    if isempty(first_valid_round)
        error('fill_zero_client_rounds:AllZeroColumn', ...
            '%s 在利用率列 %d 的 %d 轮全部为 0，无法回填。', ...
            method.client_field, util_index, epoch_count);
    end

    for round_index = 1:epoch_count
        if data.(method.client_field)(round_index, util_index) ~= 0
            continue;
        end
        if round_index == 1
            source_round = first_valid_round;
            audit.first_round_forward_fill_count = ...
                audit.first_round_forward_fill_count + 1;
        else
            source_round = round_index - 1;
        end

        data = copy_method_snapshot( ...
            data, method, source_round, round_index, util_index);
        audit_index = audit_index + 1;
        audit.positions(audit_index, :) = [round_index, util_index];
        audit.source_rounds(audit_index) = source_round;
        audit.replacement_count = audit.replacement_count + 1;
    end
end

% 保留按 find 顺序得到的原始零值位置，便于审计输入文件。
audit.original_zero_positions = [zero_rounds, zero_utils];
audit.positions = audit.positions(1:audit_index, :);
audit.source_rounds = audit.source_rounds(1:audit_index);
audit.remaining_zero_count = nnz(data.(method.client_field) == 0);
if audit.remaining_zero_count ~= 0
    error('fill_zero_client_rounds:MethodStillHasZero', ...
        '%s 修复后仍存在零客户端轮次。', method.client_field);
end
end


function data = copy_method_snapshot( ...
        data, method, source_round, target_round, util_index)
%COPY_METHOD_SNAPSHOT 将同一方法的关联数据从来源轮复制到目标轮。
for field_index = 1:numel(method.numeric_fields)
    field_name = method.numeric_fields{field_index};
    data.(field_name)(target_round, util_index) = ...
        data.(field_name)(source_round, util_index);
end

if strcmp(method.kind, 'hfl')
    for field_index = 1:numel(method.policy_fields)
        field_name = method.policy_fields{field_index};
        data = copy_policy_value(data, field_name, method.policy_index, ...
            source_round, target_round, util_index);
    end
else
    for field_index = 1:numel(method.cell_fields)
        field_name = method.cell_fields{field_index};
        data.(field_name){target_round, util_index} = ...
            data.(field_name){source_round, util_index};
    end
end
end


function data = copy_policy_value( ...
        data, field_name, policy_index, source_round, target_round, util_index)
%COPY_POLICY_VALUE 只复制 HFL 三策略 cell 中指定策略的数据。
source_policy = data.(field_name){source_round, util_index};
target_policy = data.(field_name){target_round, util_index};
if ~iscell(source_policy) || ~iscell(target_policy) || ...
        numel(source_policy) < policy_index || numel(target_policy) < policy_index
    error('fill_zero_client_rounds:InvalidPolicyCell', ...
        '%s 的轮次 cell 必须至少包含 %d 个策略位置。', ...
        field_name, policy_index);
end
target_policy{policy_index} = source_policy{policy_index};
data.(field_name){target_round, util_index} = target_policy;
end


function data = rebuild_summary_fields(data)
%REBUILD_SUMMARY_FIELDS 根据修复后的底层矩阵重新计算均值与总时间字段。
hfl_names = {'HFLSnF_fix', 'HFLSnF_los', ...
    'HFLnoSnF_fix', 'HFLnoSnF_los'};
for name_index = 1:numel(hfl_names)
    name = hfl_names{name_index};
    data.(['enum_', name]) = mean(data.(['group_num_', name]), 1);
    data.(['cnum_', name]) = mean(data.(['client_num_', name]), 1);
    data.(['mlayer_', name]) = mean(data.(['max_layer_', name]), 1);
    data.(['aver_time_agg_', name]) = mean(data.(['time_agg_', name]), 1);
    data.(['time_', name]) = data.(['mlayer_', name]) .* 16 + 100;
    data.(['toall_time_', name]) = data.(['time_', name]) + ...
        data.(['aver_time_agg_', name]);
end

fl_names = {'FLSnF', 'FLnoSnF'};
for name_index = 1:numel(fl_names)
    name = fl_names{name_index};
    data.(['cnum_', name]) = mean(data.(['client_num_', name]), 1);
    data.(['mlayer_', name]) = mean(data.(['max_layer_', name]), 1);
    data.(['aver_time_agg_', name]) = mean(data.(['time_agg_', name]), 1);
    data.(['time_', name]) = data.(['mlayer_', name]) .* 16 + 100;
    data.(['toall_time_', name]) = data.(['time_', name]) + ...
        data.(['aver_time_agg_', name]);
end
end


function zero_count = count_all_zero_clients(data, methods)
%COUNT_ALL_ZERO_CLIENTS 统计六种方法修复后剩余的客户端数量零值。
zero_count = 0;
for method_index = 1:numel(methods)
    counts = data.(methods(method_index).client_field);
    zero_count = zero_count + nnz(counts == 0);
end
end

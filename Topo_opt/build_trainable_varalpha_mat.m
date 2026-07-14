function audit = build_trainable_varalpha_mat(input_file, output_file, varAlpha)
%BUILD_TRAINABLE_VARALPHA_MAT 生成方差受控且可直接训练的 MAT 文件。
%   audit = BUILD_TRAINABLE_VARALPHA_MAT(input_file, output_file, varAlpha)
%   依次执行以下处理：
%   1. 修复六种方法中客户端数量为零的轮次；
%   2. 使用 varAlpha 控制六种客户端数量矩阵的方差；
%   3. 将 *_varctrl 人数写回训练读取的标准字段；
%   4. 同步重建 HFL 分组映射和 FL 客户端映射；
%   5. 检查人数、分组、映射、范围和整数性是否一致。
%
%   输入文件不会被覆盖。varAlpha 必须位于 [0,1]；其含义是目标方差相对
%   于零值修复后数据原方差的比例。

script_directory = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(input_file)
    input_file = fullfile(script_directory, 'result-U-6fixedge_epoch200.mat');
end
if nargin < 3 || isempty(varAlpha)
    varAlpha = 0.5;
end
if nargin < 2 || isempty(output_file)
    [input_directory, input_name, input_extension] = fileparts(char(input_file));
    alpha_token = format_alpha_token(varAlpha);
    output_file = fullfile(input_directory, ...
        [input_name, '_varAlpha_', alpha_token, '_trainable', input_extension]);
end

input_file = char(input_file);
output_file = char(output_file);
validate_pipeline_inputs(input_file, output_file, varAlpha);

% 中间文件只用于复用已经验证过的两个处理函数，流程结束后自动删除。
zero_filled_temp = [tempname(script_directory), '_zeroFilled.mat'];
controlled_temp = [tempname(script_directory), '_varControlled.mat'];
cleanup_guard = onCleanup(@() cleanup_temp_files( ...
    zero_filled_temp, controlled_temp)); %#ok<NASGU>

fprintf('步骤 1/4：修复零客户端轮次。\n');
zero_fill_audit = fill_zero_client_rounds(input_file, zero_filled_temp);

fprintf('步骤 2/4：按 varAlpha=%.6g 控制客户端数量方差。\n', varAlpha);
variance_audit = control_client_variance( ...
    zero_filled_temp, controlled_temp, varAlpha);

fprintf('步骤 3/4：重建训练人数、分组和客户端映射。\n');
data = load(controlled_temp);
valid_client_ids = infer_valid_client_ids(data);
data = promote_hfl_method(data, 'HFLSnF', valid_client_ids);
data = promote_hfl_method(data, 'HFLnoSnF', valid_client_ids);
data = promote_fl_method(data, 'FLSnF', valid_client_ids);
data = promote_fl_method(data, 'FLnoSnF', valid_client_ids);
data = rebuild_training_summaries(data);

fprintf('步骤 4/4：检查训练字段与映射一致性。\n');
validation = validate_trainable_data(data, valid_client_ids);

audit = struct();
audit.schema_version = '1.0';
audit.source_file = input_file;
audit.output_file = output_file;
audit.varAlpha = varAlpha;
audit.zero_fill = zero_fill_audit;
audit.variance_control = variance_audit;
audit.validation = validation;
audit.created_at = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));

% 用最终输入输出语义覆盖中间临时文件留下的路径元数据。
data.training_schema_version = 'varalpha-trainable-1.0';
data.training_source_file = input_file;
data.training_varAlpha = varAlpha;
data.training_created_at = audit.created_at;
data.training_note = [ ...
    '先修复零客户端轮次，再控制客户端数量方差；六组标准 client_num_* 字段', ...
    '已等于对应 *_varctrl 字段，HFL/FL 映射已同步重建，可供训练加载器直接读取。'];
data.trainable_varalpha_audit = audit;
data.variance_control_source_file = input_file;

output_directory = fileparts(output_file);
if ~isempty(output_directory) && ~isfolder(output_directory)
    mkdir(output_directory);
end
save(output_file, '-struct', 'data');

fprintf('可训练方差控制 MAT 已生成：%s\n', output_file);
fprintf('varAlpha=%.6g，修复零值 %d 个，映射检查 %d 个快照，剩余零值 0。\n', ...
    varAlpha, zero_fill_audit.total_replacements, ...
    validation.snapshot_count);
end


function validate_pipeline_inputs(input_file, output_file, varAlpha)
%VALIDATE_PIPELINE_INPUTS 检查文件路径和方差比例参数。
if ~isfile(input_file)
    error('build_trainable_varalpha_mat:SourceNotFound', ...
        '找不到输入 MAT 文件：%s', input_file);
end
if strcmpi(normalize_pipeline_path(input_file), ...
        normalize_pipeline_path(output_file))
    error('build_trainable_varalpha_mat:SameFile', ...
        '输出文件不能覆盖输入 MAT 文件。');
end
if ~isnumeric(varAlpha) || ~isscalar(varAlpha) || ~isfinite(varAlpha) || ...
        varAlpha < 0 || varAlpha > 1
    error('build_trainable_varalpha_mat:InvalidVarAlpha', ...
        'varAlpha 必须是 [0,1] 范围内的有限数值标量。');
end
end


function normalized = normalize_pipeline_path(file_path)
%NORMALIZE_PIPELINE_PATH 将相对路径转换为便于比较的绝对路径。
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


function token = format_alpha_token(varAlpha)
%FORMAT_ALPHA_TOKEN 将 varAlpha 转换为适合输出文件名的文本。
token = sprintf('%.6g', varAlpha);
token = strrep(token, '.', 'p');
token = strrep(token, '-', 'm');
end


function cleanup_temp_files(varargin)
%CLEANUP_TEMP_FILES 删除一键流程产生的临时 MAT 文件。
for file_index = 1:nargin
    file_path = varargin{file_index};
    if isfile(file_path)
        delete(file_path);
    end
end
end


function valid_client_ids = infer_valid_client_ids(data)
%INFER_VALID_CLIENT_IDS 根据节点总数和云节点编号推导合法客户端 ID。
required_fields = {'num_of_nodes', 'Cloud'};
for field_index = 1:numel(required_fields)
    if ~isfield(data, required_fields{field_index})
        error('build_trainable_varalpha_mat:MissingNodeField', ...
            '输入 MAT 缺少字段：%s', required_fields{field_index});
    end
end
node_count = double(data.num_of_nodes);
cloud_id = double(data.Cloud);
if ~isscalar(node_count) || node_count < 1 || node_count ~= round(node_count) || ...
        ~isscalar(cloud_id) || cloud_id < 1 || cloud_id > node_count || ...
        cloud_id ~= round(cloud_id)
    error('build_trainable_varalpha_mat:InvalidNodeMetadata', ...
        'num_of_nodes 和 Cloud 必须是合法整数节点编号。');
end
valid_client_ids = setdiff(1:node_count, cloud_id, 'stable');
end


function data = promote_hfl_method(data, prefix, valid_client_ids)
%PROMOTE_HFL_METHOD 将一种 HFL 方法的受控人数写回并同步重建分组映射。
mapping_field = ['actual_c2e_map_', prefix];
edge_field = ['DynEdgeSet_', prefix];
raw_group_field = ['group_num_', prefix];
raw_client_field = ['client_num_', prefix];
required_fields = {mapping_field, edge_field, raw_group_field, raw_client_field};
assert_fields_exist(data, required_fields, prefix);

policies = struct( ...
    'suffix', {'los', 'fix'}, ...
    'policy_index', {1, 3});
for policy_number = 1:numel(policies)
    suffix = policies(policy_number).suffix;
    policy_index = policies(policy_number).policy_index;
    client_field = ['client_num_', prefix, '_', suffix];
    controlled_field = [client_field, '_varctrl'];
    group_field = ['group_num_', prefix, '_', suffix];
    assert_fields_exist(data, {client_field, controlled_field, group_field}, prefix);

    controlled_counts = double(data.(controlled_field));
    [epoch_count, util_count] = size(controlled_counts);
    for util_index = 1:util_count
        for round_index = 1:epoch_count
            target_count = controlled_counts(round_index, util_index);
            validate_target_count(target_count, numel(valid_client_ids), controlled_field);

            edge_policy = extract_policy_value( ...
                data.(edge_field){round_index, util_index}, policy_index, edge_field);
            edge_ids = flatten_numeric_values(edge_policy, edge_field);
            source_mapping = extract_policy_value( ...
                data.(mapping_field){round_index, util_index}, ...
                policy_index, mapping_field);
            groups = split_hfl_groups( ...
                source_mapping, numel(edge_ids), mapping_field);
            resized_groups = resize_hfl_groups( ...
                groups, target_count, valid_client_ids);

            data = set_policy_value(data, mapping_field, round_index, ...
                util_index, policy_index, resized_groups);
            group_sizes = cellfun(@numel, resized_groups);
            data = set_policy_value(data, raw_client_field, round_index, ...
                util_index, policy_index, group_sizes);
            data = set_policy_value(data, raw_group_field, round_index, ...
                util_index, policy_index, nnz(group_sizes));
            data.(client_field)(round_index, util_index) = target_count;
            data.(group_field)(round_index, util_index) = nnz(group_sizes);

            time_field = ['time_agg_', prefix, '_', suffix];
            if isfield(data, time_field) && isfield(data, 'agg_time_per_clients')
                data.(time_field)(round_index, util_index) = calc_hfl_agg_time( ...
                    group_sizes, nnz(group_sizes), data.agg_time_per_clients);
            end
        end
    end
end
end


function data = promote_fl_method(data, prefix, valid_client_ids)
%PROMOTE_FL_METHOD 将一种普通 FL 方法的受控人数写回并同步重建映射。
client_field = ['client_num_', prefix];
controlled_field = [client_field, '_varctrl'];
mapping_field = ['c2cmap_', prefix, '_all'];
assert_fields_exist(data, ...
    {client_field, controlled_field, mapping_field}, prefix);

controlled_counts = double(data.(controlled_field));
[epoch_count, util_count] = size(controlled_counts);
for util_index = 1:util_count
    for round_index = 1:epoch_count
        target_count = controlled_counts(round_index, util_index);
        validate_target_count(target_count, numel(valid_client_ids), controlled_field);
        original_ids = flatten_numeric_values( ...
            data.(mapping_field){round_index, util_index}, mapping_field);
        selected_ids = resize_flat_mapping( ...
            original_ids, target_count, valid_client_ids);
        data.(mapping_field){round_index, util_index} = selected_ids;
        data.(client_field)(round_index, util_index) = target_count;
    end
end

time_field = ['time_agg_', prefix];
if isfield(data, time_field) && isfield(data, 'agg_time_per_clients')
    data.(time_field) = data.(client_field) .* data.agg_time_per_clients;
end
end


function assert_fields_exist(data, field_names, context)
%ASSERT_FIELDS_EXIST 检查指定处理阶段需要的 MAT 字段是否齐全。
for field_index = 1:numel(field_names)
    if ~isfield(data, field_names{field_index})
        error('build_trainable_varalpha_mat:MissingField', ...
            '%s 缺少必需字段：%s', context, field_names{field_index});
    end
end
end


function validate_target_count(target_count, maximum_count, field_name)
%VALIDATE_TARGET_COUNT 检查受控客户端数量是合法范围内的整数。
if ~isscalar(target_count) || ~isfinite(target_count) || ...
        target_count < 0 || target_count > maximum_count || ...
        target_count ~= round(target_count)
    error('build_trainable_varalpha_mat:InvalidTargetCount', ...
        '%s 包含非法目标人数：%g', field_name, target_count);
end
end


function value = extract_policy_value(policy_cell, policy_index, context)
%EXTRACT_POLICY_VALUE 从 HFL 三策略 cell 中提取指定策略。
if ~iscell(policy_cell) || numel(policy_cell) < policy_index
    error('build_trainable_varalpha_mat:InvalidPolicyCell', ...
        '%s 必须是至少含 %d 个位置的策略 cell。', context, policy_index);
end
value = policy_cell{policy_index};
end


function values = flatten_numeric_values(value, context)
%FLATTEN_NUMERIC_VALUES 递归展开数值数组或嵌套 cell 中的整数 ID。
if isempty(value)
    values = [];
    return;
end
if iscell(value)
    values = [];
    for item_index = 1:numel(value)
        values = [values, flatten_numeric_values(value{item_index}, context)]; %#ok<AGROW>
    end
    return;
end
if ~isnumeric(value) || any(~isfinite(value(:))) || ...
        any(value(:) ~= round(value(:)))
    error('build_trainable_varalpha_mat:InvalidMappingValue', ...
        '%s 包含非有限或非整数客户端 ID。', context);
end
values = double(value(:)).';
end


function groups = split_hfl_groups(mapping, edge_count, context)
%SPLIT_HFL_GROUPS 按边缘槽位数量恢复 HFL 分组边界。
if edge_count == 0
    if ~isempty(flatten_numeric_values(mapping, context))
        error('build_trainable_varalpha_mat:MappingWithoutEdge', ...
            '%s 没有边缘槽位但包含客户端。', context);
    end
    groups = cell(1, 0);
    return;
end
if iscell(mapping)
    if numel(mapping) ~= edge_count
        error('build_trainable_varalpha_mat:GroupSlotMismatch', ...
            '%s 有 %d 个映射槽位，但边缘槽位数为 %d。', ...
            context, numel(mapping), edge_count);
    end
    groups = reshape(mapping, 1, []);
elseif edge_count == 1
    groups = {mapping};
else
    error('build_trainable_varalpha_mat:MissingGroupBoundary', ...
        '%s 的非 cell 映射无法对应 %d 个边缘槽位。', context, edge_count);
end
for group_index = 1:numel(groups)
    groups{group_index} = flatten_numeric_values(groups{group_index}, context);
end
end


function resized_groups = resize_hfl_groups(groups, target_total, valid_client_ids)
%RESIZE_HFL_GROUPS 按目标总人数缩放 HFL 各组并保持客户端全局唯一。
if target_total > 0 && isempty(groups)
    error('build_trainable_varalpha_mat:NoEdgeForClients', ...
        '目标客户端数量大于 0，但当前轮没有可用边缘槽位。');
end

normalized_groups = cell(size(groups));
for group_index = 1:numel(groups)
    normalized_groups{group_index} = unique_valid_ids( ...
        groups{group_index}, valid_client_ids);
end
original_sizes = cellfun(@numel, normalized_groups);
target_sizes = allocate_group_counts(original_sizes, target_total);
resized_groups = cell(size(normalized_groups));
used_ids = [];

% 优先保留客户端原来的组归属。
for group_index = 1:numel(normalized_groups)
    original_group = normalized_groups{group_index};
    keep_count = min(numel(original_group), target_sizes(group_index));
    for client_index = 1:numel(original_group)
        client_id = original_group(client_index);
        if numel(resized_groups{group_index}) >= keep_count
            break;
        end
        if ~ismember(client_id, used_ids)
            resized_groups{group_index}(end + 1) = client_id; %#ok<AGROW>
            used_ids(end + 1) = client_id; %#ok<AGROW>
        end
    end
end

priority_ids = [];
for group_index = 1:numel(normalized_groups)
    priority_ids = [priority_ids, normalized_groups{group_index}]; %#ok<AGROW>
end
fill_pool = unique_valid_ids( ...
    [priority_ids, valid_client_ids], valid_client_ids);
for group_index = 1:numel(resized_groups)
    for client_index = 1:numel(fill_pool)
        if numel(resized_groups{group_index}) >= target_sizes(group_index)
            break;
        end
        client_id = fill_pool(client_index);
        if ~ismember(client_id, used_ids)
            resized_groups{group_index}(end + 1) = client_id; %#ok<AGROW>
            used_ids(end + 1) = client_id; %#ok<AGROW>
        end
    end
end
if sum(cellfun(@numel, resized_groups)) ~= target_total
    error('build_trainable_varalpha_mat:CannotResizeHFL', ...
        '无法构造目标人数为 %d 的 HFL 映射。', target_total);
end
end


function counts = allocate_group_counts(original_sizes, target_total)
%ALLOCATE_GROUP_COUNTS 按原组规模比例分配受控后的整数客户端数量。
slot_count = numel(original_sizes);
counts = zeros(1, slot_count);
if target_total == 0
    return;
end
if slot_count == 0
    error('build_trainable_varalpha_mat:NoGroupSlot', ...
        '目标人数大于 0，但没有可分配的组槽位。');
end

eligible = find(original_sizes > 0);
if isempty(eligible)
    eligible = 1:slot_count;
end
[~, rank_order] = sortrows( ...
    [-original_sizes(eligible).', eligible.'], [1, 2]);
ranked_eligible = eligible(rank_order);
initial_count = min(target_total, numel(ranked_eligible));
counts(ranked_eligible(1:initial_count)) = 1;

remaining = target_total - sum(counts);
if remaining == 0
    return;
end
weights = double(original_sizes(eligible));
if sum(weights) == 0
    weights = ones(size(weights));
end
quotas = remaining .* weights ./ sum(weights);
additions = floor(quotas);
counts(eligible) = counts(eligible) + additions;
remainder = remaining - sum(additions);
if remainder > 0
    fractions = quotas - additions;
    [~, fraction_order] = sortrows( ...
        [-fractions(:), -weights(:), eligible(:)], [1, 2, 3]);
    selected = eligible(fraction_order(1:remainder));
    counts(selected) = counts(selected) + 1;
end
if sum(counts) ~= target_total
    error('build_trainable_varalpha_mat:AllocationMismatch', ...
        '组内人数分配总和与目标人数不一致。');
end
end


function values = unique_valid_ids(values, valid_client_ids)
%UNIQUE_VALID_IDS 按原顺序保留合法且不重复的客户端 ID。
values = double(values(:)).';
if any(~ismember(values, valid_client_ids))
    invalid_values = values(~ismember(values, valid_client_ids));
    error('build_trainable_varalpha_mat:InvalidClientId', ...
        '映射包含非法客户端 ID：%s', mat2str(unique(invalid_values)));
end
values = unique(values, 'stable');
end


function selected_ids = resize_flat_mapping( ...
        original_ids, target_total, valid_client_ids)
%RESIZE_FLAT_MAPPING 保留原参与者优先级并将 FL 映射调整到目标人数。
original_ids = unique_valid_ids(original_ids, valid_client_ids);
selected_ids = original_ids(1:min(target_total, numel(original_ids)));
for client_index = 1:numel(valid_client_ids)
    if numel(selected_ids) >= target_total
        break;
    end
    client_id = valid_client_ids(client_index);
    if ~ismember(client_id, selected_ids)
        selected_ids(end + 1) = client_id; %#ok<AGROW>
    end
end
if numel(selected_ids) ~= target_total
    error('build_trainable_varalpha_mat:CannotResizeFL', ...
        '无法构造目标人数为 %d 的 FL 映射。', target_total);
end
end


function data = set_policy_value( ...
        data, field_name, round_index, util_index, policy_index, value)
%SET_POLICY_VALUE 修改外层 epoch×util cell 中指定 HFL 策略的数据。
policy_cell = data.(field_name){round_index, util_index};
if ~iscell(policy_cell) || numel(policy_cell) < policy_index
    error('build_trainable_varalpha_mat:InvalidPolicyForWrite', ...
        '%s 必须至少包含 %d 个策略位置。', field_name, policy_index);
end
policy_cell{policy_index} = value;
data.(field_name){round_index, util_index} = policy_cell;
end


function data = rebuild_training_summaries(data)
%REBUILD_TRAINING_SUMMARIES 根据受控训练矩阵重算全部方法汇总字段。
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


function validation = validate_trainable_data(data, valid_client_ids)
%VALIDATE_TRAINABLE_DATA 验证六种方法的人数、分组和映射逐快照一致。
validation = struct();
validation.snapshot_count = 0;
validation.standard_equals_varctrl = true;
validation.mapping_count_matches = true;
validation.all_client_ids_valid = true;
validation.no_zero_client_count = true;

hfl_policies = { ...
    'HFLSnF', 'los', 1; ...
    'HFLSnF', 'fix', 3; ...
    'HFLnoSnF', 'los', 1; ...
    'HFLnoSnF', 'fix', 3};
for method_index = 1:size(hfl_policies, 1)
    prefix = hfl_policies{method_index, 1};
    suffix = hfl_policies{method_index, 2};
    policy_index = hfl_policies{method_index, 3};
    client_field = ['client_num_', prefix, '_', suffix];
    controlled_field = [client_field, '_varctrl'];
    group_field = ['group_num_', prefix, '_', suffix];
    if ~isequal(data.(client_field), data.(controlled_field))
        error('build_trainable_varalpha_mat:ControlledCountMismatch', ...
            '%s 未逐元素等于 %s。', client_field, controlled_field);
    end
    if any(data.(client_field)(:) == 0)
        error('build_trainable_varalpha_mat:ZeroAfterControl', ...
            '%s 仍包含零客户端轮次。', client_field);
    end

    for snapshot_index = 1:numel(data.(client_field))
        mapping_policy = extract_policy_value( ...
            data.(['actual_c2e_map_', prefix]){snapshot_index}, ...
            policy_index, client_field);
        client_ids = flatten_numeric_values(mapping_policy, client_field);
        if numel(client_ids) ~= data.(client_field)(snapshot_index) || ...
                numel(unique(client_ids)) ~= numel(client_ids) || ...
                any(~ismember(client_ids, valid_client_ids))
            error('build_trainable_varalpha_mat:HFLMappingMismatch', ...
                '%s 的第 %d 个快照人数或客户端 ID 不一致。', ...
                client_field, snapshot_index);
        end
        if iscell(mapping_policy)
            effective_groups = nnz(cellfun(@(item) ...
                ~isempty(flatten_numeric_values(item, client_field)), mapping_policy));
        else
            effective_groups = ~isempty(client_ids);
        end
        if effective_groups ~= data.(group_field)(snapshot_index)
            error('build_trainable_varalpha_mat:HFLGroupMismatch', ...
                '%s 的第 %d 个快照有效组数不一致。', group_field, snapshot_index);
        end
        validation.snapshot_count = validation.snapshot_count + 1;
    end
end

fl_names = {'FLSnF', 'FLnoSnF'};
for method_index = 1:numel(fl_names)
    prefix = fl_names{method_index};
    client_field = ['client_num_', prefix];
    controlled_field = [client_field, '_varctrl'];
    mapping_field = ['c2cmap_', prefix, '_all'];
    if ~isequal(data.(client_field), data.(controlled_field))
        error('build_trainable_varalpha_mat:ControlledCountMismatch', ...
            '%s 未逐元素等于 %s。', client_field, controlled_field);
    end
    if any(data.(client_field)(:) == 0)
        error('build_trainable_varalpha_mat:ZeroAfterControl', ...
            '%s 仍包含零客户端轮次。', client_field);
    end
    for snapshot_index = 1:numel(data.(client_field))
        client_ids = flatten_numeric_values( ...
            data.(mapping_field){snapshot_index}, client_field);
        if numel(client_ids) ~= data.(client_field)(snapshot_index) || ...
                numel(unique(client_ids)) ~= numel(client_ids) || ...
                any(~ismember(client_ids, valid_client_ids))
            error('build_trainable_varalpha_mat:FLMappingMismatch', ...
                '%s 的第 %d 个快照人数或客户端 ID 不一致。', ...
                client_field, snapshot_index);
        end
        validation.snapshot_count = validation.snapshot_count + 1;
    end
end
end

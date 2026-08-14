function [data, audit] = enforce_client_coverage( ...
        data, coverage_mode, coverage_horizon)
%ENFORCE_CLIENT_COVERAGE 在不改变逐轮人数的前提下修复客户端永久缺席。
%   [data, audit] = ENFORCE_CLIENT_COVERAGE(data, coverage_mode,
%   coverage_horizon) 对六种训练映射的每个利用率分别统计指定轮次窗口内的
%   客户端参与次数。coverage_mode='preserve' 时只审计、不修改；设置为
%   'hard' 时，用永久缺席客户端替换高频客户端的一次参与。
%
%   交换只改变客户端编号，不改变每轮参与人数、HFL 分组边界或组内人数。
%   如果总参与名额不足以覆盖全部客户端，本函数会尽量扩大覆盖并发出警告。

if nargin < 2 || isempty(coverage_mode)
    coverage_mode = 'preserve';
end
if nargin < 3 || isempty(coverage_horizon)
    coverage_horizon = 150;
end
coverage_mode = validatestring(coverage_mode, {'preserve', 'hard'}, ...
    mfilename, 'coverage_mode');
validate_coverage_horizon(coverage_horizon);

valid_client_ids = infer_coverage_client_ids(data);
methods = build_coverage_method_descriptors();
validate_coverage_fields(data, methods);

epoch_count = size(data.(methods(1).client_field), 1);
effective_horizon = min(double(coverage_horizon), epoch_count);
audit = initialize_coverage_audit(coverage_mode, coverage_horizon, ...
    effective_horizon, valid_client_ids);

for method_index = 1:numel(methods)
    method = methods(method_index);
    [data, method_audit] = repair_one_method( ...
        data, method, valid_client_ids, coverage_mode, effective_horizon);
    audit.methods.(method.name) = method_audit;
    audit.total_swaps = audit.total_swaps + method_audit.total_swaps;
    audit.all_feasible_columns_covered = ...
        audit.all_feasible_columns_covered && ...
        method_audit.all_feasible_columns_covered;
end

audit.created_at = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
end


function validate_coverage_horizon(coverage_horizon)
%VALIDATE_COVERAGE_HORIZON 校验覆盖窗口是正整数标量。
if ~isnumeric(coverage_horizon) || ~isscalar(coverage_horizon) || ...
        ~isfinite(coverage_horizon) || coverage_horizon < 1 || ...
        coverage_horizon ~= round(coverage_horizon)
    error('enforce_client_coverage:InvalidHorizon', ...
        'coverage_horizon 必须是正整数标量。');
end
end


function valid_client_ids = infer_coverage_client_ids(data)
%INFER_COVERAGE_CLIENT_IDS 根据节点总数和云节点确定合法客户端编号。
if ~isfield(data, 'num_of_nodes') || ~isfield(data, 'Cloud')
    error('enforce_client_coverage:MissingNodeMetadata', ...
        '输入数据必须包含 num_of_nodes 和 Cloud。');
end
node_count = double(data.num_of_nodes);
cloud_id = double(data.Cloud);
if ~isscalar(node_count) || node_count < 1 || node_count ~= round(node_count) || ...
        ~isscalar(cloud_id) || cloud_id < 1 || cloud_id > node_count || ...
        cloud_id ~= round(cloud_id)
    error('enforce_client_coverage:InvalidNodeMetadata', ...
        'num_of_nodes 和 Cloud 必须是合法整数节点编号。');
end
valid_client_ids = setdiff(1:node_count, cloud_id, 'stable');
end


function methods = build_coverage_method_descriptors()
%BUILD_COVERAGE_METHOD_DESCRIPTORS 定义六种方法的训练人数和映射字段。
methods = repmat(struct('name', '', 'client_field', '', ...
    'mapping_field', '', 'policy_index', []), 1, 6);
methods(1) = make_coverage_descriptor('HFLSnF_fix', ...
    'client_num_HFLSnF_fix', 'actual_c2e_map_HFLSnF', 3);
methods(2) = make_coverage_descriptor('HFLSnF_los', ...
    'client_num_HFLSnF_los', 'actual_c2e_map_HFLSnF', 1);
methods(3) = make_coverage_descriptor('HFLnoSnF_fix', ...
    'client_num_HFLnoSnF_fix', 'actual_c2e_map_HFLnoSnF', 3);
methods(4) = make_coverage_descriptor('HFLnoSnF_los', ...
    'client_num_HFLnoSnF_los', 'actual_c2e_map_HFLnoSnF', 1);
methods(5) = make_coverage_descriptor('FLSnF', ...
    'client_num_FLSnF', 'c2cmap_FLSnF_all', []);
methods(6) = make_coverage_descriptor('FLnoSnF', ...
    'client_num_FLnoSnF', 'c2cmap_FLnoSnF_all', []);
end


function method = make_coverage_descriptor( ...
        name, client_field, mapping_field, policy_index)
%MAKE_COVERAGE_DESCRIPTOR 构造一种方法的覆盖处理字段说明。
method = struct('name', name, 'client_field', client_field, ...
    'mapping_field', mapping_field, 'policy_index', policy_index);
end


function validate_coverage_fields(data, methods)
%VALIDATE_COVERAGE_FIELDS 校验六种方法所需字段存在且尺寸一致。
expected_size = [];
for method_index = 1:numel(methods)
    method = methods(method_index);
    required_fields = {method.client_field, method.mapping_field};
    for field_index = 1:numel(required_fields)
        if ~isfield(data, required_fields{field_index})
            error('enforce_client_coverage:MissingField', ...
                '覆盖处理缺少字段：%s', required_fields{field_index});
        end
    end
    counts = data.(method.client_field);
    mappings = data.(method.mapping_field);
    if isempty(expected_size)
        expected_size = size(counts);
    end
    if ~isequal(size(counts), expected_size) || ...
            ~isequal(size(mappings), expected_size)
        error('enforce_client_coverage:SizeMismatch', ...
            '%s 或 %s 的尺寸与实验矩阵不一致。', ...
            method.client_field, method.mapping_field);
    end
end
end


function audit = initialize_coverage_audit(coverage_mode, requested_horizon, ...
        effective_horizon, valid_client_ids)
%INITIALIZE_COVERAGE_AUDIT 初始化覆盖处理的顶层审计结构。
audit = struct();
audit.schema_version = '1.0';
audit.mode = coverage_mode;
audit.requested_horizon = double(requested_horizon);
audit.effective_horizon = effective_horizon;
audit.valid_client_ids = valid_client_ids;
audit.total_swaps = 0;
audit.all_feasible_columns_covered = true;
audit.methods = struct();
audit.created_at = '';
end


function [data, method_audit] = repair_one_method( ...
        data, method, valid_client_ids, coverage_mode, coverage_horizon)
%REPAIR_ONE_METHOD 对一种方法的各利用率列执行覆盖审计或交换修复。
util_count = size(data.(method.client_field), 2);
column_template = initialize_column_audit(valid_client_ids, coverage_horizon);
columns = repmat(column_template, 1, util_count);
total_swaps = 0;
all_feasible_columns_covered = true;

for util_index = 1:util_count
    [data, columns(util_index)] = repair_one_column(data, method, ...
        util_index, valid_client_ids, coverage_mode, coverage_horizon);
    total_swaps = total_swaps + columns(util_index).swap_count;
    if columns(util_index).is_feasible
        all_feasible_columns_covered = all_feasible_columns_covered && ...
            columns(util_index).all_clients_covered_after;
    end
end

method_audit = struct();
method_audit.name = method.name;
method_audit.client_field = method.client_field;
method_audit.mapping_field = method.mapping_field;
method_audit.policy_index = method.policy_index;
method_audit.total_swaps = total_swaps;
method_audit.all_feasible_columns_covered = all_feasible_columns_covered;
method_audit.columns = columns;
end


function column = initialize_column_audit(valid_client_ids, coverage_horizon)
%INITIALIZE_COLUMN_AUDIT 初始化单个利用率列的覆盖审计字段。
column = struct();
column.util_index = 0;
column.coverage_horizon = coverage_horizon;
column.total_participation_slots = 0;
column.is_feasible = false;
column.participation_counts_before = zeros(size(valid_client_ids));
column.participation_counts_after = zeros(size(valid_client_ids));
column.missing_client_ids_before = [];
column.missing_client_ids_after = [];
column.covered_client_count_before = 0;
column.covered_client_count_after = 0;
column.theoretical_max_covered_clients = 0;
column.all_clients_covered_after = false;
column.swap_count = 0;
column.swaps = empty_swap_records();
end


function swaps = empty_swap_records()
%EMPTY_SWAP_RECORDS 创建字段固定的空交换记录数组。
swaps = repmat(struct('round_index', 0, 'occurrence_index', 0, ...
    'donor_client_id', 0, 'replacement_client_id', 0, ...
    'donor_count_before', 0, 'round_repair_count_before', 0), 0, 1);
end


function [data, column_audit] = repair_one_column(data, method, util_index, ...
        valid_client_ids, coverage_mode, coverage_horizon)
%REPAIR_ONE_COLUMN 修复一种方法在一个利用率下的永久缺席客户端。
column_audit = initialize_column_audit(valid_client_ids, coverage_horizon);
column_audit.util_index = util_index;
[participation_counts, total_slots] = count_column_participation( ...
    data, method, util_index, coverage_horizon, valid_client_ids);
column_audit.participation_counts_before = participation_counts;
column_audit.missing_client_ids_before = ...
    valid_client_ids(participation_counts == 0);
column_audit.covered_client_count_before = nnz(participation_counts > 0);
column_audit.total_participation_slots = total_slots;
column_audit.is_feasible = total_slots >= numel(valid_client_ids);
column_audit.theoretical_max_covered_clients = ...
    min(total_slots, numel(valid_client_ids));

round_repair_counts = zeros(coverage_horizon, 1);
swaps = empty_swap_records();
if strcmp(coverage_mode, 'hard')
    missing_ids = column_audit.missing_client_ids_before;
    for missing_index = 1:numel(missing_ids)
        replacement_id = missing_ids(missing_index);
        [candidate, found] = select_swap_candidate(data, method, util_index, ...
            coverage_horizon, valid_client_ids, participation_counts, ...
            round_repair_counts, replacement_id);
        if ~found
            break;
        end
        data = replace_mapping_occurrence(data, method, candidate.round_index, ...
            util_index, candidate.occurrence_index, replacement_id);
        participation_counts(candidate.donor_position) = ...
            participation_counts(candidate.donor_position) - 1;
        replacement_position = find(valid_client_ids == replacement_id, 1);
        participation_counts(replacement_position) = ...
            participation_counts(replacement_position) + 1;
        round_repair_counts(candidate.round_index) = ...
            round_repair_counts(candidate.round_index) + 1;

        swap = struct('round_index', candidate.round_index, ...
            'occurrence_index', candidate.occurrence_index, ...
            'donor_client_id', candidate.donor_client_id, ...
            'replacement_client_id', replacement_id, ...
            'donor_count_before', candidate.donor_count, ...
            'round_repair_count_before', candidate.round_repair_count);
        swaps(end + 1, 1) = swap; %#ok<AGROW>
    end
end

[after_counts, after_slots] = count_column_participation( ...
    data, method, util_index, coverage_horizon, valid_client_ids);
if after_slots ~= total_slots
    error('enforce_client_coverage:SlotCountChanged', ...
        '%s 的利用率列 %d 在覆盖修复后总名额发生变化。', ...
        method.name, util_index);
end
column_audit.participation_counts_after = after_counts;
column_audit.missing_client_ids_after = valid_client_ids(after_counts == 0);
column_audit.covered_client_count_after = nnz(after_counts > 0);
column_audit.all_clients_covered_after = all(after_counts > 0);
column_audit.swap_count = numel(swaps);
column_audit.swaps = swaps;

if strcmp(coverage_mode, 'hard') && ...
        column_audit.covered_client_count_after < ...
        column_audit.theoretical_max_covered_clients
    error('enforce_client_coverage:CoverageNotMaximized', ...
        '%s 的利用率列 %d 未达到理论最大覆盖数。', method.name, util_index);
end
if strcmp(coverage_mode, 'hard') && ~column_audit.is_feasible && ...
        ~column_audit.all_clients_covered_after
    warning('enforce_client_coverage:InfeasibleCoverage', ...
        ['%s 的利用率列 %d 在前 %d 轮只有 %d 个参与名额，无法覆盖全部 ' ...
        '%d 个客户端；已覆盖到理论最大值 %d。'], method.name, util_index, ...
        coverage_horizon, total_slots, numel(valid_client_ids), ...
        column_audit.covered_client_count_after);
end
if strcmp(coverage_mode, 'hard') && column_audit.is_feasible && ...
        ~column_audit.all_clients_covered_after
    error('enforce_client_coverage:FeasibleCoverageFailed', ...
        '%s 的利用率列 %d 具备足够名额但仍有永久缺席客户端。', ...
        method.name, util_index);
end
end


function [counts, total_slots] = count_column_participation( ...
        data, method, util_index, coverage_horizon, valid_client_ids)
%COUNT_COLUMN_PARTICIPATION 统计一个方法利用率列内各客户端参与次数。
counts = zeros(size(valid_client_ids));
total_slots = 0;
for round_index = 1:coverage_horizon
    mapping = extract_method_mapping(data, method, round_index, util_index);
    client_ids = flatten_mapping_ids(mapping, method.mapping_field);
    expected_count = double(data.(method.client_field)(round_index, util_index));
    if numel(client_ids) ~= expected_count || ...
            numel(unique(client_ids)) ~= numel(client_ids)
        error('enforce_client_coverage:MappingCountMismatch', ...
            '%s 第 %d 轮利用率列 %d 的映射人数或唯一性不正确。', ...
            method.name, round_index, util_index);
    end
    if any(~ismember(client_ids, valid_client_ids))
        error('enforce_client_coverage:InvalidClientId', ...
            '%s 第 %d 轮利用率列 %d 包含非法客户端编号。', ...
            method.name, round_index, util_index);
    end
    for client_index = 1:numel(client_ids)
        position = find(valid_client_ids == client_ids(client_index), 1);
        counts(position) = counts(position) + 1;
    end
    total_slots = total_slots + numel(client_ids);
end
end


function mapping = extract_method_mapping(data, method, round_index, util_index)
%EXTRACT_METHOD_MAPPING 提取指定方法、轮次和利用率的实际客户端映射。
mapping = data.(method.mapping_field){round_index, util_index};
if ~isempty(method.policy_index)
    if ~iscell(mapping) || numel(mapping) < method.policy_index
        error('enforce_client_coverage:InvalidPolicyMapping', ...
            '%s 的策略映射缺少位置 %d。', ...
            method.mapping_field, method.policy_index);
    end
    mapping = mapping{method.policy_index};
end
end


function client_ids = flatten_mapping_ids(mapping, context)
%FLATTEN_MAPPING_IDS 按稳定遍历顺序展开映射中的客户端编号。
client_ids = [];
if iscell(mapping)
    for item_index = 1:numel(mapping)
        client_ids = [client_ids, ...
            flatten_mapping_ids(mapping{item_index}, context)]; %#ok<AGROW>
    end
elseif isnumeric(mapping)
    if any(~isfinite(mapping(:))) || any(mapping(:) ~= round(mapping(:)))
        error('enforce_client_coverage:InvalidMappingValue', ...
            '%s 包含非有限或非整数客户端编号。', context);
    end
    client_ids = double(mapping(:)).';
elseif ~isempty(mapping)
    error('enforce_client_coverage:UnsupportedMappingType', ...
        '%s 包含不支持的映射类型：%s。', context, class(mapping));
end
end


function [candidate, found] = select_swap_candidate(data, method, util_index, ...
        coverage_horizon, valid_client_ids, participation_counts, ...
        round_repair_counts, replacement_id)
%SELECT_SWAP_CANDIDATE 按稳定优先级选择一次不会制造新缺席的交换。
rows = zeros(0, 7);
for round_index = 1:coverage_horizon
    mapping = extract_method_mapping(data, method, round_index, util_index);
    client_ids = flatten_mapping_ids(mapping, method.mapping_field);
    if ismember(replacement_id, client_ids)
        continue;
    end
    for occurrence_index = 1:numel(client_ids)
        donor_id = client_ids(occurrence_index);
        donor_position = find(valid_client_ids == donor_id, 1);
        donor_count = participation_counts(donor_position);
        if donor_count <= 1
            continue;
        end
        rows(end + 1, :) = [round_repair_counts(round_index), ...
            -donor_count, round_index, occurrence_index, donor_id, ...
            donor_position, donor_count]; %#ok<AGROW>
    end
end

candidate = struct('round_index', 0, 'occurrence_index', 0, ...
    'donor_client_id', 0, 'donor_position', 0, 'donor_count', 0, ...
    'round_repair_count', 0);
found = ~isempty(rows);
if ~found
    return;
end
rows = sortrows(rows, [1, 2, 3, 4, 5]);
selected = rows(1, :);
candidate.round_repair_count = selected(1);
candidate.round_index = selected(3);
candidate.occurrence_index = selected(4);
candidate.donor_client_id = selected(5);
candidate.donor_position = selected(6);
candidate.donor_count = selected(7);
end


function data = replace_mapping_occurrence(data, method, round_index, ...
        util_index, occurrence_index, replacement_id)
%REPLACE_MAPPING_OCCURRENCE 在原分组和原位置替换一个客户端编号。
outer_mapping = data.(method.mapping_field){round_index, util_index};
if isempty(method.policy_index)
    [outer_mapping, replaced, visited_count] = replace_nth_numeric_value( ...
        outer_mapping, occurrence_index, replacement_id, 0);
else
    policy_mapping = outer_mapping{method.policy_index};
    [policy_mapping, replaced, visited_count] = replace_nth_numeric_value( ...
        policy_mapping, occurrence_index, replacement_id, 0);
    outer_mapping{method.policy_index} = policy_mapping;
end
if ~replaced
    error('enforce_client_coverage:OccurrenceNotFound', ...
        '%s 的第 %d 个客户端位置不存在，共遍历 %d 个位置。', ...
        method.name, occurrence_index, visited_count);
end
data.(method.mapping_field){round_index, util_index} = outer_mapping;
end


function [value, replaced, visited_count] = replace_nth_numeric_value( ...
        value, target_index, replacement_id, visited_count)
%REPLACE_NTH_NUMERIC_VALUE 递归替换映射中按列优先遍历的第 N 个数值。
replaced = false;
if iscell(value)
    for item_index = 1:numel(value)
        [value{item_index}, replaced, visited_count] = ...
            replace_nth_numeric_value(value{item_index}, target_index, ...
            replacement_id, visited_count);
        if replaced
            return;
        end
    end
elseif isnumeric(value)
    for value_index = 1:numel(value)
        visited_count = visited_count + 1;
        if visited_count == target_index
            value(value_index) = cast(replacement_id, 'like', value);
            replaced = true;
            return;
        end
    end
elseif ~isempty(value)
    error('enforce_client_coverage:UnsupportedMappingType', ...
        '映射中出现不支持的类型：%s。', class(value));
end
end

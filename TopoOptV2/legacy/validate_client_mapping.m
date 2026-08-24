function client_ids = validate_client_mapping(mapping, expected_count, valid_client_ids, context)
%VALIDATE_CLIENT_MAPPING 校验嵌套映射中的客户端编号并返回扁平编号列表。
%   client_ids = VALIDATE_CLIENT_MAPPING(mapping, expected_count,
%   valid_client_ids, context) 递归展开 cell 映射，检查数量、重复项和编号范围。

if nargin < 4 || isempty(context)
    context = 'client mapping';
end
if ~isscalar(expected_count) || expected_count < 0 || expected_count ~= floor(expected_count)
    error('validate_client_mapping:InvalidExpectedCount', ...
        '%s 的 expected_count 必须为非负整数。', context);
end

client_ids = collect_numeric_leaves(mapping);
client_ids = client_ids(:)';

if numel(client_ids) ~= expected_count
    error('validate_client_mapping:CountMismatch', ...
        '%s 包含 %d 个客户端，但期望 %d 个。', context, numel(client_ids), expected_count);
end
if numel(unique(client_ids)) ~= numel(client_ids)
    error('validate_client_mapping:DuplicateClient', '%s 中存在重复客户端编号。', context);
end
if any(~ismember(client_ids, valid_client_ids))
    invalid_ids = client_ids(~ismember(client_ids, valid_client_ids));
    error('validate_client_mapping:InvalidClientId', ...
        '%s 中存在非法客户端编号：%s。', context, mat2str(unique(invalid_ids)));
end
end


function values = collect_numeric_leaves(value)
%COLLECT_NUMERIC_LEAVES 递归收集嵌套 cell 中所有非空数值叶子。

values = [];
if iscell(value)
    for index = 1:numel(value)
        values = [values, collect_numeric_leaves(value{index})]; %#ok<AGROW>
    end
elseif isnumeric(value) && ~isempty(value)
    values = value(:)';
elseif ~isempty(value)
    error('validate_client_mapping:UnsupportedMappingType', ...
        '映射中出现不支持的类型：%s。', class(value));
end
end

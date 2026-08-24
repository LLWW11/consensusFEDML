function config = validate_config(config)
%VALIDATE_CONFIG 合并默认值并校验 TopoOptV2 配置。
%   CONFIG 可以只提供需要覆盖的字段。函数会补齐默认字段，同时保留
%   其他模块使用的扩展字段，便于搜索器和求解器共享同一配置结构体。

defaults = topooptv2.default_config();
if nargin < 1 || isempty(config)
    config = defaults;
    return;
end

if ~isstruct(config) || ~isscalar(config)
    error('topooptv2:validate_config:InvalidConfig', ...
        'config 必须是标量结构体。');
end

config = merge_default_fields(config, defaults);
validate_schema_version(config.schema_version);

nonnegativeFields = { ...
    'communication_cost', 'storage_cost', 'compute_cost', ...
    'aggregate_output_cost', 'dissipation_cost', ...
    'communication_delay', 'storage_delay', 'compute_delay', ...
    'aggregate_output_delay', 'dissipation_delay'};
for fieldIndex = 1:numel(nonnegativeFields)
    fieldName = nonnegativeFields{fieldIndex};
    validate_finite_scalar(config.(fieldName), fieldName, 0, true);
end

validate_positive_integer(config.storage_capacity, 'storage_capacity', 1);
% 至少两个输入流量才能形成真实聚合，计算容量因此不得小于 2。
validate_positive_integer(config.compute_capacity, 'compute_capacity', 2);

weightFields = { ...
    'utility_client_weight', 'utility_cost_weight', ...
    'utility_delay_weight'};
for fieldIndex = 1:numel(weightFields)
    fieldName = weightFields{fieldIndex};
    validate_finite_scalar(config.(fieldName), fieldName, -inf, true);
end
validate_finite_scalar(config.tolerance, 'tolerance', 0, false);

if config.dissipation_cost <= config.aggregate_output_cost
    error('topooptv2:validate_config:InvalidDissipationCost', ...
        'dissipation_cost 必须严格大于 aggregate_output_cost。');
end
end


function config = merge_default_fields(config, defaults)
%MERGE_DEFAULT_FIELDS 将缺失的标准配置字段补为默认值。

defaultNames = fieldnames(defaults);
for fieldIndex = 1:numel(defaultNames)
    fieldName = defaultNames{fieldIndex};
    if ~isfield(config, fieldName) || isempty(config.(fieldName))
        config.(fieldName) = defaults.(fieldName);
    end
end
end


function validate_schema_version(schemaVersion)
%VALIDATE_SCHEMA_VERSION 校验配置模式版本为非空文本。

isTextScalar = (ischar(schemaVersion) && isrow(schemaVersion)) || ...
    (isstring(schemaVersion) && isscalar(schemaVersion));
if ~isTextScalar || strlength(string(schemaVersion)) == 0
    error('topooptv2:validate_config:InvalidSchemaVersion', ...
        'schema_version 必须是非空文本标量。');
end
end


function validate_positive_integer(value, fieldName, minimumValue)
%VALIDATE_POSITIVE_INTEGER 校验配置字段为不小于阈值的整数标量。

validate_finite_scalar(value, fieldName, minimumValue, true);
if value ~= floor(value)
    error('topooptv2:validate_config:InvalidInteger', ...
        '%s 必须是整数标量。', fieldName);
end
end


function validate_finite_scalar(value, fieldName, minimumValue, inclusive)
%VALIDATE_FINITE_SCALAR 校验配置字段为有限实数标量并满足下界。

if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ~isfinite(value)
    error('topooptv2:validate_config:InvalidScalar', ...
        '%s 必须是有限实数标量。', fieldName);
end
if inclusive
    violatesBound = value < minimumValue;
else
    violatesBound = value <= minimumValue;
end
if violatesBound
    if inclusive
        relationText = '不小于';
    else
        relationText = '大于';
    end
    error('topooptv2:validate_config:InvalidRange', ...
        '%s 必须%s %g。', fieldName, relationText, minimumValue);
end
end

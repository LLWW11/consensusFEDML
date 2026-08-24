function [TSML_BdwMat, adj_mat, num_of_nodes, sampling_info] = gen_random_tsmlg_v3(TopoOption, ...
    num_of_nodes, num_layers, num_wave, percent, variance_control_mode, ...
    isControl, varAlpha)
%GEN_RANDOM_TSMLG_V3 生成时隙多层图的链路波长容量。
%   legacy 模式保持历史随机生成方式；paired_exact 模式对每层的
%   “有向链路×波长槽位”执行完整随机排序，并精确激活目标数量的槽位。
%   isControl=true 时，varAlpha 表示每层链路容量目标方差与原方差之比。
%   旧调用省略新增参数时不执行链路容量方差收缩。

if nargin < 6 || isempty(variance_control_mode)
    variance_control_mode = 'legacy';
end
if nargin < 7 || isempty(isControl)
    isControl = false;
end
if nargin < 8 || isempty(varAlpha)
    varAlpha = 1;
end
variance_control_mode = validatestring(variance_control_mode, ...
    {'legacy', 'paired_exact'}, mfilename, 'variance_control_mode');
validate_generation_parameters(num_layers, num_wave, percent);
isControl = validate_variance_control_parameters( ...
    variance_control_mode, isControl, varAlpha);

if strcmp(TopoOption,'6node')
    num_of_nodes = 6;
    adj_mat = [                       %   6 node simple topology
        0,1,0,0,0,1;
        1,0,1,0,0,1;
        0,1,0,1,1,0;
        0,0,1,0,1,0;
        0,0,1,1,0,1;
        1,1,0,0,1,0;];
    
elseif strcmp(TopoOption,'NSFNET')
    num_of_nodes = 14;
    adj_mat = [                       %14 node NSF
        0	1	1	1	0	0	0	0	0	0	0	0	0	0;
        1	0	1	0	0	0	0	1	0	0	0	0	0	0;
        1	1	0	0	0	1	0	0	0	0	0	0	0	0;
        1	0	0	0	1	0	0	0	0	1	0	0	0	0;
        0	0	0	1	0	1	1	0	0	0	0	0	0	0;
        0	0	1	0	1	0	0	0	1	0	0	0	1	0;
        0	0	0	0	1	0	0	1	0	0	0	0	0	0;
        0	1	0	0	0	0	1	0	0	0	1	0	0	0;
        0	0	0	0	0	1	0	0	0	0	1	0	0	0;
        0	0	0	1	0	0	0	0	0	0	0	1	0	1;
        0	0	0	0	0	0	0	1	1	0	0	1	0	1;
        0	0	0	0	0	0	0	0	0	1	1	0	1	0;
        0	0	0	0	0	1	0	0	0	0	0	1	0	1;
        0	0	0	0	0	0	0	0	0	1	1	0	1	0;
        ];
    
elseif strcmp(TopoOption,'OPEN')
    num_of_nodes = 19;
    adj_mat = [    %19-node 39-link OPEN network
        0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
        1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0;
        1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1;
        0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1;
        0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0;
        0,0,0,1,1,0,1,0,0,1,1,0,0,0,0,0,0,1,1;
        0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0;
        0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0;
        0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0;
        0,0,0,0,0,1,1,1,1,0,1,0,0,0,0,0,0,0,0;
        0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,1,1,1,0;
        0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0;
        0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,0,0;
        0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0;
        0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0;
        0,1,1,0,0,0,0,0,0,0,1,0,1,1,1,0,1,0,0;
        0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0;
        0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1;
        0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0;
        ];
elseif strcmp(TopoOption,'USNET')
    num_of_nodes = 24;
    adj_mat = [      %24-node 43-link USNET network
        0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
        1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
        0,1,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
        0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
        0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
        1,1,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0;
        0,0,1,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
        0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
        0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0;
        0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0;
        0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0;
        0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0;
        0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0;
        0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0;
        0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0;
        0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,1,0,0;
        0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,1,0;
        0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1;
        0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0;
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0;
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0;
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,1,0;
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1;
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0;
        ];
elseif strcmp(TopoOption,'Metro')
    num_of_nodes = 38;
    adj_mat = [      %38-node network
        0	1	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
        1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1;
        0	0	0	1	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
        0	0	1	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
        0	0	0	0	0	1	0	0	0	1	0	1	0	0	0	0	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
        0	0	0	0	1	0	1	0	0	0	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
        0	0	1	0	0	1	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1;
        1	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	1;
        0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	1	0	0	1	0	1	0	1	1	0	0	0	0;
        0	0	0	0	1	0	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
        0	0	0	1	0	1	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
        0	0	0	0	1	1	0	0	0	0	0	0	1	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
        0	0	0	0	0	0	0	0	0	0	1	1	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
        0	0	0	0	0	0	0	0	0	0	0	0	1	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
        0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
        0	0	0	0	0	0	0	0	0	0	0	1	0	0	1	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0;
        0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0;
        0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	1	0	1	0	0	0	0	0	1	1	0	0	0	0	0	0	0	0	0	0	0	0;
        0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	1	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0;
        0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	1	0	0	0	0	0	0	1	0	0	0	0	0	0	0;
        0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0;
        0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	1	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0;
        0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	1	0	0	0	0	0	0	0	1	0	0	0	0	0	0;
        0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
        0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
        0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0;
        0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	1	0	0	0	0	0	0	0	0	1	0	1	1	0	0	0	0	0	0	0	0	0;
        0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	1	0;
        0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	1	0	0	0;
        0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
        0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0;
        0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	1	0	1	0	0	0	0	0;
        0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	1	0	0	0	0;
        0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	1	1	0	0;
        0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	1	0	0	0	0;
        0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	1	0;
        0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	1	0	0;
        0	1	0	0	0	0	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
        ];
else % 如果不是预定义选项,生成一个随机的邻接矩阵，且确保每一个节点都不是孤立的
    degree_node = 0;
    while ~(isempty(find(degree_node <= 1, 1)))
        tmp_mat = randi(num_of_nodes,num_of_nodes);
        weight_mat = tril(tmp_mat,-1)+triu(tmp_mat',1);
        adj_mat = mod(weight_mat,2);
        degree_node = sum(adj_mat);
    end
end

degree_node = sum(adj_mat);
edge_num = sum(degree_node);   %不清楚为什么不用除以2
links = find(adj_mat > 0);

if strcmp(variance_control_mode, 'paired_exact')
    raw_link_state = generate_exact_link_state( ...
        num_layers, edge_num, num_wave, percent);
elseif num_wave == 1
    %生成一个num_layers行、edge_num列的随机整数矩阵，矩阵元素取值范围是[0, edge_num]
    raw_link_state = randi([0, edge_num], num_layers, edge_num);
    active_mask = raw_link_state <= round(edge_num * percent);
    raw_link_state = zeros(num_layers, edge_num);
    raw_link_state(active_mask) = 1;
else
    raw_link_state = zeros(num_layers, edge_num);
    for i=1:num_layers
        for j=1:edge_num
            raw_link_state(i,j) = gen_rand_num(percent);
        end
    end
    raw_link_state = round(raw_link_state.*num_wave);
end

if isControl
    [link_state, topology_variance_audit] = control_link_state_variance( ...
        raw_link_state, varAlpha, num_wave);
else
    link_state = raw_link_state;
    topology_variance_audit = build_disabled_variance_audit( ...
        raw_link_state, varAlpha, num_wave);
end

tsmlg = zeros(num_of_nodes,num_of_nodes,num_layers);
for i = 1:num_layers
    tsmlg(links + (i-1).*(num_of_nodes.^2)) = link_state(i,:);
end
TSML_BdwMat = tsmlg;

sampling_info = build_sampling_info( ...
    variance_control_mode, percent, raw_link_state, link_state, edge_num, ...
    num_wave, isControl, varAlpha, topology_variance_audit);

end


function isControl = validate_variance_control_parameters( ...
        variance_control_mode, isControl, varAlpha)
%VALIDATE_VARIANCE_CONTROL_PARAMETERS 校验拓扑方差控制开关和比例。

if ~(islogical(isControl) && isscalar(isControl)) && ...
        ~(isnumeric(isControl) && isscalar(isControl) && ...
        isfinite(isControl) && any(isControl == [0, 1]))
    error('gen_random_tsmlg_v3:InvalidControlFlag', ...
        'isControl 必须是逻辑标量或数值 0/1。');
end
isControl = logical(isControl);
if ~isnumeric(varAlpha) || ~isscalar(varAlpha) || ~isfinite(varAlpha) || ...
        varAlpha < 0 || varAlpha > 1
    error('gen_random_tsmlg_v3:InvalidVarAlpha', ...
        'varAlpha 必须是 [0,1] 范围内的有限数值标量。');
end
if isControl && ~strcmp(variance_control_mode, 'paired_exact')
    error('gen_random_tsmlg_v3:UnsupportedControlMode', ...
        '拓扑方差控制仅支持 paired_exact 生成模式。');
end
end


function validate_generation_parameters(num_layers, num_wave, percent)
%VALIDATE_GENERATION_PARAMETERS 校验层数、波长数和可用比例参数。

if ~isnumeric(num_layers) || ~isscalar(num_layers) || num_layers < 1 || ...
        num_layers ~= floor(num_layers)
    error('gen_random_tsmlg_v3:InvalidLayerCount', ...
        'num_layers 必须是正整数。');
end
if ~isnumeric(num_wave) || ~isscalar(num_wave) || num_wave < 1 || ...
        num_wave ~= floor(num_wave)
    error('gen_random_tsmlg_v3:InvalidWaveCount', ...
        'num_wave 必须是正整数。');
end
if ~isnumeric(percent) || ~isscalar(percent) || ~isfinite(percent) || ...
        percent < 0 || percent > 1
    error('gen_random_tsmlg_v3:InvalidAvailability', ...
        'percent 必须是 [0,1] 范围内的有限数值标量。');
end
end


function link_state = generate_exact_link_state( ...
        num_layers, edge_num, num_wave, percent)
%GENERATE_EXACT_LINK_STATE 按精确槽位配额生成每层链路容量。

slot_count = edge_num * num_wave;
active_slot_count = round(slot_count * percent);
link_state = zeros(num_layers, edge_num);

for layer_index = 1:num_layers
    % 始终生成完整排列，保证不同 percent 消耗相同数量的随机数。
    slot_order = randperm(slot_count);
    active_slots = zeros(1, slot_count);
    active_slots(slot_order(1:active_slot_count)) = 1;
    slot_matrix = reshape(active_slots, num_wave, edge_num);
    link_state(layer_index, :) = sum(slot_matrix, 1);
end
end


function [controlled_state, audit] = control_link_state_variance( ...
        raw_state, varAlpha, num_wave)
%CONTROL_LINK_STATE_VARIANCE 逐层控制链路波长容量方差。

[num_layers, edge_num] = size(raw_state);
controlled_state = zeros(size(raw_state));
audit = initialize_topology_variance_audit(num_layers, true, varAlpha);

for layer_index = 1:num_layers
    raw_layer = double(raw_state(layer_index, :));
    target_sum = sum(raw_layer);
    raw_variance = var(raw_layer, 0);
    minimum_layer = minimum_variance_integer_vector( ...
        target_sum, edge_num, raw_layer);
    minimum_variance = var(minimum_layer, 0);
    requested_variance = varAlpha * raw_variance;
    effective_target = max(requested_variance, minimum_variance);

    if varAlpha == 1 || raw_variance == 0
        controlled_layer = raw_layer;
        selected_factor = 1;
    else
        [controlled_layer, selected_factor] = search_layer_shrink_factor( ...
            raw_layer, target_sum, effective_target, num_wave);
    end

    controlled_state(layer_index, :) = controlled_layer;
    audit = record_layer_audit(audit, layer_index, raw_layer, ...
        controlled_layer, requested_variance, effective_target, ...
        minimum_variance, selected_factor, num_wave);
end
end


function audit = build_disabled_variance_audit(raw_state, varAlpha, num_wave)
%BUILD_DISABLED_VARIANCE_AUDIT 构建未启用方差收缩时的逐层审计。

num_layers = size(raw_state, 1);
audit = initialize_topology_variance_audit(num_layers, false, varAlpha);
for layer_index = 1:num_layers
    raw_layer = double(raw_state(layer_index, :));
    raw_variance = var(raw_layer, 0);
    minimum_layer = minimum_variance_integer_vector( ...
        sum(raw_layer), numel(raw_layer), raw_layer);
    audit = record_layer_audit(audit, layer_index, raw_layer, raw_layer, ...
        raw_variance, raw_variance, var(minimum_layer, 0), 1, num_wave);
end
end


function audit = initialize_topology_variance_audit( ...
        num_layers, is_enabled, varAlpha)
%INITIALIZE_TOPOLOGY_VARIANCE_AUDIT 初始化拓扑方差控制审计结构。

audit = struct();
audit.schema_version = '1.0';
audit.is_enabled = is_enabled;
audit.requested_var_alpha = varAlpha;
numeric_fields = { ...
    'raw_variance_by_layer', 'requested_variance_by_layer', ...
    'effective_target_variance_by_layer', 'minimum_integer_variance_by_layer', ...
    'achieved_variance_by_layer', 'achieved_alpha_by_layer', ...
    'selected_shrink_factor_by_layer', 'raw_slot_sum_by_layer', ...
    'controlled_slot_sum_by_layer'};
logical_fields = { ...
    'requested_target_above_integer_floor_by_layer', ...
    'target_reached_within_tolerance_by_layer', ...
    'slot_sum_preserved_by_layer', 'all_integer_by_layer', ...
    'all_in_range_by_layer'};
for field_index = 1:numel(numeric_fields)
    audit.(numeric_fields{field_index}) = zeros(num_layers, 1);
end
for field_index = 1:numel(logical_fields)
    audit.(logical_fields{field_index}) = false(num_layers, 1);
end
end


function integer_vector = minimum_variance_integer_vector( ...
        target_sum, element_count, reference_vector)
%MINIMUM_VARIANCE_INTEGER_VECTOR 构造固定整数总和下的最小方差向量。

lower_value = floor(target_sum / element_count);
upper_count = round(target_sum - lower_value * element_count);
integer_vector = lower_value * ones(1, element_count);
if upper_count > 0
    [~, order] = sort(reference_vector, 'descend');
    integer_vector(order(1:upper_count)) = lower_value + 1;
end
end


function [best_layer, best_factor] = search_layer_shrink_factor( ...
        raw_layer, target_sum, target_variance, num_wave)
%SEARCH_LAYER_SHRINK_FACTOR 搜索最接近目标方差的容量收缩系数。

lower_factor = 0;
upper_factor = 1;
[best_layer, best_variance] = evaluate_layer_factor( ...
    raw_layer, target_sum, lower_factor, num_wave);
best_factor = lower_factor;
best_error = abs(best_variance - target_variance);

[candidate, candidate_variance] = evaluate_layer_factor( ...
    raw_layer, target_sum, upper_factor, num_wave);
[best_layer, best_factor, best_error] = update_layer_candidate( ...
    best_layer, best_factor, best_error, candidate, upper_factor, ...
    candidate_variance, target_variance);

for iteration = 1:64
    middle_factor = (lower_factor + upper_factor) / 2;
    [candidate, candidate_variance] = evaluate_layer_factor( ...
        raw_layer, target_sum, middle_factor, num_wave);
    [best_layer, best_factor, best_error] = update_layer_candidate( ...
        best_layer, best_factor, best_error, candidate, middle_factor, ...
        candidate_variance, target_variance);
    if candidate_variance < target_variance
        lower_factor = middle_factor;
    else
        upper_factor = middle_factor;
    end
end
end


function [candidate, candidate_variance] = evaluate_layer_factor( ...
        raw_layer, target_sum, shrink_factor, num_wave)
%EVALUATE_LAYER_FACTOR 计算一个收缩系数对应的整数容量和方差。

layer_mean = mean(raw_layer);
continuous = layer_mean + shrink_factor .* (raw_layer - layer_mean);
continuous = min(max(continuous, 0), num_wave);
candidate = balanced_round_layer(continuous, target_sum, num_wave);
candidate_variance = var(candidate, 0);
end


function rounded = balanced_round_layer(values, target_sum, num_wave)
%BALANCED_ROUND_LAYER 在保持层总槽位数的前提下执行平衡整数舍入。

rounded = floor(values);
remaining = round(target_sum - sum(rounded));
fractional = values - rounded;
if remaining > 0
    eligible = find(rounded < num_wave);
    [~, order] = sort(fractional(eligible), 'descend');
    selected = eligible(order(1:remaining));
    rounded(selected) = rounded(selected) + 1;
elseif remaining < 0
    eligible = find(rounded > 0);
    [~, order] = sort(fractional(eligible), 'ascend');
    selected = eligible(order(1:(-remaining)));
    rounded(selected) = rounded(selected) - 1;
end
if sum(rounded) ~= target_sum
    error('gen_random_tsmlg_v3:SlotSumNotPreserved', ...
        '平衡整数舍入未能保持层总槽位数。');
end
end


function [best_layer, best_factor, best_error] = update_layer_candidate( ...
        best_layer, best_factor, best_error, candidate, candidate_factor, ...
        candidate_variance, target_variance)
%UPDATE_LAYER_CANDIDATE 更新最接近目标方差的链路容量候选。

candidate_error = abs(candidate_variance - target_variance);
if candidate_error < best_error
    best_layer = candidate;
    best_factor = candidate_factor;
    best_error = candidate_error;
end
end


function audit = record_layer_audit(audit, layer_index, raw_layer, ...
        controlled_layer, requested_variance, effective_target, ...
        minimum_variance, selected_factor, num_wave)
%RECORD_LAYER_AUDIT 记录一层拓扑方差控制的数值与约束状态。

raw_variance = var(raw_layer, 0);
achieved_variance = var(controlled_layer, 0);
if raw_variance == 0
    achieved_alpha = 0;
else
    achieved_alpha = achieved_variance / raw_variance;
end
target_tolerance = max(1e-12, 0.05 * max(raw_variance, effective_target));
audit.raw_variance_by_layer(layer_index) = raw_variance;
audit.requested_variance_by_layer(layer_index) = requested_variance;
audit.effective_target_variance_by_layer(layer_index) = effective_target;
audit.minimum_integer_variance_by_layer(layer_index) = minimum_variance;
audit.achieved_variance_by_layer(layer_index) = achieved_variance;
audit.achieved_alpha_by_layer(layer_index) = achieved_alpha;
audit.selected_shrink_factor_by_layer(layer_index) = selected_factor;
audit.raw_slot_sum_by_layer(layer_index) = sum(raw_layer);
audit.controlled_slot_sum_by_layer(layer_index) = sum(controlled_layer);
audit.requested_target_above_integer_floor_by_layer(layer_index) = ...
    requested_variance >= minimum_variance - 1e-12;
audit.target_reached_within_tolerance_by_layer(layer_index) = ...
    abs(achieved_variance - effective_target) <= target_tolerance;
audit.slot_sum_preserved_by_layer(layer_index) = ...
    sum(raw_layer) == sum(controlled_layer);
audit.all_integer_by_layer(layer_index) = ...
    all(controlled_layer == round(controlled_layer));
audit.all_in_range_by_layer(layer_index) = ...
    all(controlled_layer >= 0 & controlled_layer <= num_wave);
end


function sampling_info = build_sampling_info( ...
        variance_control_mode, percent, raw_link_state, link_state, ...
        edge_num, num_wave, isControl, varAlpha, topology_variance_audit)
%BUILD_SAMPLING_INFO 构建拓扑槽位采样的可审计摘要。

slot_count_per_layer = edge_num * num_wave;
active_slot_count_by_layer = sum(link_state, 2);
sampling_info = struct();
sampling_info.mode = variance_control_mode;
sampling_info.is_variance_control_enabled = isControl;
sampling_info.requested_topology_var_alpha = varAlpha;
sampling_info.effective_topology_var_alpha = 1;
if isControl
    sampling_info.effective_topology_var_alpha = varAlpha;
end
sampling_info.target_available_fraction = percent;
sampling_info.edge_entry_count = edge_num;
sampling_info.wave_count = num_wave;
sampling_info.slot_count_per_layer = slot_count_per_layer;
sampling_info.raw_active_slot_count_by_layer = sum(raw_link_state, 2);
sampling_info.active_slot_count_by_layer = active_slot_count_by_layer;
sampling_info.actual_available_fraction_by_layer = ...
    active_slot_count_by_layer ./ slot_count_per_layer;
sampling_info.topology_variance_audit = topology_variance_audit;
end
% function [TSML_BdwMat, adj_mat, num_of_nodes] = gen_random_tsmlg_v3(TopoOption,num_of_nodes,num_layers,num_wave,percent)


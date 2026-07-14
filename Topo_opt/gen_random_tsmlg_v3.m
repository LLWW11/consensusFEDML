function [TSML_BdwMat, adj_mat, num_of_nodes, sampling_info] = gen_random_tsmlg_v3(TopoOption, ...
    num_of_nodes, num_layers, num_wave, percent, variance_control_mode)
%GEN_RANDOM_TSMLG_V3 生成时隙多层图的链路波长容量。
%   legacy 模式保持历史随机生成方式；paired_exact 模式对每层的
%   “有向链路×波长槽位”执行完整随机排序，并精确激活目标数量的槽位。
%   当不同利用率复用同一随机种子时，paired_exact 会生成单调嵌套容量。

if nargin < 6 || isempty(variance_control_mode)
    variance_control_mode = 'legacy';
end
variance_control_mode = validatestring(variance_control_mode, ...
    {'legacy', 'paired_exact'}, mfilename, 'variance_control_mode');
validate_generation_parameters(num_layers, num_wave, percent);

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
        edge_num = sum(degree_node);
    end
end

degree_node = sum(adj_mat);
edge_num = sum(degree_node);   %不清楚为什么不用除以2
links = find(adj_mat > 0);

if strcmp(variance_control_mode, 'paired_exact')
    link_state = generate_exact_link_state( ...
        num_layers, edge_num, num_wave, percent);
elseif num_wave == 1
    %生成一个num_layers行、edge_num列的随机整数矩阵，矩阵元素取值范围是[0, edge_num]
    link_state = randi([0, edge_num], num_layers, edge_num);
    tmp = find(link_state<=round(edge_num*percent));
    link_state = zeros(num_layers, edge_num);
    link_state(tmp) = 1;
else
    for i=1:num_layers
        for j=1:edge_num
            link_state(i,j) = gen_rand_num(percent);
        end
    end
    link_state = round(link_state.*num_wave);
end

tsmlg = zeros(num_of_nodes,num_of_nodes,num_layers);
for i = 1:num_layers
    tsmlg(links + (i-1).*(num_of_nodes.^2)) = link_state(i,:);
end
TSML_BdwMat = tsmlg;

sampling_info = build_sampling_info( ...
    variance_control_mode, percent, link_state, edge_num, num_wave);

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


function sampling_info = build_sampling_info( ...
        variance_control_mode, percent, link_state, edge_num, num_wave)
%BUILD_SAMPLING_INFO 构建拓扑槽位采样的可审计摘要。

slot_count_per_layer = edge_num * num_wave;
active_slot_count_by_layer = sum(link_state, 2);
sampling_info = struct();
sampling_info.mode = variance_control_mode;
sampling_info.target_available_fraction = percent;
sampling_info.edge_entry_count = edge_num;
sampling_info.wave_count = num_wave;
sampling_info.slot_count_per_layer = slot_count_per_layer;
sampling_info.active_slot_count_by_layer = active_slot_count_by_layer;
sampling_info.actual_available_fraction_by_layer = ...
    active_slot_count_by_layer ./ slot_count_per_layer;
end
% function [TSML_BdwMat, adj_mat, num_of_nodes] = gen_random_tsmlg_v3(TopoOption,num_of_nodes,num_layers,num_wave,percent)


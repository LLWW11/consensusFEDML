function [group_num_HFLSnF, client_num_HFLSnF, max_layer, actual_c2e_map, DynEdgeSet,...
     client_num_FLnoSnF, max_layer_FLnoSnF, c2cmap_FLnoSnF, client_num_FLSnF, max_layer_FLSnF, c2cmap_FLSnF,...
     group_num_HFLnoSnF, client_num_HFLnoSnF, max_layer_HFLnoSnF, actual_c2e_map_HFLnoSnF, ...
     DynEdgeSet_HFLnoSnF, topology_sampling_info] = varParaHFL_TSMLG_v10(TopoOption, ...
    num_layers, num_of_nodes, num_wave, util, EdgeSet_fixed, Cloud, mean_time_interval, ...
    duration, topology_seed, variance_control_mode, isControl, varAlpha)
%VARPARAHFL_TSMLG_V10 在同一动态拓扑上计算 FL/HFL 与 SnF/noSnF 对照。
%   topology_seed 用于保证每个 (epoch, util) 网络快照可重复。返回的第 1 项策略
%   表示动态选边，第 3 项策略表示固定边缘集合。variance_control_mode 可取
%   legacy 或 paired_exact。isControl 和 varAlpha 控制拓扑链路容量方差，
%   省略新增参数时不执行方差收缩，保证旧调用行为不变。

if nargin < 11 || isempty(variance_control_mode)
    variance_control_mode = 'legacy';
end
if nargin < 12 || isempty(isControl)
    isControl = false;
end
if nargin < 13 || isempty(varAlpha)
    varAlpha = 1;
end

if nargin >= 10 && ~isempty(topology_seed)
    rng(double(topology_seed), 'twister');
end

% tsmlg2adj 将该值作为层间边容量：0 禁用跨层存储，inf 允许存储转发。
Cost_of_TemporalLinks = inf;
% TopoOption = '6node';
% num_layers = 2;
% num_of_nodes = 6;
% num_wave = 5;
% util = 0.8;
MinHop=1;
MaxHop=4;
TargetHop=2;

percent = 1- util;
% Cost_of_TemporalLinks = inf;


[TSML_BdwMat_orig, adj_mat, num_of_nodes, topology_sampling_info] = ...
    gen_random_tsmlg_v3(TopoOption, num_of_nodes, num_layers, num_wave, ...
    percent, variance_control_mode, isControl, varAlpha);
% Cloud node cannot be a client or edge node
% The other nodes can be a client and an edge node simultaneously
ClientSet = 1:num_of_nodes;
ClientSet(ClientSet == Cloud) = [];

[time]= gen_tsmlg_time_axis_v1(num_layers, mean_time_interval); %exprnd(mean_time_interval),time(1)=1;

total_TSML_BdwMat = zeros (num_of_nodes,num_of_nodes,num_layers);
deadline=time(end) - time(1);
while  ~all(TSML_BdwMat_orig(:) == 0)
    tmp_TSML_BdwMat_orig = double(TSML_BdwMat_orig >= 1);
    [TSML_IdleTime] = UpdataIdleTime_v3(tmp_TSML_BdwMat_orig, time, num_of_nodes, deadline, num_layers);
    AuxTSML = TSML_IdleTime>=duration;
    
    %[AuxTopo] = tsmlg2adj(AuxTSML, num_of_nodes, num_layers, Cost_of_TemporalLinks);
    TSML_BdwMat = double(AuxTSML);
    total_TSML_BdwMat = total_TSML_BdwMat + TSML_BdwMat;
    TSML_BdwMat_orig = TSML_BdwMat_orig - tmp_TSML_BdwMat_orig;
end

%% 不用改

% HFL-SnF with Fixed EdgeSet
[group_num_HFLSnF_fixed, client_num_HFLSnF_fixed, ...
    max_layer_fixed, actual_c2e_map_fixed] = HFL_SnF_v5(total_TSML_BdwMat, ClientSet, EdgeSet_fixed, Cloud,...
    num_of_nodes, num_layers);
%%
% =========Phase 0: Baselines =========
%% 不用改
% Baseline 1: FL without SnF
num_layers_FLnoSnF=1;
Cost_of_TemporalLinks1=0;
[client_num_FLnoSnF, max_layer_FLnoSnF, c2cmap_FLnoSnF] = CommonFL_v3(total_TSML_BdwMat, ...
    num_layers_FLnoSnF, num_of_nodes, Cost_of_TemporalLinks1, util, Cloud);

% Baseline 2: FL with SnF
num_layers_FLSnF=num_layers;
Cost_of_TemporalLinks2=Cost_of_TemporalLinks;
[client_num_FLSnF, max_layer_FLSnF, c2cmap_FLSnF] = CommonFL_v3(total_TSML_BdwMat, ...
    num_layers_FLSnF, num_of_nodes, Cost_of_TemporalLinks2, util, Cloud);

% Baseline 3: HFL without SnF fixed EdgeSet
num_layers_HFLnoSnF = 1;
[group_num_HFLnoSnF_fix, client_num_HFLnoSnF_fix, ...
    max_layer_HFLnoSnF_fix, actual_c2e_map_HFLnoSnF_fix] = HFL_SnF_v5(total_TSML_BdwMat, ClientSet, EdgeSet_fixed, Cloud,...
    num_of_nodes, num_layers_HFLnoSnF);


%% 记得改一下这里的搜索方案
% Baseline 4: HFL without SnF with Los Dyn Edge Election
num_layers_HFLnoSnF_los = 1;
[OptimEdgeSet_HFLnoSnF,~] = LocalSearch_EdgeSet_v6(total_TSML_BdwMat, adj_mat, Cloud, ClientSet,...
    num_of_nodes, num_layers_HFLnoSnF, MinHop, MaxHop, TargetHop);
[group_num_HFLnoSnF_los, client_num_HFLnoSnF_los, ...
    max_layer_HFLnoSnF_los, actual_c2e_map_HFLnoSnF_los] = HFL_SnF_v5( ...
    total_TSML_BdwMat, ClientSet, OptimEdgeSet_HFLnoSnF, Cloud,...
    num_of_nodes, num_layers_HFLnoSnF_los);
% %% 不用改
%
% % HFL-SnF with Fixed EdgeSet
% [group_num_HFLSnF_fixed, client_num_HFLSnF_fixed, ...
%     max_layer_fixed, actual_c2e_map_fixed] = HFL_SnF_v5(total_TSML_BdwMat, ClientSet, EdgeSet_fixed, Cloud,...
%     num_of_nodes, num_layers);


%% 记得改一下这里的搜索方案
% HFL-SnF with Los Dyn Edge Election
[OptimEdgeSet_HFLSnF,~] = LocalSearch_EdgeSet_v6(total_TSML_BdwMat, adj_mat, Cloud, ClientSet,...
    num_of_nodes, num_layers, MinHop, MaxHop, TargetHop);
[group_num_HFLSnF_los, client_num_HFLSnF_los, ...
    max_layer_los, actual_c2e_map_los] = HFL_SnF_v5( ...
    total_TSML_BdwMat, ClientSet, OptimEdgeSet_HFLSnF, Cloud,...
    num_of_nodes, num_layers);
%%

% %使用超图的版本
% [group_num_hyper, client_num_hyper, max_layer_hyper, mapping_hyper] = HFL_SnF_hypergraph(...
%     TSML_BdwMat, ClientSet, EdgeSet, Cloud, num_nodes, num_layers);


group_num_HFLnoSnF{1,1} = group_num_HFLnoSnF_los;
% group_num_HFLSnF{1,2} = group_num_HFLSnF_los1;
group_num_HFLnoSnF{1,3} = group_num_HFLnoSnF_fix;

client_num_HFLnoSnF{1,1} = client_num_HFLnoSnF_los;
% client_num_HFLSnF{1,2} = client_num_HFLSnF_los1;
client_num_HFLnoSnF{1,3} = client_num_HFLnoSnF_fix;

group_num_HFLSnF{1,1} = group_num_HFLSnF_los;
% group_num_HFLSnF{1,2} = group_num_HFLSnF_los1;
group_num_HFLSnF{1,3} = group_num_HFLSnF_fixed;
% group_num_HFLSnF{1,4} = group_num_HFLSnF_P4;
% group_num_HFLSnF{1,5} = group_num_HFLSnF_P5;

client_num_HFLSnF{1,1} = client_num_HFLSnF_los;
% client_num_HFLSnF{1,2} = client_num_HFLSnF_los1;
client_num_HFLSnF{1,3} = client_num_HFLSnF_fixed;
% client_num_HFLSnF{1,4} = client_num_HFLSnF_P4;
% client_num_HFLSnF{1,5} = client_num_HFLSnF_P5;

max_layer{1,1} = max_layer_los;
% max_layer{1,2} = max_layer_los1;
max_layer{1,3} = max_layer_fixed;
% max_layer{1,4} = max_layer_P4;

% max_layer{1,5} = max_layer_P5;
max_layer_HFLnoSnF{1,1} = max_layer_HFLnoSnF_los;
% max_layer{1,2} = max_layer_los1;
max_layer_HFLnoSnF{1,3} = max_layer_HFLnoSnF_fix;

actual_c2e_map_HFLnoSnF{1,1} = actual_c2e_map_HFLnoSnF_los;
actual_c2e_map_HFLnoSnF{1,3} = actual_c2e_map_HFLnoSnF_fix;

DynEdgeSet_HFLnoSnF{1,1} = OptimEdgeSet_HFLnoSnF;
DynEdgeSet_HFLnoSnF{1,3} = EdgeSet_fixed;

actual_c2e_map{1,1} = actual_c2e_map_los;
% actual_c2e_map{1,2} = actual_c2e_map_los1;
actual_c2e_map{1,3} = actual_c2e_map_fixed;
% actual_c2e_map{1,4} = actual_c2e_map_P4;
% actual_c2e_map{1,5} = actual_c2e_map_P5;

DynEdgeSet{1,1} = OptimEdgeSet_HFLSnF;
% DynEdgeSet{1,2} = OptimEdgeSet1;
DynEdgeSet{1,3} = EdgeSet_fixed;
% DynEdgeSet{1,4} = DynEdgeSet_P4;
% DynEdgeSet{1,5} = DynEdgeSet_P5;



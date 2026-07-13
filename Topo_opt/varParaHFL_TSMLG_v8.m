function [group_num_HFLSnF, client_num_HFLSnF, max_layer, actual_c2e_map, DynEdgeSet,...
     client_num_FLnoSnF, max_layer_FLnoSnF, c2cmap_FLnoSnF, client_num_FLSnF, max_layer_FLSnF, c2cmap_FLSnF,...
     group_num_HFLnoSnF, client_num_HFLnoSnF, max_layer_HFLnoSnF, actual_c2e_map_HFLnoSnF] = varParaHFL_TSMLG_v8(TopoOption, ...
    num_layers, num_of_nodes, num_wave, util, EdgeSet_fixed, Cloud, Cost_of_TemporalLinks, EdgeDynEnable, SelPolicy, FirstEdges, alpha, beta)

% TopoOption = '6node';
% num_layers = 2;
% num_of_nodes = 6;
% num_wave = 5;
% util = 0.8;

percent = 1- util;
% Cost_of_TemporalLinks = inf;

[TSML_BdwMat, ~, num_of_nodes] = gen_random_tsmlg_v3(TopoOption, ...
    num_of_nodes, num_layers, num_wave, percent);
% Cloud node cannot be a client or edge node
% The other nodes can be a client and an edge node simultaneously
ClientSet = 1:num_of_nodes;
x=find(ClientSet == Cloud);
ClientSet(x) = [];

[AuxTopo] = tsmlg2adj(TSML_BdwMat, num_of_nodes, num_layers, Cost_of_TemporalLinks);

%%
% =========Phase 0: Baselines =========
% Baseline 1: FL without SnF
EdgeSet = EdgeSet_fixed;
num_layers_FLnoSnF=1;
Cost_of_TemporalLinks1=0;
[client_num_FLnoSnF, max_layer_FLnoSnF, c2cmap_FLnoSnF] = CommonFL_v3(TSML_BdwMat, ...
    num_layers_FLnoSnF, num_of_nodes, Cost_of_TemporalLinks1, util, Cloud);

% Baseline 2: FL with SnF
num_layers_FLSnF=num_layers;
Cost_of_TemporalLinks2=Cost_of_TemporalLinks;
[client_num_FLSnF, max_layer_FLSnF, c2cmap_FLSnF] = CommonFL_v3(TSML_BdwMat, ...
    num_layers_FLSnF, num_of_nodes, Cost_of_TemporalLinks2, util, Cloud);

% Baseline 3: HFL without SnF
num_layers_HFLnoSnF = 2;
Cost_of_TemporalLinks3=0;
[group_num_HFLnoSnF, client_num_HFLnoSnF, max_layer_HFLnoSnF, actual_c2e_map_HFLnoSnF] = HFL_TSMLG_v3(TSML_BdwMat, ...
    num_layers_HFLnoSnF, num_of_nodes, ...
    util, EdgeSet, Cloud, Cost_of_TemporalLinks3);

%%


SelPolicy = 1;
[group_num_HFLSnF_P1, client_num_HFLSnF_P1, ...
    max_layer_P1, actual_c2e_map_P1, DynEdgeSet_P1] = HFL_SnF_v1(TSML_BdwMat, AuxTopo, ClientSet, EdgeSet_fixed, Cloud,...
    num_of_nodes, num_layers, EdgeDynEnable, SelPolicy, FirstEdges, alpha, beta);

SelPolicy = 2;
[group_num_HFLSnF_P2, client_num_HFLSnF_P2, ...
    max_layer_P2, actual_c2e_map_P2, DynEdgeSet_P2] = HFL_SnF_v1(TSML_BdwMat, AuxTopo, ClientSet, EdgeSet_fixed, Cloud,...
    num_of_nodes, num_layers, EdgeDynEnable, SelPolicy, FirstEdges, alpha, beta);

SelPolicy = 3;
[group_num_HFLSnF_P3, client_num_HFLSnF_P3, ...
    max_layer_P3, actual_c2e_map_P3, DynEdgeSet_P3] = HFL_SnF_v1(TSML_BdwMat, AuxTopo, ClientSet, EdgeSet_fixed, Cloud,...
    num_of_nodes, num_layers, EdgeDynEnable, SelPolicy, FirstEdges, alpha, beta);

SelPolicy = 4;
[group_num_HFLSnF_P4, client_num_HFLSnF_P4, ...
    max_layer_P4, actual_c2e_map_P4, DynEdgeSet_P4] = HFL_SnF_v1(TSML_BdwMat, AuxTopo, ClientSet, EdgeSet_fixed, Cloud,...
    num_of_nodes, num_layers, EdgeDynEnable, SelPolicy, FirstEdges, alpha, beta);

SelPolicy = 5;
[group_num_HFLSnF_P5, client_num_HFLSnF_P5, ...
    max_layer_P5, actual_c2e_map_P5, DynEdgeSet_P5] = HFL_SnF_v1(TSML_BdwMat, AuxTopo, ClientSet, EdgeSet_fixed, Cloud,...
    num_of_nodes, num_layers, EdgeDynEnable, SelPolicy, FirstEdges, alpha, beta);

group_num_HFLSnF{1,1} = group_num_HFLSnF_P1;
group_num_HFLSnF{1,2} = group_num_HFLSnF_P2;
group_num_HFLSnF{1,3} = group_num_HFLSnF_P3;
group_num_HFLSnF{1,4} = group_num_HFLSnF_P4;
group_num_HFLSnF{1,5} = group_num_HFLSnF_P5;

client_num_HFLSnF{1,1} = client_num_HFLSnF_P1;
client_num_HFLSnF{1,2} = client_num_HFLSnF_P2;
client_num_HFLSnF{1,3} = client_num_HFLSnF_P3;
client_num_HFLSnF{1,4} = client_num_HFLSnF_P4;
client_num_HFLSnF{1,5} = client_num_HFLSnF_P5;

max_layer{1,1} = max_layer_P1;
max_layer{1,2} = max_layer_P2;
max_layer{1,3} = max_layer_P3;
max_layer{1,4} = max_layer_P4;
max_layer{1,5} = max_layer_P5;

actual_c2e_map{1,1} = actual_c2e_map_P1;
actual_c2e_map{1,2} = actual_c2e_map_P2;
actual_c2e_map{1,3} = actual_c2e_map_P3;
actual_c2e_map{1,4} = actual_c2e_map_P4;
actual_c2e_map{1,5} = actual_c2e_map_P5;

DynEdgeSet{1,1} = DynEdgeSet_P1;
DynEdgeSet{1,2} = DynEdgeSet_P2;
DynEdgeSet{1,3} = DynEdgeSet_P3;
DynEdgeSet{1,4} = DynEdgeSet_P4;
DynEdgeSet{1,5} = DynEdgeSet_P5;



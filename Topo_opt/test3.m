clear;clc;
Cost_of_TemporalLinks = inf;
TopoOption = 'Metro';
num_layers = 3;
num_of_nodes = 6;
num_wave = 3;
util = 0.7;
MinHop=1; 
MaxHop=4;
TargetHop=2;
Cloud = 18;

percent = 1- util;
% Cost_of_TemporalLinks = inf;
epoch = 1;
while epoch < 100
[TSML_BdwMat, adj_mat, num_of_nodes] = gen_random_tsmlg_v3(TopoOption,num_of_nodes, num_layers, num_wave, percent);
% Cloud node cannot be a client or edge node
% The other nodes can be a client and an edge node simultaneously
ClientSet = 1:num_of_nodes;
x=find(ClientSet == Cloud);
ClientSet(x) = [];

[AuxTopo] = tsmlg2adj(TSML_BdwMat, num_of_nodes, num_layers, Cost_of_TemporalLinks);

%%

%%

% HFL-SnF with Los Dyn Edge Election

    disp(epoch);
    [OptimEdgeSet,~] = LocalSearch_EdgeSet_v3(TSML_BdwMat, adj_mat, Cloud, ClientSet,...
        num_of_nodes, num_layers, MinHop, MaxHop, TargetHop);
        epoch = epoch +1;
end
% [group_num_HFLSnF_los, client_num_HFLSnF_los, ...
%     max_layer_los, actual_c2e_map_los] = HFL_SnF_v3(TSML_BdwMat, ClientSet, OptimEdgeSet, Cloud,...
%     num_of_nodes, num_layers);




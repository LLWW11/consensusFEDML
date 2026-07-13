clear;clc;
TopoOption = 'Metro';
num_of_nodes = 38;
num_layers = 3;
num_wave = 3;
Cloud = 18;
ClientSet = 1:num_of_nodes;
ClientSet(Cloud) = [];
EdgeSet = [6 9 20];
util = 0.8;
percent = 1-util;
[TSML_BdwMat, adj_mat, num_of_nodes] = gen_random_tsmlg_v3(TopoOption, ...
    num_of_nodes, num_layers, num_wave, percent);

% [group_num_HFLSnF, client_num_HFLSnF, ...
%     max_layer, actual_c2e_map] = HFL_SnF_v5(TSML_BdwMat, ClientSet, EdgeSet, Cloud,...
%     num_of_nodes, num_layers);

MinHop=1; 
MaxHop=4; 
TargetHop=2;
epoch = 10;
for i =1:epoch
    disp('Close in the end')
    [OptimEdgeSet,optim_client_num] = LocalSearch_EdgeSet_v4(TSML_BdwMat, adj_mat, Cloud, ClientSet,...
        num_of_nodes, num_layers, MinHop, MaxHop, TargetHop);
    
    disp('Close in the middle')
    [OptimEdgeSet_closemiddle,optim_client_num_closemiddle] = LocalSearch_EdgeSet_close_v4(TSML_BdwMat, adj_mat, Cloud, ClientSet,...
        num_of_nodes, num_layers, MinHop, MaxHop, TargetHop);
    client_num(i,1) = optim_client_num;
    client_num(i,2) = optim_client_num_closemiddle;
    total_OptimEdgeSet{i,1}=OptimEdgeSet;
    total_OptimEdgeSet{i,2}=OptimEdgeSet_closemiddle;

end


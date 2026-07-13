clear;clc;
load('tsml.mat', 'TSML_BdwMat')
num_of_nodes = 38;
num_layers = 3;
Cloud = 18;
ClientSet = 1:num_of_nodes;
ClientSet(Cloud) = [];
EdgeSet = [6 9 20];

[group_num_HFLSnF, client_num_HFLSnF, ...
    max_layer, actual_c2e_map] = HFL_SnF_v5(TSML_BdwMat, ClientSet, EdgeSet, Cloud,...
    num_of_nodes, num_layers);
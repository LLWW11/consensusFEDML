clear;clc;
TopoOption = 'Metro';
num_of_nodes = 38;
num_layers = 3;
num_wave = 3;
Cloud = 18;
ClientSet = 1:num_of_nodes;
ClientSet(Cloud) = [];
EdgeSet = [6 9 20];
util = 0.6;
percent = 1-util;
[TSML_BdwMat, adj_mat, num_of_nodes] = gen_random_tsmlg_v3(TopoOption, ...
    num_of_nodes, num_layers, num_wave, percent);


[group_num_HFLSnF, client_num_HFLSnF, ...
    max_layer, actual_c2e_map] = HFL_SnF_v5(TSML_BdwMat, ClientSet, EdgeSet, Cloud,...
    num_of_nodes, num_layers);
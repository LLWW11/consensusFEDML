function [EdgeSet, Success_flag] = DynEdgeSelection_v1(TSML_BdwMat,num_of_nodes, num_layers, ...
    Cloud, SelPolicy, FirstEdges, alpha, beta)
% clear;clc;
% alpha = 0; % alpha for indgree
% beta = 1 - alpha; % beta for num of all the predecessors
% Cloud = 18;
% TopoOption = 'Metro';
% num_layers = 4;
% num_of_nodes = 6;
% num_wave = 5;
% util = 0.1;
% percent = 1- util;
Cost_of_VirtualLinks = 0.01; % 1./Cost_of_VirtualLinks
% 
% [TSML_BdwMat, ~, num_of_nodes] = gen_random_tsmlg_v3(TopoOption, ...
%     num_of_nodes, num_layers, num_wave, percent);

leftover_mat = zeros(num_of_nodes,num_of_nodes);
for i=1:num_layers
    leftover_mat = leftover_mat + TSML_BdwMat(:,:,i);
end

leftover_MLG = zeros(num_of_nodes,num_of_nodes,2);
leftover_MLG(:,:,1) = leftover_mat;
leftover_MLG(:,:,2) = leftover_mat;

Cloud_second_layer = Cloud + num_of_nodes;
[Aux_Leftover] = tsmlg2adj(leftover_MLG, num_of_nodes, 2, Cost_of_VirtualLinks);
Aux_Leftover(Cloud, Cloud_second_layer) = 0;% clients cannot be aggregated at the Cloud
G_Leftover = digraph(Aux_Leftover);
G_Leftover.Edges.Weight = 1./G_Leftover.Edges.Weight;


% figure(1)
% p = plot(G_Leftover,'EdgeLabel',G_Leftover.Edges.Weight,'Layout','layered');

s = 1:num_of_nodes; % sources = all the clients
s(Cloud) = [];
t = Cloud_second_layer; % destination = the Cloud

TR = shortestpathtree(G_Leftover, s, t);
% highlight(p,TR,'EdgeColor','r')

% FirstEdges = 5;
% SelPolicy = 1;
% disp('SelPolicy 1');
[EdgeSet, Success_flag] = SelEdges_HorizonLevel_v4(TR, ...
    Cloud_second_layer, num_of_nodes, SelPolicy, FirstEdges, alpha, beta);
% figure(2)
% p1 = plot(TR,'EdgeLabel',TR.Edges.Weight,'Layout','layered');
% title('SelPolicy 1');
% if ~isempty(EdgeSet_P1)
%     highlight(p1,EdgeSet_P1,'NodeColor','r')
% end
% 
% disp('SelPolicy 2');
% SelPolicy = 2;
% [EdgeSet_P2, Success_flag_P2] = SelEdges_HorizonLevel_v3(TR, ...
%     Cloud_second_layer, num_of_nodes, SelPolicy, FirstEdges, alpha, beta);
% figure(3)
% p2 = plot(TR,'EdgeLabel',TR.Edges.Weight,'Layout','layered');
% title('SelPolicy 2');
% if ~isempty(EdgeSet_P2)
%     highlight(p2,EdgeSet_P2,'NodeColor','g')
% end
% 
% disp('SelPolicy 3');
% SelPolicy = 3;
% [EdgeSet_P3, Success_flag_P3] = SelEdges_HorizonLevel_v3(TR, ...
%     Cloud_second_layer, num_of_nodes, SelPolicy, FirstEdges, alpha, beta);
% figure(4)
% p3 = plot(TR,'EdgeLabel',TR.Edges.Weight,'Layout','layered');
% title('SelPolicy 3');
% if ~isempty(EdgeSet_P3)
%     highlight(p3,EdgeSet_P3,'NodeColor','b')
% end
% 
% disp('SelPolicy 4');
% SelPolicy = 4;
% [EdgeSet_P4, Success_flag_P4] = SelEdges_HorizonLevel_v3(TR, ...
%     Cloud_second_layer, num_of_nodes, SelPolicy, FirstEdges, alpha, beta);
% figure(5)
% p4 = plot(TR,'EdgeLabel',TR.Edges.Weight,'Layout','layered');
% title('SelPolicy 4');
% if ~isempty(EdgeSet_P4)
%     highlight(p4,EdgeSet_P4,'NodeColor','m')
% end
% 
% disp('SelPolicy 5');
% disp(['alpha = ',num2str(alpha),' beta = ',num2str(beta)]);
% SelPolicy = 5;
% [EdgeSet_P5, Success_flag_P5] = SelEdges_HorizonLevel_v3(TR, ...
%     Cloud_second_layer, num_of_nodes, SelPolicy, FirstEdges, alpha, beta);
% figure(6)
% p5 = plot(TR,'EdgeLabel',TR.Edges.Weight,'Layout','layered');
% title('SelPolicy 5');
% if ~isempty(EdgeSet_P5)
%     highlight(p5,EdgeSet_P5,'NodeColor','c')
% end


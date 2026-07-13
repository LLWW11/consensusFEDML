function [client_num, max_layer,c2c_map] = CommonFL_v3(TSML_BdwMat, ...
    num_layers, num_of_nodes, Cost_of_TemporalLinks,...
    util, Cloud)


percent = 1- util;

% 
% [TSML_BdwMat, ~, num_of_nodes] = gen_random_tsmlg_v3(TopoOption, ...
%     num_of_nodes, num_layers, num_wave, percent);
% Cloud node cannot be a client 
ClientSet = 1:num_of_nodes;
x=find(ClientSet == Cloud);
ClientSet(x) = [];

 [AuxTopo] = tsmlg2adj(TSML_BdwMat, num_of_nodes, num_layers, Cost_of_TemporalLinks);

G = digraph(AuxTopo);
% figure(1)
% plot(G,'EdgeLabel',G.Edges.Weight,'Layout','layered')

% group_num = 0;
% client_num = zeros(1, length(EdgeSet));
max_layer=0;

c2c_map = cell(1,1);

c2c_map{1,1}=[];


%%
% =========client to cloud=========
w_srcEdge = 1;
w_dstEdge = inf;
% 


[G_vC2E] =  add_vnode_vedge_v2(G, ClientSet, Cloud, ...
    num_of_nodes, num_layers, w_srcEdge, w_dstEdge);
% figure(2)
% plot(G_vC2E,'EdgeLabel',G_vC2E.Edges.Weight,'Layout','layered')

virtual_src_id = num_of_nodes.*num_layers+1;
virtual_dst_id = virtual_src_id+1;
% use max net alg to find the client-edge paths
[mf_vC2E,GF_vC2E] = maxflow(G_vC2E,virtual_src_id,virtual_dst_id);
% figure(3)% ClientSet = [1 2 3 4];
% EdgeSet = [4 5];
% Cloud = 5 ;
% plot(GF_vC2E,'EdgeLabel',GF_vC2E.Edges.Weight,'Layout','layered');
client_num = mf_vC2E;
if client_num==0
    return;
end

[x] = find(GF_vC2E.Edges.EndNodes(:,2) == virtual_dst_id);
Client_list = GF_vC2E.Edges.EndNodes(x,1);
% find the max client layer 
% actual_CLayer_list=(Client_list - mod(Client_list,num_of_nodes))./num_of_nodes + 1;
actual_CLayer_list=((Client_list - mod(Client_list,num_of_nodes))./num_of_nodes +1).*...
    (mod(Client_list,num_of_nodes)~=0) + (mod(Client_list,num_of_nodes)==0).*Client_list./num_of_nodes;
max_layer = max(actual_CLayer_list);


[c2c_map] = C2E_Mapping_v4(ClientSet, Cloud, GF_vC2E, virtual_dst_id, num_of_nodes,num_layers);

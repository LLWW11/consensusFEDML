function [group_num, client_num, max_layer, actual_c2e_map] = HFL_TSMLG_v3(TSML_BdwMat, ...
    num_layers, num_of_nodes, ...
    util, EdgeSet, Cloud, Cost_of_TemporalLinks)

% TopoOption = '6node';
% num_layers = 2;
% num_of_nodes = 6;
% num_wave = 5;
% util = 0.8;
percent = 1- util;
% Cost_of_TemporalLinks = inf;

% [TSML_BdwMat, ~, num_of_nodes] = gen_random_tsmlg_v3(TopoOption, ...
%     num_of_nodes, num_layers, num_wave, percent);
% Cloud node cannot be a client or edge node
% The other nodes can be a client and an edge node simultaneously
ClientSet = 1:num_of_nodes;
x=find(ClientSet == Cloud);
ClientSet(x) = [];

 [AuxTopo] = tsmlg2adj(TSML_BdwMat, num_of_nodes, num_layers, Cost_of_TemporalLinks);

G = digraph(AuxTopo);
% figure(1)
% plot(G,'EdgeLabel',G.Edges.Weight,'Layout','layered')

group_num = 0;
client_num = zeros(1, length(EdgeSet));
max_layer=0;

actual_c2e_map = cell(1,length(EdgeSet));
for i=1:length(EdgeSet)
    actual_c2e_map{1,i}=[];
end

%%
% =========Phase 1: client to edge=========
w_srcEdge = 1;
w_dstEdge = inf;
% 


[G_vC2E] =  add_vnode_vedge_v2(G, ClientSet, EdgeSet, ...
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
num_aggregated_clients = mf_vC2E;

% identify client-edge mapping 
[c2emap] = C2E_Mapping(ClientSet, EdgeSet, GF_vC2E, virtual_dst_id, num_of_nodes);
client_tmp=0;
for k=1:length(EdgeSet)
    client_tmp = client_tmp+length(c2emap{1,k});
end
if client_tmp~=num_aggregated_clients
    error('client_tmp~=num_aggregated_clients!')
end



if num_aggregated_clients == 0    
    return;
end
%%
% =========Phase 2: edge to cloud=========
% remove the virtual src and the virtual dst
GF2=GF_vC2E;
GF2 = rmnode(GF2,virtual_dst_id);
GF2 = rmnode(GF2,virtual_src_id);
A_usedBW = full(adjacency(GF2,'weighted'));% only available in Matlab2022
% remove the bandwidth used by the client-edge communications from AuxTopo
A_PostEdge = AuxTopo - A_usedBW;
G_PostEdge = digraph(A_PostEdge);
% figure(4)
% plot(G_PostEdge,'EdgeLabel',G_PostEdge.Edges.Weight,'Layout','layered')

% find used edge nodes
[x,~] = find(GF_vC2E.Edges.EndNodes == virtual_dst_id);

Edge_list = GF_vC2E.Edges.EndNodes(x,1);
Weight_list = GF_vC2E.Edges.Weight(x);
Weight_list = Weight_list';
Edge_list = Edge_list';
    
[G_vE2C] =  add_vnode_vedge_cloud(G_PostEdge, Edge_list, Cloud,...
    num_of_nodes, num_layers, w_srcEdge, w_dstEdge);
% figure(5)
% plot(G_vE2C,'EdgeLabel',G_vE2C.Edges.Weight,'Layout','layered')
% [G_vNodeEdge] =  add_vnode_vedge_cloud(G, EdgeSet, Cloud,...
%     num_of_nodes, num_layers, w_srcEdge, w_dstEdge);
% figure(2)
% plot(G_vNodeEdge,'EdgeLabel',G_vNodeEdge.Edges.Weight,'Layout','layered')
% 
virtual_src_id = num_of_nodes.*num_layers+1;
virtual_dst_id = virtual_src_id+1;
% use max net alg to find the edge-cloud paths
[mf_vE2C,GF_vE2C] = maxflow(G_vE2C,virtual_src_id,virtual_dst_id);
% figure(6)
% plot(GF_vE2C,'EdgeLabel',GF_vE2C.Edges.Weight,'Layout','layered')
num_aggregated_edges = mf_vE2C; 



if num_aggregated_edges == 0    
    return;
end
%%
% find aggregated edge nodes
[x,~] = find(GF_vE2C.Edges.EndNodes == virtual_src_id);
AggEdge_list = GF_vE2C.Edges.EndNodes(x,2);

% actual_AggEdge_list = mod(AggEdge_list,num_of_nodes);
actual_AggEdge_list = (mod(AggEdge_list,num_of_nodes)==0).*num_of_nodes + mod(AggEdge_list,num_of_nodes);

% remove the same nodes in different layers
AggEdge = unique(actual_AggEdge_list);

% actual_client_list = zeros(1,length(AggEdge));
for i=1:length(AggEdge)
    EdgeNode = AggEdge(i);    
    x = find(actual_AggEdge_list == EdgeNode);
    actual_clients = sum(Weight_list(x));
    client_num(i)= actual_clients;
end
group_num = length(AggEdge);
% client_num = actual_client_list;

if num_aggregated_clients < sum(client_num)
    error('num_aggregated_clients ~= client_num');
end
if num_aggregated_edges < group_num
    error('num_aggregated_edges < group_num');
end

% find the max aggregated layer 
[x] = find(GF_vE2C.Edges.EndNodes(:,2) == virtual_dst_id);
nodes_list = GF_vE2C.Edges.EndNodes(x,1);
% actual_AggLayer_list=(nodes_list - mod(nodes_list,num_of_nodes))./num_of_nodes + 1;
actual_AggLayer_list= ((nodes_list - mod(nodes_list,num_of_nodes))./num_of_nodes +1).*...
    (mod(nodes_list,num_of_nodes)~=0) + (mod(nodes_list,num_of_nodes)==0).*nodes_list./num_of_nodes;
max_layer = max(actual_AggLayer_list);

% identify the actual client-edge mapping
for i=1:length(AggEdge)
    EdgeNode = AggEdge(i);    
    x = find(EdgeSet == EdgeNode);
    actual_c2e_map{1,x} = c2emap{1,x};
end

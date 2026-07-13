function [group_num_HFLSnF, client_num_HFLSnF, ...
    max_layer, actual_c2e_map, DynEdgeSet] = HFL_SnF_v1(TSML_BdwMat, AuxTopo, ClientSet, EdgeSet_fixed, Cloud,...
    num_of_nodes, num_layers, EdgeDynEnable, SelPolicy, FirstEdges, alpha, beta)
% EdgeDynEnable = 0;




% =========Phase 1: client to edge=========
w_srcEdge = 1;
w_dstEdge = inf;
% 

if EdgeDynEnable == 1
    [EdgeSet_dyn, Flag_dyn] = DynEdgeSelection_v1(TSML_BdwMat,num_of_nodes, num_layers,...
        Cloud, SelPolicy, FirstEdges, alpha, beta);
    DynEdgeSet = EdgeSet_dyn;    
    if Flag_dyn == 1
        EdgeSet = EdgeSet_dyn;
    else
        group_num_HFLSnF = 0;
        client_num_HFLSnF = 0;
        max_layer = 0;
        actual_c2e_map = cell(1);
        return;
    end

else
    DynEdgeSet = [];
    EdgeSet = EdgeSet_fixed;
end

group_num = 0;
client_num = zeros(1, length(EdgeSet));
max_layer = 0;
actual_c2e_map = cell(1,length(EdgeSet));
for i=1:length(EdgeSet)
    actual_c2e_map{1,i}=[];
end

G = digraph(AuxTopo);

[G_vC2E] =  add_vnode_vedge_v2(G, ClientSet, EdgeSet, ...
    num_of_nodes, num_layers, w_srcEdge, w_dstEdge);
% figure(2)
% plot(G_vC2E,'EdgeLabel',G_vC2E.Edges.Weight,'Layout','layered')

virtual_src_id = num_of_nodes.*num_layers+1;
virtual_dst_id = virtual_src_id+1;
% use max net alg to find the client-edge paths
[mf_vC2E,GF_vC2E] = maxflow(G_vC2E,virtual_src_id,virtual_dst_id);
% ClientSet = [1 2 3 4];
% EdgeSet = [4 5];
% Cloud = 5 ;
% figure(1)
% P=plot(GF_vC2E,'EdgeLabel',GF_vC2E.Edges.Weight,'Layout','layered');
% highlight(P,EdgeSet,'NodeColor','r')

% [x,~] = find(GF_vC2E.Edges.EndNodes == virtual_dst_id);
% if ~isempty(x)
%     nodes_connect_vdst = GF_vC2E.Edges.EndNodes(x,1);
%     pre_nodes_connect_vdst = Search_Allpredecessors(GF_vC2E, nodes_connect_vdst);
% %     pre_nodes_connect_vdst= predecessors(GF_vC2E,nodes_connect_vdst);
%     x2 = find(pre_nodes_connect_vdst == virtual_src_id);
%     if ~isempty(x2)
%         twohopnode = pre_nodes_connect_vdst(x2);
%         for ii=1:length(EdgeSet)
%             edge=EdgeSet(ii);
%             x3 = find(twohopnode == edge);
%             if ~isempty(x3)
%                 twohopnode(x3) = [];
%             end
%         end
%     end
% end
% length(x3)
num_aggregated_clients = mf_vC2E;

if num_aggregated_clients == 0  
    group_num_HFLSnF = group_num;
    client_num_HFLSnF = client_num;
    return;
end

% identify client-edge mapping 
[c2emap] = C2E_Mapping(ClientSet, EdgeSet, GF_vC2E, virtual_dst_id, num_of_nodes);
client_tmp=0;
for k=1:length(EdgeSet)
    client_tmp = client_tmp+length(c2emap{1,k});
end
if client_tmp~=(num_aggregated_clients)
    error('client_tmp~=num_aggregated_clients!')
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
    group_num_HFLSnF = group_num;
    client_num_HFLSnF = client_num;
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
if length(client_num) < length(EdgeSet)
    disp('1');
end
% client_num = actual_client_list;

if num_aggregated_clients < sum(client_num)
    error('num_aggregated_clients ~= client_num');
end
if num_aggregated_edges < group_num
    error('num_aggregated_edges < group_num');
end

group_num_HFLSnF = group_num;
client_num_HFLSnF = client_num;

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
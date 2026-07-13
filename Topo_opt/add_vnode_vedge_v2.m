function [G_vNodeEdge] =  add_vnode_vedge_v2(G, ClientSet, EdgeSet, ...
    num_of_nodes, num_layers, w_srcEdge, w_dstEdge)

ll = 1:num_layers;

virtual_src_id = num_of_nodes.*num_layers+1;
virtual_dst_id = virtual_src_id+1;

Gv1 = addnode(G,2);


Gv2 = Gv1;
for i=1:length(ClientSet)
    node_id = ClientSet(i);
    w = w_srcEdge;
    % connect the virtual src node to each edge server on the first layer
    Gv2 = addedge(Gv2, virtual_src_id, node_id, w);
    
end

for i=1:length(EdgeSet)
    node_id = EdgeSet(i);
    node_id_list = (ll-1).*num_of_nodes + node_id;
    for j=1:num_layers
        node_id2 = node_id_list(j);
        w2 = w_dstEdge;
        % connect the edge server on each layer to the virtual dst node
        Gv2 = addedge(Gv2, node_id2, virtual_dst_id, w2);
    end
end
G_vNodeEdge = Gv2;


function [last_nodes_first_layer] = FindLastNodesFirstLayer(G, nodes_second_layer,num_of_nodes)


Nodes_Still_SecondLayer = nodes_second_layer;
last_nodes_first_layer = [];

while ~isempty(Nodes_Still_SecondLayer)
%     nodes_actual_id = mod(Nodes_Still_SecondLayer,num_of_nodes);
    nodes_actual_id = (mod(Nodes_Still_SecondLayer,num_of_nodes)==0).*num_of_nodes + mod(Nodes_Still_SecondLayer,num_of_nodes);
%     nodes_actual_layer = (Nodes_Still_SecondLayer - nodes_actual_id)./num_of_nodes + 1;
    nodes_actual_layer = ((Nodes_Still_SecondLayer - mod(Nodes_Still_SecondLayer,num_of_nodes))./num_of_nodes +1).*...
    (mod(Nodes_Still_SecondLayer,num_of_nodes)~=0) + (mod(Nodes_Still_SecondLayer,num_of_nodes)==0).*Nodes_Still_SecondLayer./num_of_nodes;
    indx1 = find(nodes_actual_layer == 1);
    if ~isempty(indx1)
        last_nodes_first_layer = [last_nodes_first_layer, Nodes_Still_SecondLayer(indx1)];
        Nodes_Still_SecondLayer(indx1) = [];
    end
    if ~isempty(Nodes_Still_SecondLayer)
        [AllPreSet, ~]=Search_Allpredecessors(G, Nodes_Still_SecondLayer);
        Nodes_Still_SecondLayer = AllPreSet;
    end
end


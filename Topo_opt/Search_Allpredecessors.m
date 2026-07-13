function [AllPreSet, num_predecessors]=Search_Allpredecessors(G, NodeSet)

AllPreSet = [];
num_nodes = length(NodeSet);
num_predecessors = 0;
for i = 1:num_nodes
    nodeID = NodeSet(i);
    preIDs = predecessors(G,nodeID);
    if ~isempty(preIDs)
        AllPreSet = [AllPreSet,preIDs'];
        num_predecessors = num_predecessors+length(preIDs);
    end
end
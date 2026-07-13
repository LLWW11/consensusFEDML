function [num_All_Pred_Lf] = Allpredecessors_Until_Leaf(G, NodeID)

[AllPreSet, num_predecessors]=Search_Allpredecessors(G, NodeID);
num_All_Pred_Lf =num_predecessors;
while ~isempty(AllPreSet)
    NodeSet = AllPreSet;
    [AllPreSet, num_predecessors]=Search_Allpredecessors(G, NodeSet);
    num_All_Pred_Lf = num_All_Pred_Lf + num_predecessors;
end
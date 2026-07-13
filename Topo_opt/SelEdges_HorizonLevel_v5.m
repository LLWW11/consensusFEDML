function [EdgeSet, Success_flag] = SelEdges_HorizonLevel_v5(GTree, ...
    Cloud_second_layer, num_of_nodes, SelPolicy, ...
    FirstEdges, alpha, beta)

indx = find(GTree.Edges.EndNodes(:,2)==Cloud_second_layer);

nodes_second_layer = GTree.Edges.EndNodes(indx,1);

last_nodes_first_layer = FindLastNodesFirstLayer(GTree, nodes_second_layer,num_of_nodes);

nodes_Cloudhop2 = last_nodes_first_layer;



EdgeSet = [];
NodeSet = [];
% Select a EdgeSet policy
if ~isempty(nodes_Cloudhop2)
    if SelPolicy == 1 
        EdgeSet = nodes_Cloudhop2;
    elseif SelPolicy == 2
        [EdgeSet, ~]=Search_Allpredecessors(GTree, nodes_Cloudhop2); 
    elseif SelPolicy == 3
        [nodes_Cloudhop3, ~]=Search_Allpredecessors(GTree, nodes_Cloudhop2);
        if ~isempty(nodes_Cloudhop3)
            [nodes_Cloudhop4, ~]=Search_Allpredecessors(GTree, nodes_Cloudhop3);         
            EdgeSet = nodes_Cloudhop4; 
        else
            EdgeSet = [];
        end 
        
    elseif SelPolicy == 4
        NodeSet = [NodeSet,nodes_Cloudhop2];
        [nodes_Cloudhop3, ~]=Search_Allpredecessors(GTree, nodes_Cloudhop2);
        if ~isempty(nodes_Cloudhop3)
            NodeSet = [NodeSet,nodes_Cloudhop3];
            [nodes_Cloudhop4, ~]=Search_Allpredecessors(GTree, nodes_Cloudhop3);         
            if ~isempty(nodes_Cloudhop4)
                NodeSet = [NodeSet,nodes_Cloudhop4];
            end      
        end 
        indegree_set  = zeros(1,length(NodeSet));
        for i=1:length(NodeSet)
            nodeid = NodeSet(i);
            indegree_set(i) = indegree(GTree,nodeid);
        end   
        [~,I] = sort(indegree_set,'descend');
        if length(I) >= FirstEdges
            EdgeSet = NodeSet(I(1:FirstEdges));
            EdgeSet = sort(EdgeSet);
        else
            EdgeSet = NodeSet(I);
            EdgeSet = sort(EdgeSet);
        end
    elseif SelPolicy == 5
        NodeSet = [NodeSet,nodes_Cloudhop2];
        [nodes_Cloudhop3, ~]=Search_Allpredecessors(GTree, nodes_Cloudhop2);
        if ~isempty(nodes_Cloudhop3)
            NodeSet = [NodeSet,nodes_Cloudhop3];
            [nodes_Cloudhop4, ~]=Search_Allpredecessors(GTree, nodes_Cloudhop3);         
            if ~isempty(nodes_Cloudhop4)
                NodeSet = [NodeSet,nodes_Cloudhop4];
            end      
        end 
        indegree_set  = zeros(1,length(NodeSet));
        num_All_Pred_Lf = zeros(1,length(NodeSet));
        
        for i=1:length(NodeSet)
            nodeid = NodeSet(i);
            indegree_set(i) = indegree(GTree,nodeid);
            num_All_Pred_Lf(i) = Allpredecessors_Until_Leaf(GTree, nodeid);
        end 
        sum_weights = alpha.*indegree_set./(max(indegree_set)) + beta.*num_All_Pred_Lf./(max(num_All_Pred_Lf));
        [~,I] = sort(sum_weights,'descend');
        if length(I) >= FirstEdges
            EdgeSet = NodeSet(I(1:FirstEdges));
            EdgeSet = sort(EdgeSet);
        else
            EdgeSet = NodeSet(I);
            EdgeSet = sort(EdgeSet);
        end
    else
        error('There is no corresponding policy!!!');
    end

elseif SelPolicy == 6
        [nodes_Cloudhop3, ~]=Search_Allpredecessors(GTree, nodes_Cloudhop2);
        if ~isempty(nodes_Cloudhop3)
            [nodes_Cloudhop4, ~]=Search_Allpredecessors(GTree, nodes_Cloudhop3);         
            EdgeSet = [nodes_Cloudhop3,nodes_Cloudhop4]; 
            EdgeSet = unique(EdgeSet);
        else
            EdgeSet = [];
        end 

else
    EdgeSet = [];
end

% remove the nodes whose indgree is zeros
if ~isempty(EdgeSet)
    remove_node_id = [];
    num_edges = length(EdgeSet);
    for i = 1:num_edges
        nodeID = EdgeSet(i);
        D = indegree(GTree,nodeID);
        if D == 0
            remove_node_id = [remove_node_id,i];
        end
    end
    if ~isempty(remove_node_id)
        EdgeSet(remove_node_id)=[];
    end
end

% check whether EdgeSet is empty
if isempty(EdgeSet)
    disp('Fail to find any edges!');
    Success_flag =0;
else
    Success_flag =1;
end

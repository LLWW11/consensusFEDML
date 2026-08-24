function [c2emap] = C2E_Mapping_v4(ClientSet, EdgeSet, GF, virtual_dst_id, num_of_nodes, num_layers)

c2emap =cell(num_layers,length(EdgeSet));


% for i=1:length(EdgeSet)
%     c2emap{1,i}=[];
% 
% end

for i=1:length(ClientSet)
    s=ClientSet(i);
    P = shortestpath(GF,s,virtual_dst_id);
    % whether s can reach virtual_dst_id
    if isempty(P)
        continue;
    end

    % whether s is a edge node
    selfedge = find(EdgeSet==s);
    if ~isempty(selfedge)
        c2emap{1,selfedge}=[c2emap{1,selfedge},s];
        continue;
    end
    indx = 1:length(P)-1;
    indx = sort(indx,'descend');
    P2=P(indx);
    for j=1:length(P)-1       
        next_node = (mod(P2(j),num_of_nodes)==0).*num_of_nodes + mod(P2(j),num_of_nodes);
        next_node_layer=((P2(j) - mod(P2(j),num_of_nodes))./num_of_nodes +1).*...
            (mod(P2(j),num_of_nodes)~=0) + (mod(P2(j),num_of_nodes)==0).*P2(j)./num_of_nodes;
        agg_edge_indx = find(EdgeSet == next_node);
        if isempty(agg_edge_indx)
            continue;
        else
            c2emap{next_node_layer,agg_edge_indx}=[c2emap{next_node_layer,agg_edge_indx},s];
            break;
        end        
    end

    
    for j=1:length(P)-1
          next_node = P(j+1);
          curr_node = P(j);    
          x1 = find(GF.Edges.EndNodes(:,1) ==curr_node);
          x2 = find(GF.Edges.EndNodes(:,2) ==next_node);
          x3 = intersect(x1,x2);
          if isempty(x3)
            error('No such edge !!!')
          end          
          GF.Edges.Weight(x3) = GF.Edges.Weight(x3) - 1;
          if GF.Edges.Weight(x3) <0
            error('Edges.Weight  <0 !!!');
          elseif GF.Edges.Weight(x3) == 0
              GF = rmedge(GF,x3);
          else
              continue;
          end
                  
    end
end
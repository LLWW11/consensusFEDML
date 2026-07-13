function [c2emap] = C2E_Mapping_v2(ClientSet, EdgeSet, GF_vE2C, virtual_dst_id, num_of_nodes)

c2emap =cell(1,length(EdgeSet));


for i=1:length(EdgeSet)
    c2emap{1,i}=[];

end

for i=1:length(ClientSet)
    s=ClientSet(i);

    P = shortestpath(GF_vE2C,s,virtual_dst_id);
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

    for j=1:length(P)-1
%         next_node = mod(P(j+1),num_of_nodes);
        indx = 1:length(P)-1;
        indx = sort(indx,'descend');
        P2=P(indx);
        next_node = (mod(P2(j),num_of_nodes)==0).*num_of_nodes + mod(P2(j),num_of_nodes);

        agg_edge_indx = find(EdgeSet == next_node);
        if isempty(agg_edge_indx)
            continue;
        else
            c2emap{1,agg_edge_indx}=[c2emap{1,agg_edge_indx},s];
            break;
        end        
    end
end
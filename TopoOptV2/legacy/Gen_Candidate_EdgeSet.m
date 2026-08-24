function [CandidateEdgeSet, InitialEdgeSet, LeftoverEdgeSet] = Gen_Candidate_EdgeSet(adj_mat, Cloud, ...
    num_of_nodes, MinHop, MaxHop, TargetHop)
G_adjmat = digraph(adj_mat);
distan_Cloud = distances(G_adjmat,Cloud);
CandidateEdgeSet = [];
InitialEdgeSet = [];
LeftoverEdgeSet =[];
for i=1:num_of_nodes
    if distan_Cloud(i) >= MinHop && distan_Cloud(i) <= MaxHop  
        CandidateEdgeSet = [CandidateEdgeSet,i];
        if distan_Cloud(i) == TargetHop
            InitialEdgeSet = [InitialEdgeSet,i];
        else
            LeftoverEdgeSet = [LeftoverEdgeSet,i];
        end
    end
end

if isempty(CandidateEdgeSet)
    error('CandidateEdgeSet is empty!!!');
end

if isempty(InitialEdgeSet)
    error('InitialEdgeSet is empty!!!');
end

if isempty(LeftoverEdgeSet)
    error('LeftoverEdgeSet is empty!!!');
end
function [AuxTopo] = tsmlg2adj(TSML_BdwMat, num_of_nodes, num_layers, Cost_of_TemporalLinks)

AuxTSML = TSML_BdwMat;
storage_matrix = (eye(num_of_nodes)).*Cost_of_TemporalLinks;
tmp = find(isnan(storage_matrix));

storage_matrix = fillmissing(storage_matrix,'constant',0);


for i = 1:num_layers
        %generate auxiliary topology
        if i == 1
            AuxTopo = reshape(AuxTSML(:,:,i),num_of_nodes,num_of_nodes);
        else
            [x,y] = size(AuxTopo);
            tmpTopo = AuxTopo;
            AuxTopo = zeros(i*num_of_nodes);                
            AuxTopo(1:x,1:y) = tmpTopo;
            AuxTopo(x+1:x+num_of_nodes,y+1:y+num_of_nodes) = reshape(AuxTSML(:,:,i), num_of_nodes, num_of_nodes);
            AuxTopo(x-num_of_nodes+1:x,y+1:y+num_of_nodes) = storage_matrix;
        end
end
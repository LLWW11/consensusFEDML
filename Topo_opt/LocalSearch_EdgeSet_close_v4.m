function [OptimEdgeSet,optim_client_num] = LocalSearch_EdgeSet_close_v4(TSML_BdwMat, adj_mat, Cloud, ClientSet,...
    num_of_nodes, num_layers, MinHop, MaxHop, TargetHop)
% clc;clear;
% 
% TopoOption = 'Metro';
% num_layers = 4;
% num_of_nodes = 6;
% num_wave = 5;
% util = 0.8;
% Cost_of_TemporalLinks = inf;
% percent = 1- util;
% Cloud = 18;
% MinHop=1; 
% MaxHop=4;
% TargetHop=2;
% 
% 
% [TSML_BdwMat, adj_mat, num_of_nodes] = gen_random_tsmlg_v3(TopoOption, ...
%     num_of_nodes, num_layers, num_wave, percent);
% ClientSet = 1:num_of_nodes;
% ClientSet(Cloud) = [];



[CandidateEdgeSet, InitialEdgeSet, ~] = Gen_Candidate_EdgeSet(adj_mat, Cloud, ...
    num_of_nodes, MinHop, MaxHop, TargetHop);

%%==== Initialization =======
CurrentEdgeSet = InitialEdgeSet;
[~, client_num_HFLSnF, ~, ~] = HFL_SnF_v5(TSML_BdwMat, ClientSet, CurrentEdgeSet, Cloud,...
        num_of_nodes, num_layers);
current_client_num = sum(client_num_HFLSnF);
indx_emptyedge2 = find(client_num_HFLSnF == 0);
if ~isempty(indx_emptyedge2)
    TempEdgeSet2= CurrentEdgeSet;
    TempEdgeSet2(indx_emptyedge2) = [];
else
    TempEdgeSet2= CurrentEdgeSet;
end
CurrentEdgeSet = TempEdgeSet2;
pre_client_num = 0;
PreEdgeSet = CurrentEdgeSet;
last_round = 0;
round = 1;
while (pre_client_num < current_client_num) || last_round == 1
    disp(['Round = ',num2str(round)]);
    %%==== Open =======
    idx_leftover = ~ismember(CandidateEdgeSet, CurrentEdgeSet);
    LeftoverEdgeSet_open = CandidateEdgeSet(idx_leftover);
    TempEdgeSet = CurrentEdgeSet;
    for i=1:length(LeftoverEdgeSet_open)
        new_edge = LeftoverEdgeSet_open(i);
        TempEdgeSet = [CurrentEdgeSet, new_edge];    
        [~, temp_client_num, ~, tmp_actual_c2e_map] = HFL_SnF_v5(TSML_BdwMat, ClientSet, TempEdgeSet, Cloud,...
            num_of_nodes, num_layers);
        if sum(temp_client_num) > current_client_num
            % delete edges without any clients
%             indx_emptyedge1 = cellfun(@isempty, tmp_actual_c2e_map);
            indx_emptyedge2 = find(temp_client_num == 0);
            if ~isempty(indx_emptyedge2)
                TempEdgeSet2= TempEdgeSet;
                TempEdgeSet2(indx_emptyedge2) = [];
            else
                TempEdgeSet2= TempEdgeSet;
            end
            CurrentEdgeSet = TempEdgeSet2;
            current_client_num = sum(temp_client_num);
        else
            continue;
        end
    end
%     [~,edgenum] = size(tmp_actual_c2e_map);
%     for kk=1:edgenum
%         if isempty(tmp_actual_c2e_map{1,kk})
%             disp('Some edges are empty in Open !!!')
%         end
%     end
    
    %%==== Close =======
    Wait_Search = CurrentEdgeSet;
    for i=1:length(Wait_Search)
        current_node = Wait_Search(i);
        TempEdgeSet = CurrentEdgeSet;
        remove_indx = find(TempEdgeSet == current_node);
        if isempty(remove_indx)
            error('remove_indx is empty!!!');
        else
            TempEdgeSet(remove_indx)=[];
        end
        
        [~, temp_client_num, ~, tmp_actual_c2e_map] = HFL_SnF_v5(TSML_BdwMat, ClientSet, TempEdgeSet, Cloud,...
            num_of_nodes, num_layers);
        if sum(temp_client_num) >= current_client_num
            % delete edges without any clients
%             indx_emptyedge1 = cellfun(@isempty, tmp_actual_c2e_map);
            indx_emptyedge2 = find(temp_client_num == 0);
            if ~isempty(indx_emptyedge2)
                TempEdgeSet2= TempEdgeSet;
                TempEdgeSet2(indx_emptyedge2) = [];
            else
                TempEdgeSet2= TempEdgeSet;
            end
            CurrentEdgeSet = TempEdgeSet2;
            current_client_num = sum(temp_client_num);
        else
            continue;
        end
    end

    [~, ~, ~, tmp_actual_c2e_map] = HFL_SnF_v5(TSML_BdwMat, ClientSet, CurrentEdgeSet, Cloud,...
            num_of_nodes, num_layers);
    [~,edgenum] = size(tmp_actual_c2e_map);
    for kk=1:edgenum
        if isempty(tmp_actual_c2e_map{1,kk})
            disp('Some edges are empty in Close !!!')
        end
    end

    %%==== Swap =======
    idx_leftover = ~ismember(CandidateEdgeSet, CurrentEdgeSet);
    LeftoverEdgeSet_swap = CandidateEdgeSet(idx_leftover);

    if length(LeftoverEdgeSet_swap) ~= length(unique(LeftoverEdgeSet_swap))
        disp('数组中存在重复元素');
    end

    for j=1:length(LeftoverEdgeSet_swap)
        for i=1:length(CurrentEdgeSet)
            TempEdgeSet = CurrentEdgeSet;
            TempEdgeSet(i) = LeftoverEdgeSet_swap(j);
            [~, temp_client_num, ~, tmp_actual_c2e_map] = HFL_SnF_v5(TSML_BdwMat, ClientSet, TempEdgeSet, Cloud,...
                num_of_nodes, num_layers);
            if sum(temp_client_num) > current_client_num
                % delete edges without any clients
%             indx_emptyedge1 = cellfun(@isempty, tmp_actual_c2e_map);
                indx_emptyedge2 = find(temp_client_num == 0);
                if ~isempty(indx_emptyedge2)
                    TempEdgeSet2= TempEdgeSet;
                    TempEdgeSet2(indx_emptyedge2) = [];
                else
                    TempEdgeSet2= TempEdgeSet;
                end
                CurrentEdgeSet = TempEdgeSet2;
                current_client_num = sum(temp_client_num);
                break;
            else
                continue;
            end
        end
    end
%     [~,edgenum] = size(tmp_actual_c2e_map);
%     for kk=1:edgenum
%         if isempty(tmp_actual_c2e_map{1,kk})
%             disp('Some edges are empty in Swap !!!')
%         end
%     end

    


    %%==== Stop Condition =====
    if pre_client_num < current_client_num
        pre_client_num = current_client_num;
        PreEdgeSet = CurrentEdgeSet;
        last_round = 1;
    elseif length(PreEdgeSet) > length(CurrentEdgeSet) 
        pre_client_num = current_client_num;
        PreEdgeSet = CurrentEdgeSet;
        last_round = 1;
    else
        last_round = 0;
    end
    round = round +1;
end

OptimEdgeSet = sort(CurrentEdgeSet);
optim_client_num =current_client_num;

% %%==== Doule Check =====
% random_indx = randi([1 length(OptimEdgeSet)]);
% RandomEdgeSet = OptimEdgeSet;
% RandomEdgeSet(random_indx) = [];
% 
% [~, random_client_num, ~, ~] = HFL_SnF_v2(TSML_BdwMat, ClientSet, RandomEdgeSet, Cloud,...
%                 num_of_nodes, num_layers);
% 
% 
% if random_client_num <= optim_client_num
%     disp('Congratulation!!! The EdgeSet is optimal!!!')
% else
%     disp('Fail!!! The EdgeSet is not optimal!!!')
% end


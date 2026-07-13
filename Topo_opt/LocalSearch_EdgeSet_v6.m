function [OptimEdgeSet,optim_client_num] = LocalSearch_EdgeSet_v6(TSML_BdwMat, adj_mat, Cloud, ClientSet,...
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

[CandidateEdgeSet, InitialEdgeSet, ~] = Gen_Candidate_EdgeSet(adj_mat, Cloud, num_of_nodes, MinHop, MaxHop, TargetHop);

%%==== Initialization =======
CurrentEdgeSet = InitialEdgeSet;
[group_num_HFLSnF, client_num_HFLSnF, max_layer, ~] = HFL_SnF_v5(TSML_BdwMat, ClientSet, CurrentEdgeSet, Cloud,...
    num_of_nodes, num_layers);
% client_num = zeros(1, length(EdgeSet));
current_client_num = sum(client_num_HFLSnF);

indx_emptyedge2 = find(client_num_HFLSnF == 0);
if ~isempty(indx_emptyedge2)
    TempEdgeSet2= CurrentEdgeSet;
    TempEdgeSet2(indx_emptyedge2) = [];
else
    TempEdgeSet2= CurrentEdgeSet;
end  %这部分去掉那些没有客户端节点的边缘节点

CurrentEdgeSet = TempEdgeSet2;
pre_client_num = 0;
% gamma = 6000;     % 37 + 6000/200
gamma = 6000;
pre_t = inf;
PreEdgeSet = CurrentEdgeSet;
last_round = 0;
round = 1;


% optim_obj_current = current_client_num + gamma/current_time_total_;
optim_obj_current = OptimObj(group_num_HFLSnF,current_client_num,gamma,max_layer);
% optim_obj_pre = pre_t;
% optim_obj_pre = pre_client_num + gamma/pre_t;
optim_obj_pre = -inf;
% optim_obj_pre = 0.3*pre_client_num/37 - 0.7*pre_t/200;
% while (optim_obj_pre < optim_obj_current  &&  pre_client_num <= current_client_num && optim_obj_current-optim_obj_pre>0.1) || last_round == 1
while (optim_obj_pre < optim_obj_current) || last_round == 1
    disp(['Round = ',num2str(round)]);
    %%======== Open =========
    idx_leftover = ~ismember(CandidateEdgeSet, CurrentEdgeSet);
    LeftoverEdgeSet_open = CandidateEdgeSet(idx_leftover);%剩余的节点集合
    TempEdgeSet = CurrentEdgeSet;
    for i=1:length(LeftoverEdgeSet_open)
        new_edge = LeftoverEdgeSet_open(i);
        TempEdgeSet = [CurrentEdgeSet, new_edge];
        [temp_group_num, temp_client_num, temp_max_layer, ~] = HFL_SnF_v5(TSML_BdwMat, ClientSet, TempEdgeSet, Cloud,num_of_nodes, num_layers);
        optim_obj_temp = OptimObj(temp_group_num, temp_client_num, gamma, temp_max_layer);
        if optim_obj_temp > optim_obj_current
            % if optim_obj_temp < optim_obj_current && sum(temp_client_num) >= current_client_num
            indx_emptyedge2 = find(temp_client_num == 0);            % delete edges without any clients
            if ~isempty(indx_emptyedge2)
                TempEdgeSet2= TempEdgeSet;
                TempEdgeSet2(indx_emptyedge2) = [];
            else
                TempEdgeSet2= TempEdgeSet;
            end
            optim_obj_current = optim_obj_temp;
            CurrentEdgeSet = TempEdgeSet2;
            current_client_num = sum(temp_client_num);
        else
            continue;
        end
    end
    %%==== Swap =======
    idx_leftover = ~ismember(CandidateEdgeSet, CurrentEdgeSet);
    LeftoverEdgeSet_swap = CandidateEdgeSet(idx_leftover);
    if length(LeftoverEdgeSet_swap) ~= length(unique(LeftoverEdgeSet_swap))
        disp('数组中存在重复元素');%剩余的客户端节点中存在重复的元素?
    end
    for j=1:length(LeftoverEdgeSet_swap)
        for i=1:length(CurrentEdgeSet)
            TempEdgeSet = CurrentEdgeSet;
            TempEdgeSet(i) = LeftoverEdgeSet_swap(j);
            [temp_group_num, temp_client_num, temp_max_layer, ~] = HFL_SnF_v5(TSML_BdwMat, ClientSet, TempEdgeSet, Cloud,...
                num_of_nodes, num_layers);
            optim_obj_temp = OptimObj(temp_group_num, temp_client_num, gamma, temp_max_layer);
            if optim_obj_temp > optim_obj_current
                % if optim_obj_temp < optim_obj_current && sum(temp_client_num) >= current_client_num
                indx_emptyedge2 = find(temp_client_num == 0);                % delete edges without any clients
                if ~isempty(indx_emptyedge2)
                    TempEdgeSet2= TempEdgeSet;
                    TempEdgeSet2(indx_emptyedge2) = [];
                else
                    TempEdgeSet2= TempEdgeSet;
                end
                optim_obj_current = optim_obj_temp;
                CurrentEdgeSet = TempEdgeSet2;
                current_client_num = sum(temp_client_num);
                break;
            else
                continue;
            end
        end
    end
    %%==== Close =======
    if length(CurrentEdgeSet) ~= length(unique(CurrentEdgeSet))
        disp('Same edges exist in CurrentEdgeSet')
    end
    Wait_Search = CurrentEdgeSet;
    for i=1:length(Wait_Search)
        current_node = Wait_Search(i);
        TempEdgeSet = CurrentEdgeSet;
        remove_indx = find(TempEdgeSet == current_node);
        if isempty(remove_indx)
            % error('remove_indx is empty!!!');
            continue;
        else
            TempEdgeSet(remove_indx)=[];
        end
        [temp_group_num, temp_client_num, temp_max_layer, ~] = HFL_SnF_v5(TSML_BdwMat, ClientSet, TempEdgeSet, Cloud,...
            num_of_nodes, num_layers);
        optim_obj_temp = OptimObj(temp_group_num, temp_client_num, gamma, temp_max_layer);
        if optim_obj_temp > optim_obj_current
            % if optim_obj_temp < optim_obj_current && sum(temp_client_num) >= current_client_num
            indx_emptyedge2 = find(temp_client_num == 0);            % delete edges without any clients
            if ~isempty(indx_emptyedge2)
                TempEdgeSet2= TempEdgeSet;
                TempEdgeSet2(indx_emptyedge2) = [];
            else
                TempEdgeSet2= TempEdgeSet;
            end
            optim_obj_current = optim_obj_temp;
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
    %%==== Stop Condition =====
    % if optim_obj_pre < optim_obj_current && pre_client_num <= current_client_num && optim_obj_current-optim_obj_pre>0.1
    if optim_obj_pre < optim_obj_current
        optim_obj_pre = optim_obj_current;
        pre_client_num = current_client_num;
        disp(['optim_obj_current = ',num2str(optim_obj_current)]);
        disp(['current_client_num = ',num2str(current_client_num)]);
        PreEdgeSet = CurrentEdgeSet;
        last_round = 1;
        %   精简结构，在优化目标不在增大的情况下，边缘节点越少?
        % elseif length(PreEdgeSet) > length(CurrentEdgeSet)
        %     pre_client_num = current_client_num;
        %     PreEdgeSet = CurrentEdgeSet;
        %     last_round = 1;
    else
        last_round = 0;
    end
    round = round +1;
    % %%==== Stop Condition =====
    % if pre_client_num < current_client_num
    %     pre_client_num = current_client_num;
    %     PreEdgeSet = CurrentEdgeSet;
    %     last_round = 1;
    % elseif length(PreEdgeSet) > length(CurrentEdgeSet)
    %     pre_client_num = current_client_num;
    %     PreEdgeSet = CurrentEdgeSet;
    %     last_round = 1;
    % else
    %     last_round = 0;
    % end
    % round = round +1;
end
OptimEdgeSet = sort(CurrentEdgeSet);
optim_client_num =current_client_num;


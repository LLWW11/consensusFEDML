function summary = test_result_generation_smoke()
%TEST_RESULT_GENERATION_SMOKE 对修复后的拓扑结果生成流程执行小规模冒烟测试。
%   使用 3 个 epoch 和 util=[0.3,0.7]，逐快照验证策略位置、参与人数、
%   客户端映射和聚合时间公式，并返回验证摘要。

epoch_num = 3;
total_util = [0.3, 0.7];
TopoOption = 'Metro';
num_layers = 8;
num_of_nodes = 38;
num_wave = 5;
mean_time_interval = 100;
duration = 200;
EdgeSet = [7 8 9 12 20 27];
Cloud = 18;
agg_time_per_clients = 2;
base_seed = 20260711;
valid_client_ids = setdiff(1:num_of_nodes, Cloud);
topology_seed_matrix = base_seed + reshape( ...
    0:(epoch_num * numel(total_util) - 1), epoch_num, numel(total_util));

time_HFLSnF_fix = zeros(epoch_num, numel(total_util));
time_HFLSnF_dynamic = zeros(epoch_num, numel(total_util));
time_HFLnoSnF_fix = zeros(epoch_num, numel(total_util));
time_HFLnoSnF_dynamic = zeros(epoch_num, numel(total_util));

for util_index = 1:numel(total_util)
    util = total_util(util_index);
    for epoch_index = 1:epoch_num
        seed = topology_seed_matrix(epoch_index, util_index);
        [group_HFLSnF, clients_HFLSnF, ~, maps_HFLSnF, edges_HFLSnF, ...
            client_FLnoSnF, ~, map_FLnoSnF, client_FLSnF, ~, map_FLSnF, ...
            group_HFLnoSnF, clients_HFLnoSnF, ~, maps_HFLnoSnF, edges_HFLnoSnF] = ...
            varParaHFL_TSMLG_v10(TopoOption, num_layers, num_of_nodes, num_wave, ...
            util, EdgeSet, Cloud, mean_time_interval, duration, seed);

        % 策略 1 是动态边缘，策略 3 是固定边缘。
        dynamic_group_HFLSnF = group_HFLSnF{1,1};
        fixed_group_HFLSnF = group_HFLSnF{1,3};
        dynamic_group_HFLnoSnF = group_HFLnoSnF{1,1};
        fixed_group_HFLnoSnF = group_HFLnoSnF{1,3};
        dynamic_clients_HFLSnF = clients_HFLSnF{1,1};
        fixed_clients_HFLSnF = clients_HFLSnF{1,3};
        dynamic_clients_HFLnoSnF = clients_HFLnoSnF{1,1};
        fixed_clients_HFLnoSnF = clients_HFLnoSnF{1,3};

        assert(isequal(edges_HFLSnF{1,3}, EdgeSet), ...
            'HFL-SnF 的策略 3 未保存固定 EdgeSet。');
        assert(isequal(edges_HFLnoSnF{1,3}, EdgeSet), ...
            'HFL-noSnF 的策略 3 未保存固定 EdgeSet。');
        assert(fixed_group_HFLSnF <= numel(EdgeSet), ...
            'HFL-SnF 固定边缘组数超过 6。');
        assert(fixed_group_HFLnoSnF <= numel(EdgeSet), ...
            'HFL-noSnF 固定边缘组数超过 6。');

        assert_group_participant_count(dynamic_group_HFLSnF, dynamic_clients_HFLSnF, ...
            maps_HFLSnF{1,1}, valid_client_ids, 'HFL-SnF-dynamic');
        assert_group_participant_count(fixed_group_HFLSnF, fixed_clients_HFLSnF, ...
            maps_HFLSnF{1,3}, valid_client_ids, 'HFL-SnF-fixed');
        assert_group_participant_count(dynamic_group_HFLnoSnF, dynamic_clients_HFLnoSnF, ...
            maps_HFLnoSnF{1,1}, valid_client_ids, 'HFL-noSnF-dynamic');
        assert_group_participant_count(fixed_group_HFLnoSnF, fixed_clients_HFLnoSnF, ...
            maps_HFLnoSnF{1,3}, valid_client_ids, 'HFL-noSnF-fixed');
        validate_client_mapping(map_FLnoSnF, client_FLnoSnF, valid_client_ids, 'FL-noSnF');
        validate_client_mapping(map_FLSnF, client_FLSnF, valid_client_ids, 'FL-SnF');

        % 保存值由公共函数计算，再与显式公式逐单元比较。
        time_HFLSnF_dynamic(epoch_index, util_index) = calc_hfl_agg_time( ...
            dynamic_clients_HFLSnF, dynamic_group_HFLSnF, agg_time_per_clients);
        time_HFLSnF_fix(epoch_index, util_index) = calc_hfl_agg_time( ...
            fixed_clients_HFLSnF, fixed_group_HFLSnF, agg_time_per_clients);
        time_HFLnoSnF_dynamic(epoch_index, util_index) = calc_hfl_agg_time( ...
            dynamic_clients_HFLnoSnF, dynamic_group_HFLnoSnF, agg_time_per_clients);
        time_HFLnoSnF_fix(epoch_index, util_index) = calc_hfl_agg_time( ...
            fixed_clients_HFLnoSnF, fixed_group_HFLnoSnF, agg_time_per_clients);

        assert(time_HFLSnF_dynamic(epoch_index, util_index) == explicit_hfl_time( ...
            dynamic_clients_HFLSnF, dynamic_group_HFLSnF, agg_time_per_clients));
        assert(time_HFLSnF_fix(epoch_index, util_index) == explicit_hfl_time( ...
            fixed_clients_HFLSnF, fixed_group_HFLSnF, agg_time_per_clients));
        assert(time_HFLnoSnF_dynamic(epoch_index, util_index) == explicit_hfl_time( ...
            dynamic_clients_HFLnoSnF, dynamic_group_HFLnoSnF, agg_time_per_clients));
        assert(time_HFLnoSnF_fix(epoch_index, util_index) == explicit_hfl_time( ...
            fixed_clients_HFLnoSnF, fixed_group_HFLnoSnF, agg_time_per_clients));
        assert(client_FLnoSnF * agg_time_per_clients == 2 * client_FLnoSnF);
        assert(client_FLSnF * agg_time_per_clients == 2 * client_FLSnF);
    end
end

summary = struct();
summary.schema_version = '2.0-smoke';
summary.epoch_num = epoch_num;
summary.total_util = total_util;
summary.EdgeSet = EdgeSet;
summary.topology_seed_matrix = topology_seed_matrix;
summary.checked_snapshot_count = epoch_num * numel(total_util);
summary.passed = true;
fprintf('冒烟测试通过：%d 个快照，固定边缘数 %d。\n', ...
    summary.checked_snapshot_count, numel(EdgeSet));
end


function assert_group_participant_count(group_count, client_counts, mapping, valid_client_ids, context)
%ASSERT_GROUP_PARTICIPANT_COUNT 校验有效组数、参与人数和客户端映射一致。

if group_count ~= nnz(client_counts)
    error('test_result_generation_smoke:GroupMismatch', ...
        '%s 的有效组数与非零客户端组数不一致。', context);
end
participant_count = sum(client_counts(:));
validate_client_mapping(mapping, participant_count, valid_client_ids, context);
end


function aggregation_time = explicit_hfl_time(client_counts, group_count, per_model_time)
%EXPLICIT_HFL_TIME 直接按 2*max(n_e)+2*E 公式计算期望聚合时间。

if isempty(client_counts)
    max_client_count = 0;
else
    max_client_count = max(client_counts(:));
end
aggregation_time = per_model_time * max_client_count + per_model_time * group_count;
end

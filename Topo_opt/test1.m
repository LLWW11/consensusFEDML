%
clear;clc;
tic;
epoch_num =200;

TopoOption = 'Metro';
num_layers = 8;
num_of_nodes = 38;% only available for random networks
num_wave = 5;
total_util = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8];
% total_util = 0.5;
mean_time_interval = 100;
duration = 200;
% EdgeSet = [9 8 12];   %3节点
% EdgeSet = [7 9 20 27];   % 4节点
% 主实验固定使用 6 个边缘节点，与结果文件名和实验配置保持一致。
EdgeSet = [7 8 9 12 20 27]; % 6节点
% EdgeSet = [6 7 8 9 12 20 27]; % 7节点
% EdgeSet = [7 8 9 12 20 27]; % 6节点
% EdgeSet = [5 6 7 8 9 12 20 27];   % 8节点
Cloud = 18;
base_seed = 20260711;
schema_version = '3.0-variance-controlled';
variance_control_mode = 'paired_exact';
source_script = [mfilename('fullpath'), '.m'];
valid_client_ids = setdiff(1:num_of_nodes, Cloud);

% 仅测试时允许通过环境变量缩小规模，默认正式配置始终保持 200×8。
smoke_test_enabled = strcmp(getenv('TOPO_OPT_SMOKE_TEST'), '1');
if smoke_test_enabled
    epoch_num = 3;
    total_util = [0.3 0.7];
end

% 同一 epoch 的所有利用率复用相同种子，以便进行配对且单调嵌套的拓扑比较。
epoch_seed_vector = base_seed + (0:(epoch_num - 1))';
topology_seed_matrix = repmat(epoch_seed_vector, 1, length(total_util));
trace_id = cell(epoch_num, length(total_util));
for trace_util_index = 1:length(total_util)
    for trace_epoch_index = 1:epoch_num
        trace_id{trace_epoch_index, trace_util_index} = sprintf( ...
            'u%02d_epoch%03d_seed%d', round(total_util(trace_util_index) * 10), ...
            trace_epoch_index, topology_seed_matrix(trace_epoch_index, trace_util_index));
    end
end
%% 参数初始化
group_num_HFLSnF_los = zeros(epoch_num,length(total_util));
client_num_HFLSnF_los = zeros(epoch_num,length(total_util));
max_layer_HFLSnF_los = zeros(epoch_num,length(total_util));

group_num_HFLSnF = cell(epoch_num,length(total_util));
client_num_HFLSnF = cell(epoch_num,length(total_util));
max_layer_HFLSnF = cell(epoch_num,length(total_util));

group_num_HFLnoSnF_los = zeros(epoch_num,length(total_util));
client_num_HFLnoSnF_los = zeros(epoch_num,length(total_util));
max_layer_HFLnoSnF_los = zeros(epoch_num,length(total_util));

group_num_HFLnoSnF = cell(epoch_num,length(total_util));
client_num_HFLnoSnF = cell(epoch_num,length(total_util));
max_layer_HFLnoSnF = cell(epoch_num,length(total_util));

% 保存每一种方法在每个网络快照上的真实客户端映射，不能只保留参与人数。
actual_c2e_map_HFLSnF = cell(epoch_num,length(total_util));
actual_c2e_map_HFLnoSnF = cell(epoch_num,length(total_util));
DynEdgeSet_HFLSnF = cell(epoch_num,length(total_util));
DynEdgeSet_HFLnoSnF = cell(epoch_num,length(total_util));
c2cmap_FLnoSnF_all = cell(epoch_num,length(total_util));
c2cmap_FLSnF_all = cell(epoch_num,length(total_util));
topology_sampling_info = cell(epoch_num,length(total_util));

group_num_HFLSnF_fix = zeros(epoch_num,length(total_util));
client_num_HFLSnF_fix = zeros(epoch_num,length(total_util));
max_layer_HFLSnF_fix = zeros(epoch_num,length(total_util));

group_num_FLnoSnF = zeros(epoch_num,length(total_util));
client_num_FLnoSnF = zeros(epoch_num,length(total_util));
max_layer_FLnoSnF = zeros(epoch_num,length(total_util));

group_num_FLSnF = zeros(epoch_num,length(total_util));
client_num_FLSnF = zeros(epoch_num,length(total_util));
max_layer_FLSnF = zeros(epoch_num,length(total_util));

group_num_HFLnoSnF_fix = zeros(epoch_num,length(total_util));
client_num_HFLnoSnF_fix = zeros(epoch_num,length(total_util));
max_layer_HFLnoSnF_fix = zeros(epoch_num,length(total_util));

agg_time_per_clients = 2;%一个模型的聚合时间约等于2s
time_agg_FLnoSnF = zeros(epoch_num,length(total_util));
time_agg_FLSnF = zeros(epoch_num,length(total_util));
time_agg_HFLnoSnF_fix = zeros(epoch_num,length(total_util));
time_agg_HFLnoSnF_los = zeros(epoch_num,length(total_util));
time_agg_HFLSnF_fix = zeros(epoch_num,length(total_util));
time_agg_HFLSnF_los = zeros(epoch_num,length(total_util));
%% 主要
% 每个 (epoch, util) 快照彼此独立，使用线性索引让 worker 均匀领取任务。
snapshot_count = epoch_num * length(total_util);
parfor snapshot_index = 1:snapshot_count
    [i, j] = ind2sub([epoch_num, length(total_util)], snapshot_index);
    util = total_util(j);
    fprintf('epoch=%d, util=%.1f\n', i, util);

    [local_group_num_HFLSnF, local_client_num_HFLSnF, local_max_layer_HFLSnF, ...
        local_actual_c2e_map_HFLSnF, local_DynEdgeSet_HFLSnF, ...
        local_client_num_FLnoSnF, local_max_layer_FLnoSnF, local_c2cmap_FLnoSnF, ...
        local_client_num_FLSnF, local_max_layer_FLSnF, local_c2cmap_FLSnF, ...
        local_group_num_HFLnoSnF, local_client_num_HFLnoSnF, local_max_layer_HFLnoSnF, ...
        local_actual_c2e_map_HFLnoSnF, local_DynEdgeSet_HFLnoSnF, ...
        local_topology_sampling_info] = ...
        varParaHFL_TSMLG_v10(TopoOption, num_layers, num_of_nodes, num_wave, ...
        util, EdgeSet, Cloud, mean_time_interval, duration, ...
        topology_seed_matrix(snapshot_index), variance_control_mode);

    % 第 1 项对应动态策略，第 3 项对应固定边缘策略。
    local_group_num_HFLSnF_los = local_group_num_HFLSnF{1,1};
    local_group_num_HFLSnF_fix = local_group_num_HFLSnF{1,3};
    local_client_num_sum_HFLSnF = cellfun(@sum, local_client_num_HFLSnF);
    local_client_num_HFLSnF_los = local_client_num_sum_HFLSnF(1);
    local_client_num_HFLSnF_fix = local_client_num_sum_HFLSnF(3);
    local_max_layer_HFLSnF_los = local_max_layer_HFLSnF{1,1};
    local_max_layer_HFLSnF_fix = local_max_layer_HFLSnF{1,3};

    local_group_num_HFLnoSnF_los = local_group_num_HFLnoSnF{1,1};
    local_group_num_HFLnoSnF_fix = local_group_num_HFLnoSnF{1,3};
    local_client_num_HFLnoSnF_los = sum(local_client_num_HFLnoSnF{1,1});
    local_client_num_HFLnoSnF_fix = sum(local_client_num_HFLnoSnF{1,3});
    local_max_layer_HFLnoSnF_los = local_max_layer_HFLnoSnF{1,1};
    local_max_layer_HFLnoSnF_fix = local_max_layer_HFLnoSnF{1,3};

    % 四种 HFL 情况独立计算时间，空组也不能影响另一种策略。
    local_time_agg_HFLnoSnF_fix = calc_hfl_agg_time( ...
        local_client_num_HFLnoSnF{1,3}, local_group_num_HFLnoSnF_fix, agg_time_per_clients);
    local_time_agg_HFLnoSnF_los = calc_hfl_agg_time( ...
        local_client_num_HFLnoSnF{1,1}, local_group_num_HFLnoSnF_los, agg_time_per_clients);
    local_time_agg_HFLSnF_fix = calc_hfl_agg_time( ...
        local_client_num_HFLSnF{1,3}, local_group_num_HFLSnF_fix, agg_time_per_clients);
    local_time_agg_HFLSnF_los = calc_hfl_agg_time( ...
        local_client_num_HFLSnF{1,1}, local_group_num_HFLSnF_los, agg_time_per_clients);
    local_time_agg_FLnoSnF = local_client_num_FLnoSnF * agg_time_per_clients;
    local_time_agg_FLSnF = local_client_num_FLSnF * agg_time_per_clients;

    % 对每种方法的真实参与映射执行逐轮完整性校验。
    validate_client_mapping(local_c2cmap_FLnoSnF, local_client_num_FLnoSnF, ...
        valid_client_ids, sprintf('FL-noSnF epoch=%d util=%.1f', i, util));
    validate_client_mapping(local_c2cmap_FLSnF, local_client_num_FLSnF, ...
        valid_client_ids, sprintf('FL-SnF epoch=%d util=%.1f', i, util));
    validate_client_mapping(local_actual_c2e_map_HFLnoSnF{1,1}, ...
        local_client_num_HFLnoSnF_los, valid_client_ids, ...
        sprintf('HFL-noSnF-dynamic epoch=%d util=%.1f', i, util));
    validate_client_mapping(local_actual_c2e_map_HFLnoSnF{1,3}, ...
        local_client_num_HFLnoSnF_fix, valid_client_ids, ...
        sprintf('HFL-noSnF-fixed epoch=%d util=%.1f', i, util));
    validate_client_mapping(local_actual_c2e_map_HFLSnF{1,1}, ...
        local_client_num_HFLSnF_los, valid_client_ids, ...
        sprintf('HFL-SnF-dynamic epoch=%d util=%.1f', i, util));
    validate_client_mapping(local_actual_c2e_map_HFLSnF{1,3}, ...
        local_client_num_HFLSnF_fix, valid_client_ids, ...
        sprintf('HFL-SnF-fixed epoch=%d util=%.1f', i, util));

    % 所有结果均按 snapshot_index 写入，保持最终矩阵仍为 epoch×util。
    group_num_HFLSnF{snapshot_index} = local_group_num_HFLSnF;
    client_num_HFLSnF{snapshot_index} = local_client_num_HFLSnF;
    max_layer_HFLSnF{snapshot_index} = local_max_layer_HFLSnF;
    actual_c2e_map_HFLSnF{snapshot_index} = local_actual_c2e_map_HFLSnF;
    DynEdgeSet_HFLSnF{snapshot_index} = local_DynEdgeSet_HFLSnF;
    group_num_HFLnoSnF{snapshot_index} = local_group_num_HFLnoSnF;
    client_num_HFLnoSnF{snapshot_index} = local_client_num_HFLnoSnF;
    max_layer_HFLnoSnF{snapshot_index} = local_max_layer_HFLnoSnF;
    actual_c2e_map_HFLnoSnF{snapshot_index} = local_actual_c2e_map_HFLnoSnF;
    DynEdgeSet_HFLnoSnF{snapshot_index} = local_DynEdgeSet_HFLnoSnF;
    c2cmap_FLnoSnF_all{snapshot_index} = local_c2cmap_FLnoSnF;
    c2cmap_FLSnF_all{snapshot_index} = local_c2cmap_FLSnF;
    topology_sampling_info{snapshot_index} = local_topology_sampling_info;

    group_num_HFLSnF_los(snapshot_index) = local_group_num_HFLSnF_los;
    group_num_HFLSnF_fix(snapshot_index) = local_group_num_HFLSnF_fix;
    client_num_HFLSnF_los(snapshot_index) = local_client_num_HFLSnF_los;
    client_num_HFLSnF_fix(snapshot_index) = local_client_num_HFLSnF_fix;
    max_layer_HFLSnF_los(snapshot_index) = local_max_layer_HFLSnF_los;
    max_layer_HFLSnF_fix(snapshot_index) = local_max_layer_HFLSnF_fix;
    group_num_HFLnoSnF_los(snapshot_index) = local_group_num_HFLnoSnF_los;
    group_num_HFLnoSnF_fix(snapshot_index) = local_group_num_HFLnoSnF_fix;
    client_num_HFLnoSnF_los(snapshot_index) = local_client_num_HFLnoSnF_los;
    client_num_HFLnoSnF_fix(snapshot_index) = local_client_num_HFLnoSnF_fix;
    max_layer_HFLnoSnF_los(snapshot_index) = local_max_layer_HFLnoSnF_los;
    max_layer_HFLnoSnF_fix(snapshot_index) = local_max_layer_HFLnoSnF_fix;
    client_num_FLnoSnF(snapshot_index) = local_client_num_FLnoSnF;
    max_layer_FLnoSnF(snapshot_index) = local_max_layer_FLnoSnF;
    client_num_FLSnF(snapshot_index) = local_client_num_FLSnF;
    max_layer_FLSnF(snapshot_index) = local_max_layer_FLSnF;
    time_agg_HFLnoSnF_fix(snapshot_index) = local_time_agg_HFLnoSnF_fix;
    time_agg_HFLnoSnF_los(snapshot_index) = local_time_agg_HFLnoSnF_los;
    time_agg_HFLSnF_fix(snapshot_index) = local_time_agg_HFLSnF_fix;
    time_agg_HFLSnF_los(snapshot_index) = local_time_agg_HFLSnF_los;
    time_agg_FLnoSnF(snapshot_index) = local_time_agg_FLnoSnF;
    time_agg_FLSnF(snapshot_index) = local_time_agg_FLSnF;
end
%
% dirname = ['Result/',TopoOption,'_Wave',num2str(num_wave),...
%     '_Layer',num2str(num_layers),'_Cloud',num2str(Cloud),...
%     '_Edge',num2str(length(EdgeSet))];
% mkdir(dirname)
% filename = ['./',dirname,'/','','epoch_',num2str(epoch_num),'.mat'];
% save(filename);
%% 结果
enum_HFLSnF_fix = mean(group_num_HFLSnF_fix);
cnum_HFLSnF_fix = mean(client_num_HFLSnF_fix);
mlayer_HFLSnF_fix = mean(max_layer_HFLSnF_fix);
aver_time_agg_HFLSnF_fix = mean(time_agg_HFLSnF_fix);

enum_HFLSnF_los = mean(group_num_HFLSnF_los);
cnum_HFLSnF_los = mean(client_num_HFLSnF_los);
mlayer_HFLSnF_los = mean(max_layer_HFLSnF_los);
aver_time_agg_HFLSnF_los = mean(time_agg_HFLSnF_los);

cnum_FLnoSnF = mean(client_num_FLnoSnF);
mlayer_FLnoSnF = mean(max_layer_FLnoSnF);
aver_time_agg_FLnoSnF = mean(time_agg_FLnoSnF);

cnum_FLSnF  = mean(client_num_FLSnF);
mlayer_FLSnF = mean(max_layer_FLSnF);
aver_time_agg_FLSnF = mean(time_agg_FLSnF);

enum_HFLnoSnF_los = mean(group_num_HFLnoSnF_los);
cnum_HFLnoSnF_los = mean(client_num_HFLnoSnF_los);
mlayer_HFLnoSnF_los = mean(max_layer_HFLnoSnF_los);
aver_time_agg_HFLnoSnF_los = mean(time_agg_HFLnoSnF_los);

enum_HFLnoSnF_fix = mean(group_num_HFLnoSnF_fix);
cnum_HFLnoSnF_fix = mean(client_num_HFLnoSnF_fix);
mlayer_HFLnoSnF_fix = mean(max_layer_HFLnoSnF_fix);
aver_time_agg_HFLnoSnF_fix = mean(time_agg_HFLnoSnF_fix);

time_HFLSnF_los = mlayer_HFLSnF_los*16+100;
time_HFLSnF_fix = mlayer_HFLSnF_fix*16+100;
time_FLSnF = mlayer_FLSnF*16+100;
time_FLnoSnF = mlayer_FLnoSnF*16+100;
time_HFLnoSnF_fix = mlayer_HFLnoSnF_fix*16+100;
time_HFLnoSnF_los = mlayer_HFLnoSnF_los*16+100;

toall_time_HFLSnF_los = time_HFLSnF_los + aver_time_agg_HFLSnF_los;
toall_time_HFLSnF_fix = time_HFLSnF_fix + aver_time_agg_HFLSnF_fix;
toall_time_FLSnF = time_FLSnF + aver_time_agg_FLSnF;
toall_time_FLnoSnF = time_FLnoSnF + aver_time_agg_FLnoSnF;
toall_time_HFLnoSnF_fix = time_HFLnoSnF_fix + aver_time_agg_HFLnoSnF_fix;
toall_time_HFLnoSnF_los = time_HFLnoSnF_los + aver_time_agg_HFLnoSnF_los;

% 绘图保存结果

% 保存前执行跨轮约束检查，固定边缘策略不能产生超过 EdgeSet 的有效组。
assert(all(group_num_HFLSnF_fix(:) <= length(EdgeSet)), ...
    'HFL-SnF 固定边缘组数超过 EdgeSet 大小。');
assert(all(group_num_HFLnoSnF_fix(:) <= length(EdgeSet)), ...
    'HFL-noSnF 固定边缘组数超过 EdgeSet 大小。');

% 保存前验证配对种子和每层精确槽位配额，防止受控模式静默退化。
assert(all(all(topology_seed_matrix == topology_seed_matrix(:, 1))), ...
    '同一 epoch 的不同利用率未复用相同拓扑种子。');
for sampling_index = 1:numel(topology_sampling_info)
    [~, util_index] = ind2sub(size(topology_sampling_info), sampling_index);
    sampling = topology_sampling_info{sampling_index};
    expected_active_slots = round( ...
        sampling.slot_count_per_layer * (1 - total_util(util_index)));
    assert(all(sampling.active_slot_count_by_layer == expected_active_slots), ...
        '受控拓扑的实际激活槽位数与目标配额不一致。');
end

created_at = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
if smoke_test_enabled
    filename = sprintf('smoke-result-U-%dfixedge_epoch%d_variance_controlled.mat', ...
        length(EdgeSet), epoch_num);
else
    filename = sprintf('result-U-%dfixedge_epoch%d_variance_controlled.mat', ...
        length(EdgeSet), epoch_num);
end

% 显式保存分析和复现所需变量，避免把 parfor 临时变量混入结果文件。
save(filename, 'schema_version', 'source_script', 'created_at', 'base_seed', ...
    'variance_control_mode', 'smoke_test_enabled', 'topology_seed_matrix', ...
    'topology_sampling_info', 'trace_id', 'TopoOption', 'num_layers', ...
    'num_of_nodes', 'num_wave', 'total_util', 'mean_time_interval', 'duration', ...
    'EdgeSet', 'Cloud', 'epoch_num', 'agg_time_per_clients', ...
    'group_num_HFLSnF', 'client_num_HFLSnF', 'max_layer_HFLSnF', ...
    'group_num_HFLnoSnF', 'client_num_HFLnoSnF', 'max_layer_HFLnoSnF', ...
    'group_num_HFLSnF_fix', 'client_num_HFLSnF_fix', 'max_layer_HFLSnF_fix', ...
    'group_num_HFLSnF_los', 'client_num_HFLSnF_los', 'max_layer_HFLSnF_los', ...
    'group_num_HFLnoSnF_fix', 'client_num_HFLnoSnF_fix', 'max_layer_HFLnoSnF_fix', ...
    'group_num_HFLnoSnF_los', 'client_num_HFLnoSnF_los', 'max_layer_HFLnoSnF_los', ...
    'client_num_FLnoSnF', 'max_layer_FLnoSnF', ...
    'client_num_FLSnF', 'max_layer_FLSnF', ...
    'actual_c2e_map_HFLSnF', 'actual_c2e_map_HFLnoSnF', ...
    'DynEdgeSet_HFLSnF', 'DynEdgeSet_HFLnoSnF', ...
    'c2cmap_FLnoSnF_all', 'c2cmap_FLSnF_all', ...
    'time_agg_FLnoSnF', 'time_agg_FLSnF', ...
    'time_agg_HFLnoSnF_fix', 'time_agg_HFLnoSnF_los', ...
    'time_agg_HFLSnF_fix', 'time_agg_HFLSnF_los', ...
    'enum_HFLSnF_fix', 'cnum_HFLSnF_fix', 'mlayer_HFLSnF_fix', 'aver_time_agg_HFLSnF_fix', ...
    'enum_HFLSnF_los', 'cnum_HFLSnF_los', 'mlayer_HFLSnF_los', 'aver_time_agg_HFLSnF_los', ...
    'cnum_FLnoSnF', 'mlayer_FLnoSnF', 'aver_time_agg_FLnoSnF', ...
    'cnum_FLSnF', 'mlayer_FLSnF', 'aver_time_agg_FLSnF', ...
    'enum_HFLnoSnF_fix', 'cnum_HFLnoSnF_fix', 'mlayer_HFLnoSnF_fix', 'aver_time_agg_HFLnoSnF_fix', ...
    'enum_HFLnoSnF_los', 'cnum_HFLnoSnF_los', 'mlayer_HFLnoSnF_los', 'aver_time_agg_HFLnoSnF_los', ...
    'time_HFLSnF_los', 'time_HFLSnF_fix', 'time_FLSnF', 'time_FLnoSnF', ...
    'time_HFLnoSnF_fix', 'time_HFLnoSnF_los', ...
    'toall_time_HFLSnF_los', 'toall_time_HFLSnF_fix', ...
    'toall_time_FLSnF', 'toall_time_FLnoSnF', ...
    'toall_time_HFLnoSnF_fix', 'toall_time_HFLnoSnF_los');

figure(1)
plot(total_util, toall_time_HFLSnF_fix,'-d',...
    total_util, toall_time_HFLSnF_los,'-s',...
    total_util, toall_time_FLSnF,'-x',...
    total_util, toall_time_FLnoSnF,'-o',...
    total_util, toall_time_HFLnoSnF_fix,'-hexagram',...
    total_util, toall_time_HFLnoSnF_los,'-v');
title('time');
xlabel('Utilization');
ylabel('total time');
legend('HFLSnF-fix','HFLSnF-adapt','FLSnF','FLnoSnF','HFLnoSnF-fix','HFLnoSnF-adapt',Location='best');
grid on;

figure(2)
plot(total_util, enum_HFLSnF_fix,'-d',...
    total_util, enum_HFLSnF_los,'-s',....
    total_util, enum_HFLnoSnF_fix,'-hexagram',...
    total_util, enum_HFLnoSnF_los,'-v');
title('group num')
xlabel('Utilization');
ylabel('number');
legend('HFLSnF-fix','HFLSnF-adapt','HFLnoSnF-fix','HFLnoSnF-adapt');
grid on;

figure(3)
plot(total_util, cnum_HFLSnF_fix,'-d',...
    total_util, cnum_HFLSnF_los,'-s',...
    total_util, cnum_FLSnF,'-x',...
    total_util, cnum_FLnoSnF,'-o',...
    total_util, cnum_HFLnoSnF_fix,'-hexagram',...
    total_util, cnum_HFLnoSnF_los,'-v');
title('Client Number')
xlabel('Utilization');
ylabel('number');
legend('HFLSnF-fix','HFLSnF-adapt','FLSnF','FLnoSnF','HFLnoSnF-fix','HFLnoSnF-adapt',Location='best');
grid on;

% figure(4)
% plot(total_util, aver_time_agg_HFLSnF_fix,'-d',...
%     total_util, aver_time_agg_HFLSnF_los,'-s',...
%     total_util, aver_time_agg_FLSnF,'-x',...
%     total_util, aver_time_agg_FLnoSnF,'-o',...
%     total_util, aver_time_agg_HFLnoSnF_fix,'-hexagram',...
%     total_util, aver_time_agg_HFLnoSnF_los,'-v');
% title('Time of Aggregating')
% xlabel('Utilization');
% ylabel('Time of Aggregating ');
% legend('HFLSnF-fix','HFLSnF-adapt','FLSnF','FLnoSnF','HFLnoSnF-fix','HFLnoSnF-adapt',Location='best');
% grid on;

toc;

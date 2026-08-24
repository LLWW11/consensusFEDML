function summary = test_topology_variance_control()
%TEST_TOPOLOGY_VARIANCE_CONTROL 验证拓扑链路容量方差控制功能。
%   检查关闭控制与当前 paired_exact 一致、varAlpha=1 恒等、方差随
%   varAlpha 下降而不增加，以及槽位总数、整数范围和可重复性约束。

TopoOption = 'Metro';
num_of_nodes = 38;
num_layers = 8;
num_wave = 5;
percent = 0.5;
seed = 20260711;
alpha_values = [1, 0.75, 0.5, 0.25, 0];

rng(seed, 'twister');
[baseline_topology, ~, ~, baseline_info] = gen_random_tsmlg_v3( ...
    TopoOption, num_of_nodes, num_layers, num_wave, percent, ...
    'paired_exact', false, 0);
assert(~baseline_info.is_variance_control_enabled && ...
    baseline_info.effective_topology_var_alpha == 1, ...
    '关闭控制时的有效 varAlpha 应为 1。');

previous_variance = [];
for alpha_index = 1:numel(alpha_values)
    varAlpha = alpha_values(alpha_index);
    rng(seed, 'twister');
    [controlled_topology, ~, ~, sampling_info] = gen_random_tsmlg_v3( ...
        TopoOption, num_of_nodes, num_layers, num_wave, percent, ...
        'paired_exact', true, varAlpha);
    audit = sampling_info.topology_variance_audit;

    assert(all(audit.slot_sum_preserved_by_layer), ...
        '方差控制前后的层总槽位数不一致。');
    assert(all(audit.all_integer_by_layer & audit.all_in_range_by_layer), ...
        '方差控制结果不满足整数性或容量范围约束。');
    assert(isequal(sampling_info.raw_active_slot_count_by_layer, ...
        sampling_info.active_slot_count_by_layer), ...
        '方差控制改变了层总激活槽位数。');
    if varAlpha == 1
        assert(isequal(controlled_topology, baseline_topology), ...
            'varAlpha=1 时拓扑未与关闭控制的结果完全一致。');
    end
    if ~isempty(previous_variance)
        assert(all(audit.achieved_variance_by_layer <= ...
            previous_variance + 1e-12), ...
            'varAlpha 下降时出现链路容量方差增加。');
    end
    previous_variance = audit.achieved_variance_by_layer;
end

% 相同配置和种子必须逐元素复现受控拓扑。
rng(seed, 'twister');
first_topology = gen_random_tsmlg_v3( ...
    TopoOption, num_of_nodes, num_layers, num_wave, percent, ...
    'paired_exact', true, 0.5);
rng(seed, 'twister');
second_topology = gen_random_tsmlg_v3( ...
    TopoOption, num_of_nodes, num_layers, num_wave, percent, ...
    'paired_exact', true, 0.5);
assert(isequal(first_topology, second_topology), ...
    '相同种子无法复现受控拓扑。');

% 验证跨 epoch 方差摘要的基本计算接口。
example = struct();
example.client_num_example = [1 2; 2 4; 3 6];
variance_summary = build_epoch_variance_summary(example, [0.3, 0.7]);
assert(isequal(variance_summary.metrics.client_num_example.mean, [2, 4]));
assert(isequal(variance_summary.metrics.client_num_example.variance, [1, 4]));

summary = struct();
summary.schema_version = '1.0-test';
summary.alpha_values = alpha_values;
summary.seed = seed;
summary.passed = true;
fprintf('拓扑方差控制测试通过：%d 个 varAlpha，种子 %d。\n', ...
    numel(alpha_values), seed);
end

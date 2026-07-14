function summary = test_controlled_topology_generation()
%TEST_CONTROLLED_TOPOLOGY_GENERATION 验证精确槽位和配对拓扑生成行为。
%   检查目标槽位配额、不同利用率的容量嵌套、同种子可重复性，以及
%   省略模式参数时与显式 legacy 模式保持兼容。

TopoOption = 'Metro';
num_of_nodes = 38;
num_layers = 8;
num_wave = 5;
util_values = [0.1, 0.3, 0.5, 0.8];
seed = 20260711;
previous_topology = [];

for util_index = 1:numel(util_values)
    util = util_values(util_index);
    percent = 1 - util;
    rng(seed, 'twister');
    [topology, adjacency, returned_node_count, sampling_info] = ...
        gen_random_tsmlg_v3(TopoOption, num_of_nodes, num_layers, ...
        num_wave, percent, 'paired_exact');

    expected_active_slots = round( ...
        sampling_info.slot_count_per_layer * percent);
    assert(all(sampling_info.active_slot_count_by_layer == expected_active_slots), ...
        '实际激活槽位数与目标配额不一致。');
    assert(returned_node_count == num_of_nodes && ...
        isequal(size(adjacency), [num_of_nodes, num_of_nodes]), ...
        '受控生成器返回的拓扑尺寸不正确。');
    if ~isempty(previous_topology)
        assert(all(topology(:) <= previous_topology(:)), ...
            '高利用率拓扑不是低利用率拓扑的容量子集。');
    end
    previous_topology = topology;
end

% 相同种子和配置必须逐元素复现。
rng(seed, 'twister');
topology_first = gen_random_tsmlg_v3( ...
    TopoOption, num_of_nodes, num_layers, num_wave, 0.5, 'paired_exact');
rng(seed, 'twister');
topology_second = gen_random_tsmlg_v3( ...
    TopoOption, num_of_nodes, num_layers, num_wave, 0.5, 'paired_exact');
assert(isequal(topology_first, topology_second), ...
    '受控拓扑在相同种子下无法复现。');

% 默认模式必须继续等价于历史 legacy 模式。
rng(seed, 'twister');
legacy_default = gen_random_tsmlg_v3( ...
    TopoOption, num_of_nodes, num_layers, num_wave, 0.5);
rng(seed, 'twister');
legacy_explicit = gen_random_tsmlg_v3( ...
    TopoOption, num_of_nodes, num_layers, num_wave, 0.5, 'legacy');
assert(isequal(legacy_default, legacy_explicit), ...
    '默认生成模式与显式 legacy 模式不一致。');

summary = struct();
summary.schema_version = '1.0-test';
summary.seed = seed;
summary.util_values = util_values;
summary.passed = true;
fprintf('受控拓扑生成测试通过：%d 个利用率，种子 %d。\n', ...
    numel(util_values), seed);
end

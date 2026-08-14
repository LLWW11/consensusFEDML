%% 零客户端轮次修复入口脚本
% 使用方法：
% 1. 在 MATLAB 中打开本脚本。
% 2. 修改下面的 input_file 和 output_file。
% 3. 点击 MATLAB 编辑器顶部的“运行”按钮。
%
% 处理规则：
% - 六种方法分别检查客户端数量。
% - 某方法某轮客户端数量为 0 时，使用同方法、同利用率的上一轮完整数据覆盖。
% - 第一轮为 0 时，使用该列首个非零轮次回填。
% - 输入文件不会被覆盖。

clear;
clc;

%% 参数设置
% 先加入当前脚本目录，确保从任意 MATLAB 工作目录运行时都能找到辅助函数。
script_directory = fileparts(mfilename('fullpath'));
addpath(script_directory);

% 自动定位 Topo_opt 原始输入目录和 postprocess 输出目录。
paths = postprocess_paths();

% 指定需要处理的原始 MAT 文件。
input_file = fullfile(paths.topology_directory, ...
    'result-U-6fixedge_epoch200.mat');

% 指定修复后 MAT 文件的保存位置和文件名。
output_file = fullfile(paths.output_directory, ...
    'result-U-6fixedge_epoch200_zeroFilled.mat');

% 客户端覆盖模式：'preserve' 保留旧行为，'hard' 修复永久缺席客户端。
coverage_mode = 'preserve';

% 覆盖窗口轮数；超过 MAT 实际轮数时会自动截断。
coverage_horizon = 150;

%% 执行零客户端轮次修复
fprintf('开始检查零客户端轮次……\n');
fprintf('输入文件：%s\n', input_file);
fprintf('输出文件：%s\n', output_file);

audit = fill_zero_client_rounds( ...
    input_file, output_file, coverage_mode, coverage_horizon);

%% 显示处理结果
fprintf('\n处理完成。\n');
fprintf('替换单元总数：%d\n', audit.total_replacements);
fprintf('修复后零客户端单元数：%d\n', audit.remaining_zero_count);
fprintf('覆盖模式：%s\n', audit.coverage.mode);
fprintf('覆盖窗口：%d\n', audit.coverage.effective_horizon);
fprintf('覆盖交换次数：%d\n', audit.coverage.total_swaps);
fprintf('输出文件：%s\n', output_file);

if audit.all_zero_removed
    fprintf('检查通过：六种方法中已不存在客户端数量为 0 的轮次。\n');
else
    error('run_fill_zero_client_rounds:RemainingZero', ...
        '处理完成后仍存在客户端数量为 0 的轮次。');
end

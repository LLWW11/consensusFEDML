%% 一键生成方差受控且可直接训练的 MAT 文件
% 使用方法：
% 1. 在 MATLAB 中打开本脚本。
% 2. 修改“参数设置”中的 input_file 和 varAlpha。
% 3. 点击 MATLAB 编辑器顶部的“运行”按钮。
% 4. 脚本会自动修复零值、控制方差、重建映射并生成最终训练文件。

clear;
clc;

%% 参数设置
% 脚本目录用于保证从任意 MATLAB 工作目录运行都能找到输入文件和函数。
script_directory = fileparts(mfilename('fullpath'));

% 指定原始实验结果 MAT 文件。
input_file = fullfile(script_directory, ...
    'result-U-6fixedge_epoch200.mat');

% 设置目标方差比例，取值范围为 [0,1]。
% 例如 0.5 表示目标方差为零值修复后原方差的 50%。
varAlpha = 0.1;

% 客户端覆盖模式：'preserve' 保留旧行为，'hard' 保证前若干轮尽量全覆盖。
coverage_mode = 'preserve';

% 覆盖窗口轮数；当前联邦训练使用前 150 轮。
coverage_horizon = 150;

% 根据 varAlpha 自动生成清晰的最终训练文件名。
alpha_token = sprintf('%.6g', varAlpha);
alpha_token = strrep(alpha_token, '.', 'p');
alpha_token = strrep(alpha_token, '-', 'm');
output_file = fullfile(script_directory, ...
    ['result-U-6fixedge_epoch200_varAlpha_', alpha_token, '_trainable.mat']);

%% 执行完整处理流程
fprintf('开始生成方差受控训练数据。\n');
fprintf('输入文件：%s\n', input_file);
fprintf('varAlpha：%.6g\n', varAlpha);
fprintf('输出文件：%s\n\n', output_file);

audit = build_trainable_varalpha_mat(input_file, output_file, varAlpha, ...
    coverage_mode, coverage_horizon);

%% 显示最终结果
fprintf('\n全部处理完成。\n');
fprintf('零客户端替换数量：%d\n', audit.zero_fill.total_replacements);
fprintf('映射验证快照数量：%d\n', audit.validation.snapshot_count);
fprintf('覆盖模式：%s\n', audit.coverage.mode);
fprintf('覆盖窗口：%d\n', audit.coverage.effective_horizon);
fprintf('覆盖交换次数：%d\n', audit.coverage.total_swaps);
fprintf('最终文件可直接用于 HFLSnF_dynEdge 训练：\n%s\n', output_file);

function summary = run_smoke_tests()
%RUN_SMOKE_TESTS 执行 TopoOptV2 静态检查与固定人工小图测试。
%   本入口临时隔离 MATLAB 搜索路径，验证新包和冻结基线的解析位置，
%   随后运行最小费用流、耗散图和单轮 S1 至 S3 冒烟测试。函数不会调用
%   legacy/test1_varControl.m，不保存正式结果，也不生成图形。

projectRoot = fileparts(mfilename('fullpath'));
originalPath = path;
pathCleanup = onCleanup(@() path(originalPath));

% 使用干净路径防止同名旧函数或其他工作区目录污染验证结果。
restoredefaultpath;
addpath(projectRoot, '-begin');
addpath(fullfile(projectRoot, 'legacy'), '-begin');
addpath(fullfile(projectRoot, 'tests'), '-begin');
addpath(fullfile(projectRoot, 'examples'), '-begin');
rehash;

verify_resolved_path('topooptv2.run_single_round', ...
    fullfile(projectRoot, '+topooptv2'));
verify_resolved_path('varParaHFL_TSMLG_v10', ...
    fullfile(projectRoot, 'legacy'));

staticSummary = run_static_checks(projectRoot);
solverResults = runtests(fullfile( ...
    projectRoot, 'tests', 'test_min_cost_max_flow.m'));
assert(~isempty(solverResults) && all([solverResults.Passed]), ...
    '最小费用最大流人工小图测试存在失败或未完成项。');

networkSummary = test_dissipation_network();
pipelineSummary = test_topooptv2_smoke();
conditionalSummary = test_conditional_aggregation_output();
lineageSummary = test_multilevel_group_lineage();

summary = struct();
summary.schema_version = '1.0-smoke-suite';
summary.static = staticSummary;
summary.solver_test_count = numel(solverResults);
summary.network = networkSummary;
summary.pipeline = pipelineSummary;
summary.conditional_output = conditionalSummary;
summary.multilevel_lineage = lineageSummary;
summary.passed = staticSummary.passed && networkSummary.passed && ...
    pipelineSummary.passed && conditionalSummary.passed && ...
    lineageSummary.passed && all([solverResults.Passed]);
fprintf('TopoOptV2 最小验证通过：%d 个求解器用例，%d 个新文件静态检查。\n', ...
    summary.solver_test_count, staticSummary.file_count);
end


function summary = run_static_checks(projectRoot)
%RUN_STATIC_CHECKS 对新代码、测试和示例执行 MATLAB Code Analyzer。

sourceFiles = dir(fullfile(projectRoot, '+topooptv2', '*.m'));
% tests 下所有文件均做静态检查；两个冻结拓扑回归测试不会被本入口执行。
testFiles = dir(fullfile(projectRoot, 'tests', '*.m'));
exampleFiles = dir(fullfile(projectRoot, 'examples', '*.m'));
rootFile = dir(fullfile(projectRoot, 'run_smoke_tests.m'));
% dir 对单文件模式返回行向量，对通配符模式可能返回列向量；统一成列向量
% 后再拼接，避免测试文件数量变化时出现维度不一致。
files = [sourceFiles(:); testFiles(:); exampleFiles(:); rootFile(:)];

messageCount = 0;
for fileIndex = 1:numel(files)
    filePath = fullfile(files(fileIndex).folder, files(fileIndex).name);
    messages = checkcode(filePath, '-id');
    messageCount = messageCount + numel(messages);
    if ~isempty(messages)
        messageText = strjoin({messages.message}, ' | ');
        error('TopoOptV2:StaticCheckFailed', ...
            '静态检查未通过 %s：%s', filePath, messageText);
    end
end

summary = struct('file_count', numel(files), ...
    'message_count', messageCount, 'passed', true);
end


function verify_resolved_path(functionName, expectedFolder)
%VERIFY_RESOLVED_PATH 使用 which -all 确认函数首个解析位置位于指定目录。

resolvedPaths = which(functionName, '-all');
if isempty(resolvedPaths)
    error('TopoOptV2:MissingFunction', '无法解析函数 %s。', functionName);
end
if ischar(resolvedPaths)
    resolvedPaths = cellstr(resolvedPaths);
elseif isstring(resolvedPaths)
    resolvedPaths = cellstr(resolvedPaths);
end
resolvedPath = resolvedPaths{1};
resolvedFolder = fileparts(resolvedPath);
normalizedFolder = strrep(resolvedFolder, '/', filesep);
normalizedExpected = strrep(expectedFolder, '/', filesep);
if ~strcmpi(normalizedFolder, normalizedExpected)
    error('TopoOptV2:UnexpectedFunctionPath', ...
        '函数 %s 被解析到非预期位置：%s。', functionName, resolvedPath);
end
end

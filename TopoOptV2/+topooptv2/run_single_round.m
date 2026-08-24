function [solution, screening, scenario] = run_single_round(inputData, varargin)
%RUN_SINGLE_ROUND 构建一次场景并完成 S1 至 S3 的单轮搜索。
%   SOLUTION = topooptv2.run_single_round(TSML_BDWMAT, CLIENTIDS,
%   CLOUDID, CONFIG) 先构建统一时空场景，再执行聚合点预筛选和五阶段
%   搜索。CONFIG 可以省略。
%
%   SOLUTION = topooptv2.run_single_round(SCENARIO, CONFIG) 可复用已经构建
%   的场景，常用于人工小图测试。函数不保存 MAT 文件、不生成图片，也不
%   更新任何跨轮信息。
%
%   [SOLUTION, SCREENING, SCENARIO] 同时返回预筛选结果和实际使用的场景，
%   便于调用方检查路径、集合及接口来源。

if nargin < 1
    error('topooptv2:run_single_round:MissingInput', ...
        '必须提供容量张量或统一场景结构体。');
end

if is_unified_scenario(inputData)
    if numel(varargin) > 1
        error('topooptv2:run_single_round:TooManyScenarioArguments', ...
            '复用 scenario 时最多再提供一个 config。');
    end
    scenario = inputData;
    if isempty(varargin)
        config = [];
    else
        config = varargin{1};
    end
else
    if numel(varargin) < 2 || numel(varargin) > 3
        error('topooptv2:run_single_round:InvalidRawArguments', ...
            ['容量张量调用形式必须为 run_single_round(', ...
            'TSML_BdwMat, clientIds, cloudId, config)。']);
    end
    clientIds = varargin{1};
    cloudId = varargin{2};
    if numel(varargin) == 3
        config = varargin{3};
    else
        config = [];
    end
    scenario = topooptv2.build_scenario( ...
        inputData, clientIds, cloudId, config);
end

screening = topooptv2.prescreen_aggregators(scenario, config);
[solution, trace] = topooptv2.search_five_stage( ...
    scenario, screening, config);
solution.search_trace = trace;
solution.prescreening = screening;
end

function isScenario = is_unified_scenario(value)
%IS_UNIFIED_SCENARIO 判断输入是否为满足最小接口的统一场景。
requiredFields = {'edges', 'source_id', 'sink_id', 'node_count', ...
    'physical_node_count', 'layer_count', 'client_ids', 'cloud_id'};
isScenario = isstruct(value) && isscalar(value) && ...
    all(isfield(value, requiredFields)) && istable(value.edges);
end

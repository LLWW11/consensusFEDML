function [bestSolution, trace] = search_five_stage(scenario, screening, config)
%SEARCH_FIVE_STAGE 按五阶段顺序搜索单轮聚合节点集合。
%   [BESTSOLUTION, TRACE] = topooptv2.search_five_stage(SCENARIO,
%   SCREENING, CONFIG) 严格执行基准确立、候选增选、初次缩减、外围增选
%   和末次缩减。每个试探只有在效用严格提升超过 1e-9 时才提交，否则
%   保持提交前的集合和解，实现完整回滚。
%
%   SCREENING 可以省略，此时函数调用 prescreen_aggregators。若第二个
%   参数不是包含 initial/candidate/others 的预筛选结构体，则将其视为
%   CONFIG，便于直接调用 search_five_stage(SCENARIO, CONFIG)。

if nargin < 3
    config = [];
end
if nargin < 2
    screening = [];
end
if ~isempty(screening) && ~is_screening_struct(screening)
    if ~isempty(config)
        error('topooptv2:search_five_stage:AmbiguousArguments', ...
            '同时提供三个参数时，第二个参数必须是预筛选结构体。');
    end
    config = screening;
    screening = [];
end

config = resolve_search_config(scenario, config);
if isempty(screening)
    screening = topooptv2.prescreen_aggregators(scenario, config);
else
    validate_screening(screening, scenario);
end

improvementTolerance = 1e-9;
trace = repmat(empty_trace_record(), 0, 1);
traceOrder = 0;

% 第一阶段：以预筛选的初始集合建立基准解。
baselineSet = normalize_set(screening.initial, scenario);
bestSolution = topooptv2.solve_aggregation_scheme( ...
    scenario, baselineSet, config);
currentSet = bestSolution.effective_aggregators;
traceOrder = traceOrder + 1;
trace(end + 1, 1) = make_trace_record(traceOrder, ...
    'baseline', 'establish', nan, zeros(1, 0), baselineSet, ...
    currentSet, -inf, bestSolution, true);

% 第二阶段：逐个试探候选交汇点，按节点编号确定性处理。
candidateSet = normalize_set(screening.candidate, scenario);
[bestSolution, currentSet, trace, traceOrder] = run_addition_stage( ...
    'candidate_addition', candidateSet, scenario, config, bestSolution, ...
    currentSet, trace, traceOrder, improvementTolerance);

% 第三阶段：对候选增选后的整个当前集合执行第一次缩减扫描。
initialRemovalSet = currentSet;
[bestSolution, currentSet, trace, traceOrder] = run_removal_stage( ...
    'initial_reduction', initialRemovalSet, scenario, config, ...
    bestSolution, currentSet, trace, traceOrder, improvementTolerance);

% 第四阶段：逐个试探预筛选以外的外围物理节点。
peripheralSet = normalize_set(screening.others, scenario);
[bestSolution, currentSet, trace, traceOrder] = run_addition_stage( ...
    'peripheral_addition', peripheralSet, scenario, config, bestSolution, ...
    currentSet, trace, traceOrder, improvementTolerance);

% 第五阶段：对最终集合做一次完整的确定性缩减扫描。
finalRemovalSet = currentSet;
[bestSolution, currentSet, trace, ~] = run_removal_stage( ...
    'final_reduction', finalRemovalSet, scenario, config, bestSolution, ...
    currentSet, trace, traceOrder, improvementTolerance);

bestSolution.effective_aggregators = currentSet;
bestSolution.search_trace = trace;
bestSolution.search_screening = screening;
bestSolution.search_stage_count = 5;
bestSolution.improvement_tolerance = improvementTolerance;
end

function isScreening = is_screening_struct(value)
%IS_SCREENING_STRUCT 判断结构体是否满足预筛选结果的最小接口。
isScreening = isstruct(value) && isscalar(value) && ...
    all(isfield(value, {'initial', 'candidate', 'others'}));
end

function config = resolve_search_config(scenario, overrideConfig)
%RESOLVE_SEARCH_CONFIG 合并场景配置和搜索调用的覆盖配置。
if isfield(scenario, 'config') && isstruct(scenario.config)
    config = scenario.config;
else
    config = struct();
end
if ~isempty(overrideConfig)
    if ~isstruct(overrideConfig) || ~isscalar(overrideConfig)
        error('topooptv2:search_five_stage:InvalidConfig', ...
            'config 必须是标量结构体。');
    end
    overrideNames = fieldnames(overrideConfig);
    for fieldIndex = 1:numel(overrideNames)
        fieldName = overrideNames{fieldIndex};
        config.(fieldName) = overrideConfig.(fieldName);
    end
end
config = topooptv2.validate_config(config);
end

function validate_screening(screening, scenario)
%VALIDATE_SCREENING 校验外部提供的三个预筛选集合。
if ~is_screening_struct(screening)
    error('topooptv2:search_five_stage:InvalidScreening', ...
        'screening 必须包含 initial、candidate 和 others。');
end
fieldNames = {'initial', 'candidate', 'others'};
for fieldIndex = 1:numel(fieldNames)
    normalize_set(screening.(fieldNames{fieldIndex}), scenario);
end
end

function normalized = normalize_set(values, scenario)
%NORMALIZE_SET 校验、去重并排序搜索使用的物理节点集合。
if isempty(values)
    normalized = zeros(1, 0);
    return;
end
if ~isnumeric(values) || ~isreal(values) || ~isvector(values) || ...
        any(~isfinite(values)) || any(values ~= floor(values)) || ...
        any(values < 1) || any(values > scenario.physical_node_count)
    error('topooptv2:search_five_stage:InvalidNodeSet', ...
        '搜索节点集合必须是物理节点范围内的整数向量。');
end
normalized = sort(unique(double(values(:).')));
normalized(normalized == scenario.cloud_id) = [];
end

function [bestSolution, currentSet, trace, traceOrder] = ...
        run_addition_stage(stageName, trialNodes, scenario, config, ...
        bestSolution, currentSet, trace, traceOrder, tolerance)
%RUN_ADDITION_STAGE 逐点执行增选，并仅提交严格改进的试探解。
trialNodes = setdiff(trialNodes, currentSet, 'stable');
for nodeIndex = 1:numel(trialNodes)
    aggregatorId = trialNodes(nodeIndex);
    beforeSet = currentSet;
    beforeUtility = bestSolution.utility;
    trialSet = sort(unique([currentSet, aggregatorId]));
    trialSolution = topooptv2.solve_aggregation_scheme( ...
        scenario, trialSet, config);
    accepted = is_strict_improvement( ...
        bestSolution, trialSolution, tolerance);
    if accepted
        bestSolution = trialSolution;
        currentSet = trialSolution.effective_aggregators;
    end
    traceOrder = traceOrder + 1;
    trace(end + 1, 1) = make_trace_record(traceOrder, stageName, ...
        'add', aggregatorId, beforeSet, trialSet, currentSet, ...
        beforeUtility, trialSolution, accepted); %#ok<AGROW>
end
end

function [bestSolution, currentSet, trace, traceOrder] = ...
        run_removal_stage(stageName, trialNodes, scenario, config, ...
        bestSolution, currentSet, trace, traceOrder, tolerance)
%RUN_REMOVAL_STAGE 逐点执行缩减，并仅提交严格改进的试探解。
for nodeIndex = 1:numel(trialNodes)
    aggregatorId = trialNodes(nodeIndex);
    if ~ismember(aggregatorId, currentSet)
        continue;
    end
    beforeSet = currentSet;
    beforeUtility = bestSolution.utility;
    trialSet = setdiff(currentSet, aggregatorId, 'stable');
    trialSolution = topooptv2.solve_aggregation_scheme( ...
        scenario, trialSet, config);
    accepted = is_strict_improvement( ...
        bestSolution, trialSolution, tolerance);
    if accepted
        bestSolution = trialSolution;
        currentSet = trialSolution.effective_aggregators;
    end
    traceOrder = traceOrder + 1;
    trace(end + 1, 1) = make_trace_record(traceOrder, stageName, ...
        'remove', aggregatorId, beforeSet, trialSet, currentSet, ...
        beforeUtility, trialSolution, accepted); %#ok<AGROW>
end
end

function accepted = is_strict_improvement(bestSolution, trialSolution, tolerance)
%IS_STRICT_IMPROVEMENT 判断试探解是否为有限且超过阈值的严格改进。
accepted = isfinite(trialSolution.utility) && ...
    trialSolution.utility > bestSolution.utility + tolerance;
end

function record = make_trace_record(order, stageName, actionName, ...
        aggregatorId, beforeSet, trialSet, afterSet, beforeUtility, ...
        trialSolution, accepted)
%MAKE_TRACE_RECORD 构造一条完整记录提交或回滚状态的搜索轨迹。
record = empty_trace_record();
record.order = order;
record.stage = stageName;
record.action = actionName;
record.aggregator_id = aggregatorId;
record.before_set = beforeSet;
record.trial_set = trialSet;
record.after_set = afterSet;
record.before_utility = beforeUtility;
record.trial_utility = trialSolution.utility;
record.improvement = trialSolution.utility - beforeUtility;
record.accepted = accepted;
record.trial_status = trialSolution.status;
record.error_message = trialSolution.error_message;
end

function record = empty_trace_record()
%EMPTY_TRACE_RECORD 返回五阶段搜索轨迹的空结构模板。
record = struct('order', 0, 'stage', '', 'action', '', ...
    'aggregator_id', nan, 'before_set', zeros(1, 0), ...
    'trial_set', zeros(1, 0), 'after_set', zeros(1, 0), ...
    'before_utility', nan, 'trial_utility', nan, 'improvement', nan, ...
    'accepted', false, 'trial_status', '', 'error_message', '');
end

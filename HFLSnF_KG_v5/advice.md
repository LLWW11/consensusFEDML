# 数据划分方向前期评估记录

> 本文是V5实施前的方案比较材料，不是当前执行合同。文中的预估重叠区间、训练成本和建议参数均属于实施前预测，不得作为正式实验结果。当前冻结方法和实测指标以[图语义感知划分实验计划](图语义感知划分实验计划.md)及校准合同为准。

HFLSnF_KG_v4 数据划分改进计划 (Plan Mode)
1. 当前划分现状诊断
代码定位: HFLSnF_KG_v4/tasks/kge/federated_data.py:598-653 (partition_train_triples_by_head), 692-1015 (_search_overlap_partition_candidate + partition_train_triples_by_overlap_target), 1041-1190 (calibrate_entity_overlap_levels)
- 机制: FB15k-237 train 272115 全量按头实体捆绑 (head_to_row_indices, 604-612)，LPT 贪心按 head_counts 降序分配到37客户端 (628-634)。V4 新增 balanced_head_entity_overlap_target 在负载窗口内 (upper_limit=floor(mean*(1+0.05)), balance_window=floor(mean*0.05/2) 743-750) 联合优化 joint_error = entity_error/scale_E + relation_error/scale_R (811-813)，外层 8重启 (restart_seed=seed+ i*104729 904-905) + 端点夹逼二分8步 (952-975) 反解 guidance。
- 可控区间: configs/overlap/partition_calibration_contract.json:26-30 正式8重启共同可达区间 low 0.17224 / high 0.28535 / span 0.113 (校准 federated_data.py:1118-1120)，误差 <0.005 合同内，但 relation_overlap ~0.86 固化 (237 关系 RF~32 实验流程说明.md:39)，无法调关系异构。
- 校验: 实验流程说明.md:10-12, federated_data.py:103-169 三元组互斥完整 + 头实体互斥 + 实体重叠 (ρ_E=(RF_E-1)/36)，relation_overlap_tolerance 0.02, load_tolerance 0.05。
核心痛点 (为何称不上“更好”):
1. 跨度窄: 0.113 仅覆盖 0-1 的11%，low_endpoints 到顶 high 因头原子包+负载5%+关系2% 三重紧约束 (重叠率消融实验计划.md:4.2)，测不到 ρ_E→0 / →1 极端，无法回答单调性/拐点。high 0.285 仍远低 1.0。
2. 头原子性锁死多样性: 大头实体 (如 head_counts Top-100占大量 triples) 无法拆分，低重叠必须把关联尾实体强行隔离，导致 low 负载偏差 0.0142 高于 high 0.00034 (contract.json:130-166)，混淆“重叠效应 vs 负载不均”。
3. 关系同质: partition_train_triples_by_relation_stratified:1193-1264 已实现但未纳入正式矩阵 (SUPPORTED 但 experiment.py:60-110 未用)，当前所有划分 ρ_R~0.86 单点，无法解耦 实体vs关系异构。
4. ** heuristic 非最优:** 贪心 min(joint, relation_error, entity_error, load, tie_rank) 791-821 + 二分是局部搜索，无全局最优保证；8重启扩大仅 0.1125→0.1131 (overlap_calibration_validation.json vs contract.json)。
5. 缺乏图/语义真实性: 未用 FB15k-237 关系域 (location/film/sports) 或 图社团 (METIS/Louvain) 模拟机构隔离；也无经典 FL Dirichlet α 数质量化异构 (grep Dirichlet 零命中)。
6. 通信度量失真: 实验流程说明.md:306 密集上传 3.8MB/round 与 ρ_E 脱钩，当前仅报告 logical_sparse_activity_bytes (tasks/kge/overlap_ablation.py:543-565)，无法体现实体稀疏收益，对比不公平。
2. 更好划分方式分类 (按目标选型)
A. 拓宽可控区间 – 保持头互斥前提下的改进搜索
方案	原理	预期跨度	代价
A1 松弛负载/允许微拆大头包	load_tolerance 0.05→0.10 或 对 head_counts>阈值 的超大头包允许按尾实体二次切分 (仍保头互斥但特例拆分)	0.10-0.45 (+200%)	负载不均引入新混淆，需报告 cv
A2 全局优化替代贪心	将分配建模为 `min	RF_E - target	带load/ρ_R约束的 0-1 整数规划，用OR-Tools CP-SAT/ 模拟退火 / 遗传算法 替代federated_data.py:692` 贪心
A3 多目标 Pareto 前沿	不固定 0.02/0.05 单阈值，搜索 ρ_E vs ρ_R vs load_dev Pareto 集，让用户选权衡点	连续前沿	报告复杂度上升
B. 引入经典 FL Non-IID 维度 (与当前正交)
- B1 Relation-Dirichlet (α 扫描): 每个关系 r 的 triples 按 Dirichlet(α) 分给 K=37 (α=0.1 强Non-IID → 10 近IID)，复用 federated_data.py:1193 的 rng.shuffle 思路但换 np.random.dirichlet。三元组可互斥但头不再互斥 (RELATION_STRATIFIED 的 Dirichet 变体)，用于对比“头互斥 vs 关系异构谁更影响 MRR”。α∈{0.1,0.5,1.0,10} 4档 ×3种子 =12格。
- B2 Quantity-Skew: 固定 ρ_E≈0.24，让客户端 triple 数服从 LogNormal(σ) (σ=0,0.5,1.0)，检验负载不均独立效应。
- B3 Entity-Dirichlet: 按实体 (而非关系) Dirichlet 划分，模拟热门实体长尾。
此类为 FL 社区标准 (FedAvg/FedE 文献)，可直接与 FedE 随机划分 (1paperAbout/FedE-master/dataloader.py:get_all_clients) 对比，论文审稿人敏感点。
C. 图结构/语义感知划分 (最贴近真实联邦 KG)
- C1 METIS 最小割: 构实体无向图 (head-tail 边)，pymetis 划分 K=37 最小化割边 → 天然降低 ρ_E 且保留社团内聚。预期 ρ_E 0.05-0.15 可达，关系自动聚类。
- C2 Louvain 社区分层: 先 Louvain 得 ~50-100 社团，再 LPT 合并到37，类似 C1 但无需指定 K。
- C3 关系域语义划分: 按 Freebase 域 (/location/*, /film/*, /sports/* 237关系可映射 ~10域) 把 triples 按域分簇，每客户端拿1-2域为主 + 10% 交叉尾实体 可控 ρ_E。模拟“医院A主攻疾病-药物，医院B主攻地理”。实现仅需 data.py:KnowledgeGraphDataset 加 relation_to_domain 映射。
C类 ρ_E 可低至 0.05 且 ρ_R 同时降低 (不同客户端主导不同关系)，能首次研究“关系+实体双重隔离”场景，是 HFLSnF 声称解决跨域共享的关键验证。
D. 混合正交网格 (推荐最终论文形态)
ρ_E (low/med/high, A方案拓宽版) × 关系异构 (B1 α / C3域) × 负载 (均衡 vs skew) 三维，但为控制成本选 3×2 简化: 实体重叠3档 × (头互斥 vs 关系Dirichlet α=0.5) =6组合 ×3种子=18训练，或 实体重叠 × 域划分 同理。能回答“重叠效应是否在不同异构类型下稳健”。
3. 推荐实施路线图 (分阶段，复用现有合约)
Phase 0 (只读验证, 1天): 复算 calibrate_entity_overlap_levels 在 load_tolerance 0.10 下的可达区间，预估 A1 收益；统计 FB15k-237 关系域分布 (237 关系前缀统计) 评估 C3 可行性。
Phase 1 (最小侵入, 1周):
1. 在 tasks/kge/federated_data.py 新增 DIRICHLET_RELATION = "dirichlet_relation" 策略 (SUPPORTED 扩展)，实现 partition_train_triples_by_dirichlet_relation(dataset,k,seed,alpha) (参考 partition_train_triples_by_relation_stratified 结构，替换按关系 rng.permutation 为 dirichlet 分配比例 + multinomial 抽样)。
2. 复活 RELATION_STRATIFIED_TRIPLE_BALANCED 入正式对比 (作为 IID 关系基线)。
3. 加 B1 α=0.5 与 C3 域划分 各3种子 (共6新YAML)，与现有 balanced_head_entity_overlap_target 同跑 run_overlap_ablation.py:validate 流程，复用 partition_hash 校验 (federated_data.py:545-559)。
Phase 2 (拓宽重叠, 1-2周):
1. 实现 A2 模拟退火版 partition_train_triples_by_overlap_target_sa (可选 scipy.optimize 或 自写退火，目标函数同 joint_error 但允许交换头包)。
2. 校准 load_tolerance 0.08 新区间，目标 0.10 / 0.25 / 0.40 三档，生成 configs/overlap_sa/ 9YAML，独立合同 partition_calibration_contract_sa.json。
3. 若 C1/C2 选入，集成 pymetis / python-louvain 依赖 (HFLSnF_KG_v4/data/README.md:11 后新增划分依赖说明)。
Phase 3 (评估与报告, 1周):
1. 扩展 tasks/kge/overlap_ablation.py:scenarios_from_contract 支持多策略矩阵，reports/ 新增 ρ_E vs MRR 双轴图 + α 维度热力图。
2. 区分 actual_dense_upload_bytes vs logical_sparse_bytes (core/aggregation.py:392 注释口径)，若引入稀疏聚合 (RowMaskedFedAvgAggregator) 则实测节省。
涉及文件: tasks/kge/federated_data.py (核心新增 ~400L), tasks/kge/data.py (域映射), experiment.py:build_federated_data (分发新策略), configs/overlap/* (新YAML+contract), tasks/kge/overlap_ablation.py + run_overlap_partition_calibration.py (校验适配), tests/test_overlap_partition_calibration.py (新增断言)。
4. 风险与取舍
- A类 保持头互斥，论文连贯但跨度仍受图天然尾重用 (~14.5k实体平均度 272115*2/14505≈37.5) 限制，难破 ρ_E<0.08。
- B/C类 打破头互斥，与 实验流程说明.md:11 互斥保证冲突，需在 FederatedKnowledgeGraphData.__post_init__:157-169 加分支豁免，否则 ValueError。
- 训练成本: 当前9新训练待跑 (重叠率消融实验计划.md:5 12格缺9)，再加6-9格将达15-18次 150round×37client×3epoch (~2-3 GPU周)，需 run_overlap_ablation.py:formal150 的 seed42 pilot gate 扩展到每新策略。
# Search 架构设计（当前实现）
更新日期：2026-03-30

## 1. 目标与边界
Search 层负责：
1. 查询预处理与意图分析。
2. KB 检索（RAG）与 Web 触发决策。
3. Web 结果质量评估与融合策略决策。
4. 输出统一命中结果、引用、置信度与可解释 trace。

Search 层不负责最终答案生成，答案组装由 ReAct 层完成。

## 2. 当前组件
实现位置：`src/core/search/`

1. `QueryPreprocessor`
   - 产出 `core_entities / intent_terms / constraint_terms / temporal_terms`
   - 给出 `need_web_search` 和 `route_mode` 的先验提示。
2. `QueryAnalyzer`
   - 计算四类核心信号：
     - `temporal_intent_score`
     - `domain_relevance_score`
     - `oov_entity_score`
     - `kb_coverage_score`
   - 产出 `QueryAnalysis(need_web_search/reasons/route_mode/query_intent)`。
3. `RulePlanner`
   - 消费 analyzer 信号，生成 `PlannerOutput`。
   - 当前是轻量两态决策：是否需要 Web（`need_web_search`）。
4. `WebResultEvaluator`
   - 对 Web 结果计算结构化评估指标：
     - `result_count/top1_score/top3_mean/score_gap`
     - `domain_diversity/freshness_ratio/noise_ratio/conflict_detected`
5. `WebRouter`
   - 根据 analyzer + evaluator 决定 `fusion_strategy`：
     - `direct_fusion`
     - `rag_fusion`
     - `none`
6. `SearchOrchestrator`
   - 编排 RAG -> 分析 -> 路由 -> 融合 -> trace 封装全流程。

## 3. 执行链路（`SearchOrchestrator.search_with_trace`）
1. 执行一次 bootstrap planner（初始 trace）。
2. 执行 `rag_searcher.search_with_trace(query)`，得到本地命中。
3. `QueryAnalyzer.analyze(...)` 计算检索信号。
4. `Phase A` 串行补偿逻辑：
   - KB 为空 -> 强制触发 Web（`web_dominant`）。
   - KB 低置信（低于 `phase_a_rag_confidence_threshold`，默认 0.58）-> 触发 Web。
5. `RulePlanner.plan(...)` 生成当前 retrieval plan。
6. `_apply_web_routing(...)`：
   - 未开启/不需要/路由不可用时返回本地结果并记录 `skip_reason`。
   - 需要 Web 时调用 `web_searcher.search`。
   - 对结果做 `WebResultEvaluator.evaluate`。
   - 交给 `WebRouter.route` 决策融合模式。
   - 执行 `direct_fusion` 或 `rag_fusion`。
7. 统一输出：
   - `OrchestratorResult.hits`（`UnifiedSearchHit`）
   - `citations`（`build_grouped_citations`）
   - `retrieval_confidence`
   - `trace_search`

## 4. 融合策略（当前实现）
`direct_fusion` 与 `rag_fusion` 都会进入动态打分融合（`_dynamic_fuse_hits`），区别在于 Web 取样数量与返回上限。

动态融合要点：
1. 计算每条候选的多维分解分数：
   - `relevance_score`
   - `evidence_score`
   - `freshness_score`
   - `authority_score`
   - `conflict_risk`
2. 依据查询信号计算动态 `alpha`（Web 权重）：
   - 受 `temporal_intent_score`、`oov_entity_score`、`kb_gap`、`route_mode` 影响。
3. 应用硬阈值过滤：
   - 证据不足
   - 时效不足（时效查询）
   - 冲突风险过高
4. 计算 `final_score = (1-alpha)*score_kb + alpha*score_web`，排序去重后返回。

## 5. 关键 Trace 字段
Search trace 由 `build_orchestrator_trace` 统一组装，关键字段：

1. `analysis`：查询分析信号与 reason codes。
2. `planner`：plan_id/source_route/route_mode/retrieval_plan。
3. `rag`：是否执行、跳过原因。
4. `web`：
   - `requested/executed/execution_skipped/skip_reason`
   - `fusion_strategy`
   - `metrics`（含 evaluator 指标 + analysis 信号）
   - `reasons`
   - `fallback_used`
5. `orchestrator`：
   - `active_architecture=planner_orchestrator_rag_web`
   - `web_routing_owner=search_orchestrator`
6. `final_results`：最终候选快照。

## 6. 回退与失败语义
1. Web 路由不可用：`web_routing_unavailable`，返回 KB 结果。
2. Web 未启用：`web_search_disabled`，返回 KB 结果。
3. Web 查询异常：`web_search_error`（可附带 `provider_misconfigured`）。
4. Web 无结果：`web_no_results`，回退 KB。
5. 融合后为空：
   - `direct_fusion_empty`
   - `rag_fusion_empty`
   均回退 KB。

## 7. 当前版本定位
当前是 `Phase A: RAG-first + 低置信补 Web` 的可观测实现：
1. 默认成本友好（优先用 KB）。
2. 在时效/OOV/低覆盖场景自动补 Web。
3. 保留详细 trace，便于后续阈值调参与策略演进。


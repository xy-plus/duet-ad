# 工作区规则

## Git 管理

任何改动完成后都做 git 提交（用户已长期授权本工作区的 git 变更操作，无需逐次确认）。

## 失败项目复测：从中断节点续跑，不要从头再来

项目失败后要复测时，优先使用断点续跑，不要直接新建项目：

- 失败发生在 Picsart submit 之前（无 submit 痕迹）→ `POST /api/conversations/{id}/resume-pre-picsart`（body `{"confirm": true}`），复用分段、抽帧、分析、翻译、索引、图版等全部前置产物。
- 已有冻结的 H3 前置产物 → `POST /api/conversations/{id}/resume-picsart`（`{"confirm": true}`），只重提 H3。
- `submission_unknown`（Picsart submit 结果未知、无 task ID）仍会挡住续跑：先与供应商人工对账，确认未建任务/未扣费后，把 `state.json` 里 `picsart_submission_receipts` 对应回执的 `status` 从 `unknown` 改为 `rejected`，再走 resume-picsart。Seedream 侧的 rejected/submission_unknown 回执不需要人工处理：续跑会自动重置该 sheet 回执并重提（记 `seedream_receipt_reset`）。

只有续跑门禁明确拒绝且无法对账时，才新建项目重跑。

## 部署

- 主工作区服务：3301（caddy 前端）→ 3302（后端），`systemctl --user restart duet-ad1-h3-max-r2v.service`。
- release 工作区 `/home/xy/duet-ad1-h3-max-r2v-release`（分支 `release/v1-compat-3211`）：3211（前端，含对外 v1 API）→ 3212（后端），服务 `duet-ad1-h3-max-r2v-release{,-caddy}.service`。
- 重启服务会掐断在飞项目：重启后标 `process_restarted`，但满足续跑门禁的项目会在服务启动时自动排队续跑（记 `auto_resumed_after_restart`；无 accepted 提交日志佐证的 `submission_started` 回执自动降级为 rejected 后走 resume-picsart 等效路径）。部署前仍应先确认没有在跑的项目，避免浪费在飞任务的进度与费用。

## 工作方式

- 主代理只负责验收结果和与用户对齐；实施类工作（改代码、跑实验、写文档）一律拆给子代理并行执行，不要把实施工作揽在主代理手里串行做。
- 任何相互独立的事情都必须并行来做：多个子代理同时派、独立任务同时跑，不要为了"稳"做没必要的串行等待（例如一个实例的测试在跑，另一个独立实例的测试应同时开跑，不等前者结束）。

## E2E 前置门禁

- 发起 E2E 前先跑节点重放测试（release 树 `docs/node-replay-tests.md`）：先 `scripts/harvest_node_fixtures.py` 采集 fixtures，再 `pytest tests/test_node_replay_{local,external,boundary}.py -n auto`，全绿才放行 E2E。

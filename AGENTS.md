# 工作区规则

## Git 管理

任何改动完成后都做 git 提交（用户已长期授权本工作区的 git 变更操作，无需逐次确认）。

## 付费纪律：积分很贵，不随意续跑、不随意测试

- 任何会消耗平台/供应商积分的操作（confirm:true 的执行、续跑、真实生成、E2E），一律先向用户确认再动手；验收一律用 quote-only。
- 失败原因未查明时不要重试：先定位根因，再决定是否续跑。
- 唯一例外：供应商明确拒绝（回执 rejected，已确认未建任务/未扣费）的重提可以直接续跑。

## 失败项目复测：从中断节点续跑，不要从头再来

项目失败后要复测时，优先使用断点续跑，不要直接新建项目：

- 失败发生在 Picsart submit 之前（无 submit 痕迹）→ `POST /api/conversations/{id}/resume-pre-picsart`（body `{"confirm": true}`），复用分段、抽帧、分析、翻译、索引、图版等全部前置产物。
- 前置产物可信但灰块渲染参数或 H3 提示词逻辑变了 → `POST /api/conversations/{id}/resume-h3-prep`（`{"confirm": true}`），复用前置产物、重做 H3 准备（灰块视频与提示词）后重提 H3。
- succeeded 项目也可显式调用 resume-h3-prep 做参数实验；这会覆盖旧成片。
- 已有冻结的 H3 前置产物 → `POST /api/conversations/{id}/resume-picsart`（`{"confirm": true}`），只重提 H3。
- `submission_unknown`（Picsart submit 结果未知、无 task ID）仍会挡住续跑：先与供应商人工对账，确认未建任务/未扣费后，把 `state.json` 里 `picsart_submission_receipts` 对应回执的 `status` 从 `unknown` 改为 `rejected`，再走 resume-picsart。Seedream 侧的 rejected/submission_unknown 回执不需要人工处理：续跑会自动重置该 sheet 回执并重提（记 `seedream_receipt_reset`）。
- **已成功/已提交但需要修改冻结上游产物后重跑**（如注入 detect_label 修检测目标）：stock 门禁会拒（resume-pre-picsart 拒有 Picsart 痕迹的项目，resume-picsart 逐字重放旧 prompt）。手术流程（案例：项目 89f36eee，2026-09-15）：① 备份 state.json，移出 logs/picsart-* 与 output/（改名留存不删除）；② state.json 删 `picsart_submission_receipts`、`h3_prepared_artifacts`，status 改 failed（error 记操作原因）；③ 改 `enhanced_element_index`（state 与 work/ 下两份同步）；④ 因索引变化须沿图版哈希链重钉（用 app 自身函数重算，禁止手拼）：spec 的 `enhanced_element_index_sha256`、`upstream_bindings`（`_replacement_sheet_upstream_bindings`）、整个 spec（`build_replacement_sheet_spec`）、`spec_sha256` 两处、work/global-reference-sheet/spec.json 落盘、seedream.spec 回执（sha256/size）、h3_reference_images 里的 spec_sha256；⑤ 本地离线跑 `_load_pre_picsart_resume` 验证门禁全通后再调 API；⑥ 重启后端加载新代码后 `resume-pre-picsart {"confirm": true}`。

只有续跑门禁明确拒绝且无法对账时，才新建项目重跑。

## 部署

- 共享 venv 在 `/home/xy/duet-ad1/.venv`（历史路径，duet-ad1 目录已删但 venv 保留）：全部后端（3212/3222/3242/3302）的 run-backend 脚本都 exec 这里的 uvicorn，删除会导致所有后端重启即死。
- IP 证书（8.166.140.227，LE shortlived）存储在 `/home/xy/duet-video-v2/live/caddy/data`：duet-video-v2 已删，仅保留该证书骨架目录，禁止再删。443 系统服务 `duet-caddy.system.service`（root 系统管理器，xy 无权重启）自动续签；各用户级 caddy（3211/3221/3241/3301）只在启动时读证书文件，由 `duet-user-caddy-cert-refresh.timer` 每天 04:50 重启加载新证书。

- 主工作区服务：3301（caddy 前端）→ 3302（后端），`systemctl --user restart duet-ad1-h3-max-r2v.service`。
- 物品灰块检测后端：provider 链路顺序 wavespeed → sandbase → fal（实现见 `app/object_detect.py`，按顺序取第一个已配置 key 的 provider）。背景：fal 账号 2026-09-19 起被官方 Admin lock（storage upload 403）；同日接入的 sandbase.ai（同一模型 meta/sam-3-1/video-rle，`SANDBASE_API_KEY`）模型 worker 连续 generation_timeout（三次真实提交含官方示例视频全中，标定见 `data-verify/sandbase-sam31-calib-20260919/README.md`）；同日接入 WaveSpeed（wavespeed-ai/sam3-video-rle，`WAVESPEED_API_KEY`）并排在链路第一位。wavespeed 输出形态已经 2026-09-19 真实付费调用校准（标定见 `data-verify/wavespeed-calib-20260919/README.md`：outputs 为 list、rle 产物为逐帧 COCO 压缩 counts 字符串、未检出帧为全背景空串、非空 mask 端到端解析已验证通过）。**措辞敏感：prompt `handheld fan`（空格）整段零检出，`handheld-fan`（连字符）95/150 帧检出**——启用前须确认上游 detect label 措辞，必要时做连字符归一化（未实现）；多目标逗号 prompt 的多 outputs 项形态仍未真实验证；本树 `.env` 维持 `OBJECT_MASK_ENABLED=false`，mask 节点整体跳过、行为与 `object_mask_degraded` 等效（trace 记 `object_mask_disabled`）。启用前把 OBJECT_MASK_ENABLED 改回 true 并重启 3212。
- 生产：3211（caddy，含对外 v1 API）→ 3212（后端），服务 `duet-ad1-h3-max-r2v-release{,-caddy}.service`。3212 后端跑本工作树（分支 `experiment/object-gray-mask-multiref`，2026-09-16 由 online 树切入，runbook 见 `deploy/production-cutover-3212.md`，回滚锚点 online 树 HEAD `fff908d` + unit 备份 `~/.config/systemd/user/duet-ad1-h3-max-r2v-release.service.pre-cutover-20260915`），DATA_DIR 指向 `/home/xy/duet-ad1-h3-max-r2v-release/data`；3211 caddy 用 release 树的 `deploy/caddy-3211.json` 纯反代 3212，web 静态文件由后端进程从本树 `web/` 提供。online 树 `/home/xy/duet-ad1-h3-max-r2v-online` 保留不动作回滚锚点。
- 重启服务会掐断在飞项目：重启后标 `process_restarted`，但满足续跑门禁的项目会在服务启动时自动排队续跑（记 `auto_resumed_after_restart`；无 accepted 提交日志佐证的 `submission_started` 回执自动降级为 rejected 后走 resume-picsart 等效路径）。部署前仍应先确认没有在跑的项目，避免浪费在飞任务的进度与费用。

## 工作方式

- 出现异常（卡死、失败、意外行为）时，先向用户讲清楚发生了什么（现象、根因、影响面、证据），用户知情后再开始修复；修复中每个新发现的根因也要先报再修。
- 主代理只负责验收结果和与用户对齐；实施类工作（改代码、跑实验、写文档）一律拆给子代理并行执行，不要把实施工作揽在主代理手里串行做。

## E2E 前置门禁

- 发起 E2E 前先跑节点重放测试（见 `docs/node-replay-tests.md`）：先 `scripts/harvest_node_fixtures.py` 采集 fixtures，再 `NODE_REPLAY_LIVE=1 pytest tests/test_node_replay_{local,external,boundary}.py -n auto`（节点重放套件默认跳过，必须显式开启），全绿才放行 E2E。

# 工作区规则

## 结论纪律：事实与推断必须分离

任何结论性陈述（尤其是对供应商行为、机制原因的解释）必须有文档或实验事实支撑，禁止靠猜测下结论；推断必须显式标注"推断/未经证实"，事实部分注明证据出处（文档、探针凭证）。适用于回复、代码注释、文档、报告。

## Git 管理

任何改动完成后都做 git 提交（用户已长期授权本工作区的 git 变更操作，无需逐次确认）。

## 付费纪律：积分很贵，不随意续跑、不随意测试

- 任何会消耗平台/供应商积分的操作（confirm:true 的执行、续跑、真实生成、E2E），一律先向用户确认再动手；验收一律用 quote-only。
- 失败原因未查明时不要重试：先定位根因，再决定是否续跑。
- 唯一例外：供应商明确拒绝（回执 rejected，已确认未建任务/未扣费）的重提可以直接续跑。

## 失败项目复测：从中断节点续跑，不要从头再来

项目失败后要复测时，优先使用断点续跑，不要直接新建项目：

- 失败发生在 Seedance submit 之前（无 submit 痕迹）→ `POST /api/conversations/{id}/resume-pre-picsart`（body `{"confirm": true}`），复用分段、抽帧、分析、翻译、索引、图版等全部前置产物。
- 前置产物可信但灰块渲染参数或 H3 提示词逻辑变了 → `POST /api/conversations/{id}/resume-h3-prep`（`{"confirm": true}`），复用前置产物、重做 H3 准备（灰块视频与提示词）后重提 H3。失效重冻会连同删除 h3-context-ir 节点产物 `h3_prompt_origin.txt` 与 `context_ir.json`（否则旧幂等指纹会挡住新提示词重新增强）。
- succeeded 项目也可显式调用 resume-h3-prep 做参数实验；这会覆盖旧成片。
- 已有冻结的 H3 前置产物 → `POST /api/conversations/{id}/resume-picsart`（`{"confirm": true}`），只重提 Seedance；白名单素材账本里 Active 的 asset:// 引用直接复用，已清理的按内容 sha256 重新加白。
- `submission_unknown`（Seedance submit 结果未知、无 task ID）仍会挡住续跑：先与供应商人工对账，确认未建任务/未扣费后，把 `state.json` 里 `seedance_submission_receipts` 对应回执的 `status` 从 `unknown` 改为 `rejected`，再走 resume-picsart。Seedream 侧的 rejected/submission_unknown 回执不需要人工处理：续跑会自动重置该 sheet 回执并重提（记 `seedream_receipt_reset`）。
- **已成功/已提交但需要修改冻结上游产物后重跑**（如注入 detect_label 修检测目标）：stock 门禁会拒（resume-pre-picsart 拒有 Seedance 痕迹的项目，resume-picsart 逐字重放旧 prompt）。手术流程（案例：项目 89f36eee，2026-09-15）：① 备份 state.json，移出 logs/seedance-* 与 output/（改名留存不删除）；② state.json 删 `seedance_submission_receipts`、`h3_prepared_artifacts`，status 改 failed（error 记操作原因）；③ 改 `enhanced_element_index`（state 与 work/ 下两份同步）；④ 因索引变化须沿图版哈希链重钉（用 app 自身函数重算，禁止手拼）：spec 的 `enhanced_element_index_sha256`、`upstream_bindings`（`_replacement_sheet_upstream_bindings`）、整个 spec（`build_replacement_sheet_spec`）、`spec_sha256` 两处、work/global-reference-sheet/spec.json 落盘、seedream.spec 回执（sha256/size）、h3_reference_images 里的 spec_sha256；⑤ 本地离线跑 `_load_pre_picsart_resume` 验证门禁全通后再调 API；⑥ 重启后端加载新代码后 `resume-pre-picsart {"confirm": true}`。

只有续跑门禁明确拒绝且无法对账时，才新建项目重跑。

提示词审核状态 `awaiting_prompt_review`（创建时 `prompt_review=true` 的项目在 H3 冻结后暂停）不参与 auto-resume：重启后原样保留，只能由 `POST /api/conversations/{id}/prompt-review/confirm`（对外 v1 为 `/api/v1/video-generations/{id}/prompt-review/confirm`）确认后继续，确认等价走 resume-picsart 路径。

succeeded 项目编辑冻结提示词重跑用 `POST /api/conversations/{id}/regenerate`（body `{"segments":[{"segment_index":N,"prompt":"..."}]}`，仅 succeeded 且存在 prompt-review.json 时可用）：先把当前版本（成片、分段成片、各段冻结提示词、submit 证据日志，以及存在时的 h3-context-ir 产物 `context_ir.json`/`h3_prompt_origin.txt`，归档为 `prompts/segment-NNNN.context-ir.json|.origin.txt`）归档到 `output/history/v{N}/` 并记入 state 的 `generation_history`，再重建提示词、清零 Seedance 账本（已过 fail-closed 审计）后走 resume-picsart 等价路径全新提交；不 redo 分段/检测/灰块等任何前置产物。状态迁移原子且先于文件操作（先 queued+has_video=false 再归档），归档失败整体回滚并把项目标 failed（`regenerate_failed`）。历史版本成片经 `GET /api/conversations/{id}/files/history/v{N}/generated.mp4` 读取。对外 v1 API 无此功能。

## 部署

- 共享 venv 在 `/home/xy/duet-ad1/.venv`（历史路径，duet-ad1 目录只剩 venv 与 data）：生产 3212 的 run-backend 脚本 exec 这里的 uvicorn，删除会导致后端重启即死。
- IP 证书（8.166.140.227，LE shortlived）存储在 `/home/xy/duet-video-v2/live/caddy/data`：duet-video-v2 已删，仅保留该证书骨架目录，禁止再删。443 系统服务 `duet-caddy.system.service`（root 系统管理器，xy 无权重启）自动续签；3211 caddy 只在启动时读证书文件，由 `duet-user-caddy-cert-refresh.timer` 每天 04:50 重启加载新证书。

- 本工作树（experiment/seedance25-r2v）实例：3271（caddy）→ 3272（后端，DATA_DIR 为 worktree 内 `data-e2e/`），unit 为 `duet-ad1-h3-max-r2v-seedance25-e2e-327{1-caddy,2}.service`（`systemctl --user restart duet-ad1-h3-max-r2v-seedance25-e2e-3272.service`）。出片模型为 BytePlus Seedance 2.5（`dreamina-seedance-2-5-260628`，ModelArk 国际站）；全部非文字素材走 BytePlus AIGC 素材白名单（CreateAsset 预存 + GetAsset Active + asset:// 引用，实现见 `app/byteplus_assets.py`，管线集成见 `app/pipeline.py` 的 `_seedance_asset_reference`）。推理 key（`SEEDANCE_INTL_API_KEY`）与白名单 AK/SK（`BYTEPLUS_ACCESS_KEY_ID`/`BYTEPLUS_SECRET_ACCESS_KEY`）由 `deploy/run-backend-3272.sh` 从 `/home/xy/duet-ad1-h3-max-r2v/data-verify/seedance25-recon-20260929/duet.env` 逐键提取导出，禁止明文入码/入日志/入提交；合同细节见该目录 REPORT.md。`SEEDANCE_PROXY=http://127.0.0.1:7897`；PUBLIC_API_ENABLED 保持 false（按秒定价基于 Picsart 成本，对 Seedance 严重倒挂，启用前必须重定价）。3261/3262 旧实验实例（`duet-ad1-h3-max-r2v-seedance25-326{1-caddy,2}.service`）已停止并 disable（unit 文件保留可再启）。
- 生产（唯一在跑环境）：3211（caddy，含对外 v1 API）→ 3212（后端），服务 `duet-ad1-h3-max-r2v-release{,-caddy}.service`。2026-09-25 起代码与数据合一：3212 跑主工作区 `/home/xy/duet-ad1-h3-max-r2v`（分支 `main`），DATA_DIR 为主工作区 `data/`；3211 caddy 用主树 `deploy/caddy-3211.json` 纯反代 3212，web 静态文件由后端进程从主树 `web/` 提供。重启用 `systemctl --user restart duet-ad1-h3-max-r2v-release.service`。其余实验/验证环境（3221/3231/3241/3281/3291/3301 前端及对应后端与工作树）已于 2026-09-25 全部关停删除。
- 物品灰块检测后端：provider 链路顺序 fal → wavespeed → sandbase（实现见 `app/object_detect.py`，按顺序取第一个已配置 key 的 provider）。背景：fal 旧账号 2026-09-19 被官方 Admin lock，临时接入 sandbase（worker 连续 generation_timeout，不可用）与 WaveSpeed 救急；2026-09-20 换新 fal 账号后回到 fal 优先——fal 跑 SAM 3.1 且 `FAL_SAM3_THRESHOLD` 可调（默认 0.2），wavespeed 跑 SAM 3 无阈值参数且词表稍长整组零检出（标定见 `data-verify/wavespeed-calib-20260919/README.md`）。SAM 措辞敏感：实测 `person, hair`、`face`、`handheld-fan` 有效，`human`/`human face`/`human hair`/`human head` 及 6+ 词长 prompt 全零检出；pipeline 固定人物目标为 `person, hair`。本树 `.env` `OBJECT_MASK_ENABLED=true`（2026-09-20 启用）。
- 用户系统（2026-09-22 起，fastapi-users，实现见 `app/users.py`）：账号密码登录，cookie/Bearer JWT 双通道；自助注册已关闭（注册路由与前端入口已摘除），账号只来自 INITIAL_ADMIN_* 播种或管理员离线建号（`UserAuth.create_user`）。用户库在 `<DATA_DIR>/users.db`（USERS_DB_PATH 可覆盖，生产即 `/home/xy/duet-ad1-h3-max-r2v/data/users.db`）；首次启动且库为空时按本树 `.env` 的 `INITIAL_ADMIN_EMAIL`/`INITIAL_ADMIN_PASSWORD` 播种 superuser 管理员（生产为 admin@duet.com），并把存量无 `owner_id` 项目迁移归属到它（落盘一次）。旧共享口令 ACCESS_TOKEN 继续有效并映射为管理员，老脚本不用改。
- 重启服务会掐断在飞项目：重启后标 `process_restarted`，但满足续跑门禁的项目会在服务启动时自动排队续跑（记 `auto_resumed_after_restart`；无 accepted 提交日志佐证的 `submission_started` 回执自动降级为 rejected 后走 resume-picsart 等效路径）。部署前仍应先确认没有在跑的项目，避免浪费在飞任务的进度与费用。

## 工作方式

- 出现异常（卡死、失败、意外行为）时，先向用户讲清楚发生了什么（现象、根因、影响面、证据），用户知情后再开始修复；修复中每个新发现的根因也要先报再修。
- 主代理只负责验收结果和与用户对齐；实施类工作（改代码、跑实验、写文档）一律拆给子代理并行执行，不要把实施工作揽在主代理手里串行做。

## E2E 前置门禁

- 发起 E2E 前先跑节点重放测试（见 `docs/node-replay-tests.md`）：先 `scripts/harvest_node_fixtures.py` 采集 fixtures（历史项目在 trash 时用 `--projects-root` 指过去），再 `NODE_REPLAY_LIVE=1 pytest tests/test_node_replay_{local,external,boundary}.py -n auto`（节点重放套件默认跳过，必须显式开启），全绿才放行 E2E。
- boundary fixtures 的失败现场来自 `logs/boundary-snapshots/`（项目置 failed 时由 `Pipeline._persist_failure` / `ProjectStore.mark_failed` 即时固化，harvest 只从这里采集）；失败 attempt 的节点证据靠 attempt 编号递增天然保留，不需额外固化。

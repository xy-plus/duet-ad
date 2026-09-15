# 生产 3212 切线上线 runbook：online 树 → experiment/object-gray-mask-multiref

日期：2026-09-15 晚。目标：生产 3212 后端从 online 工作树切到实验分支
`experiment/object-gray-mask-multiref`（工作树
`/home/xy/duet-ad1-h3-max-r2v-object-mask`），对外 3211 入口、端口、
DATA_DIR、public v1 API 合同全部不变。

## 0. 生产启动链路现状（已核实）

| 组件 | 真相 |
| --- | --- |
| 3212 后端 unit | `duet-ad1-h3-max-r2v-release.service`（live 文件在 `/home/xy/.config/systemd/user/`） |
| 3212 代码路径 | live unit 的 `WorkingDirectory`/`ExecStart` 指向 online 工作树 `/home/xy/duet-ad1-h3-max-r2v-online`（分支 `online/pre-adaptation`），启动脚本 `deploy/run-backend-3212.sh`；repo 内 deploy/ 副本写的是 release 树，已过时，以 live unit 为准 |
| env 来源 | 启动脚本分层 source（后者覆盖前者）：`/home/xy/.config/duet-ad1/service.env` → `/home/xy/.config/claude/deepseek.env` → `/home/xy/duet-ad1-h3-max-r2v/.env`；代理由 unit 的 `Environment=HTTP(S)_PROXY=http://127.0.0.1:7897` 注入 |
| DATA_DIR | 启动脚本显式 `export DATA_DIR=/home/xy/duet-ad1-h3-max-r2v-release/data`，**不随代码树切换** |
| 3211 caddy | `duet-ad1-h3-max-r2v-release-caddy.service` 用 release 树的 `deploy/caddy-3211.json`，纯 TLS + 反代 `127.0.0.1:3212`，与代码树无关，**本次不动** |
| web 静态文件 | 由 FastAPI 进程从**代码树**的 `web/` 提供（`app/main.py` 里 `Path(__file__).parents[1] / "web"`），代码树一切web 跟着切，caddy 无感知 |
| public v1 API | 由 3212 同一进程挂载 `/api/v1`（`PUBLIC_API_ENABLED=1` + clients 文件），随代码树切换 |

## 1. 切换方案：选定与理由

两个候选方案：

| | 方案 A：新建专用于生产的工作树 | 方案 B（选定）：3212 unit 直接指向 object-mask 工作树 |
| --- | --- | --- |
| 做法 | `git worktree add` 一棵新树检出实验分支 | live unit 的 `WorkingDirectory`/`ExecStart` 改指 `/home/xy/duet-ad1-h3-max-r2v-object-mask` |
| 障碍 | 实验分支已被 object-mask 工作树检出，git 不允许同分支两处检出，必须另开分支或 detached HEAD，凭空多一个分支实体 | 无 |
| 验收一致性 | 新树与验收环境不是同一棵树 | 今晚 E2E/验收就是在本树 3242 实例上跑的，同一棵树、同一个 HEAD，方差最小 |
| 回滚 | 同样是一条 unit 指回命令 | 一条 unit 指回 online 树的命令 |
| 风险 | — | object-mask 树此后仍会有新提交；但运行中的进程只读启动时的代码，**重启前核对 HEAD == 验收 commit** 即可兜底（见 §2 第 6 步） |

结论：选方案 B。DATA_DIR 由启动脚本显式钉在 release 树，数据不随代码树走；
web/ 由后端进程从代码树提供，随 unit 一起切，无需额外操作。

## 2. 上线前 checklist（按顺序执行）

1. **确认无在飞项目**（付费纪律：重启会掐断在飞任务）：

   ```bash
   python3 -c "
   import json, glob
   hits = [p for p in glob.glob('/home/xy/duet-ad1-h3-max-r2v-release/data/projects/*/state.json')
           if json.load(open(p)).get('status') in ('queued', 'running')]
   print('IN-FLIGHT:', hits) if hits else print('no in-flight projects')
   "
   ```

   必须输出 `no in-flight projects` 才继续。本 runbook 编写时（2026-09-15）已扫描为零，上线前必须重扫。

2. **全量测试全绿**（在实验工作树上）：

   ```bash
   cd /home/xy/duet-ad1-h3-max-r2v-object-mask && /home/xy/duet-ad1/.venv/bin/python -m pytest tests -n auto
   ```

3. **节点重放测试全绿**（E2E 前置门禁，见 `docs/node-replay-tests.md`）：

   ```bash
   cd /home/xy/duet-ad1-h3-max-r2v-object-mask && /home/xy/duet-ad1/.venv/bin/python scripts/harvest_node_fixtures.py
   cd /home/xy/duet-ad1-h3-max-r2v-object-mask && /home/xy/duet-ad1/.venv/bin/python -m pytest tests/test_node_replay_local.py tests/test_node_replay_external.py tests/test_node_replay_boundary.py -n auto
   ```

4. **备份 DATA_DIR**（state 完整性兜底）：

   ```bash
   tar -C /home/xy/duet-ad1-h3-max-r2v-release -czf /home/xy/release-data-backup-20260915.tar.gz data
   ```

5. **记录回滚锚点**：记录 online 树当前 HEAD 与 live unit 文件。

   ```bash
   git -C /home/xy/duet-ad1-h3-max-r2v-online rev-parse HEAD
   cp /home/xy/.config/systemd/user/duet-ad1-h3-max-r2v-release.service /home/xy/.config/systemd/user/duet-ad1-h3-max-r2v-release.service.pre-cutover-20260915
   ```

   编写时 online HEAD 为 `fff908d`，以当时命令输出为准并记入上线记录。

6. **钉住实验分支验收 HEAD**：记录验收通过的 commit（重启前核对未被新提交顶掉）：

   ```bash
   git -C /home/xy/duet-ad1-h3-max-r2v-object-mask rev-parse HEAD
   ```

   若输出不等于验收 commit：先确认新提交已在 3242 实例回归验收，否则
   `git -C /home/xy/duet-ad1-h3-max-r2v-object-mask reset --hard <验收commit>`（需主代理确认，不得静默执行）。

7. **确认实验工作树 .env 三个键存在**（`FAL_KEY`、`FAL_PROXY=http://127.0.0.1:7897`、`MULTIREF_MODE=direct|composed`）：

   ```bash
   grep -c -E '^(FAL_KEY|FAL_PROXY|MULTIREF_MODE)=' /home/xy/duet-ad1-h3-max-r2v-object-mask/.env
   ```

   必须输出 `3`。缺哪个从 `/home/xy/duet-ad1-h3-max-r2v-object-mask/.env`
   的实验验收记录补齐；`FAL_KEY` 的值只存在于该 .env（已 gitignore），
   **禁止写入任何 git 跟踪文件**。生产启动脚本直接 source 这个 .env，
   不需要往别处复制密钥。`ARTIFACT_RETENTION=lean` 由启动脚本显式 export
   （见 §3），不依赖 .env。

## 3. 切换步骤

代码侧改动（已随本 runbook 一并提交在实验分支，无需上线时现写）：

- `deploy/run-backend-3212.sh`（本树）：生产启动脚本，在 online 版基础上
  追加 source 本树 .env（提供 `FAL_KEY`/`FAL_PROXY`/`MULTIREF_MODE`）并
  `export ARTIFACT_RETENTION=lean`；`DATA_DIR` 仍钉
  `/home/xy/duet-ad1-h3-max-r2v-release/data`，`PUBLIC_BASE_URL` 仍
  `https://8.166.140.227:3211`，端口仍 3212。
- `deploy/duet-ad1-h3-max-r2v-release.service`（本树）：unit 副本的
  `WorkingDirectory`/`ExecStart` 已改指本树，与切换后的 live unit 一致。

执行（全程不需要碰 caddy）：

```bash
cp /home/xy/duet-ad1-h3-max-r2v-object-mask/deploy/duet-ad1-h3-max-r2v-release.service /home/xy/.config/systemd/user/duet-ad1-h3-max-r2v-release.service
systemctl --user daemon-reload
systemctl --user restart duet-ad1-h3-max-r2v-release.service
systemctl --user status duet-ad1-h3-max-r2v-release.service --no-pager
journalctl --user -u duet-ad1-h3-max-r2v-release.service -n 50 --no-pager
```

注意：restart 后无 accepted 提交日志佐证的 `submission_started` 回执会自动降级为
rejected 并走续跑等效路径；§2 第 1 步已确认无在飞项目，正常应无任何自动续跑，
日志里出现 `auto_resumed_after_restart` 视为异常，立即排查。

## 4. 切换后验证（全部只读 / quote-only，禁止 confirm:true）

1. 进程与端口（未带 token 应返回 401/403，证明进程已起、端口在听）：

   ```bash
   curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:3212/api/capabilities
   ```

2. 内部 capabilities（带 web 口令，值从主代理处取，不要落盘）：

   ```bash
   curl -s -H 'Authorization: Bearer <WEB_ACCESS_TOKEN>' https://8.166.140.227:3211/api/capabilities
   ```

   预期 `replacement.max_images=4`、`defaults.add_subtitles=false`。

3. web 登录冒烟：浏览器开 `https://8.166.140.227:3211`，口令登录，列表正常加载
   （web/ 已随代码树切到实验版本，确认页面正常即证明静态文件链路正确）。

4. 对外 v1 capabilities（API key 从
   `/home/xy/.config/duet-ad1-h3-r2v/public-api-clients.json` 取）：

   ```bash
   curl -s -H 'X-API-Key: <CLIENT_API_KEY>' https://8.166.140.227:3211/api/v1/video-generations/capabilities
   ```

5. **quote-only 冒烟**（不扣费；严禁 confirm）：

   ```bash
   curl -s -X POST https://8.166.140.227:3211/api/v1/video-generations/quotes \
     -H 'X-API-Key: <CLIENT_API_KEY>' \
     -F 'source_video_url=<一条可公开访问的 https 视频URL>' \
     -F 'aspect_ratio=9:16' -F 'resolution=480p'
   ```

   返回 201 + credits 报价即通过。**不得**调 `POST /api/v1/video-generations`
   或对任何 quote 发 `{"confirm": true}`——真实生成需用户单独批准。

## 5. 回滚

代码树指回 online 即可，DATA_DIR 全程未动：

```bash
cp /home/xy/.config/systemd/user/duet-ad1-h3-max-r2v-release.service.pre-cutover-20260915 /home/xy/.config/systemd/user/duet-ad1-h3-max-r2v-release.service
systemctl --user daemon-reload
systemctl --user restart duet-ad1-h3-max-r2v-release.service
```

回滚后按 §4 第 1、2 步复验。若切换后已有新项目用实验分支产物跑了一半被重启掐断，
按 AGENTS.md「失败项目复测」走断点续跑门禁，不要从头新建。

## 6. 对外 API 兼容性声明

- **单图逐位兼容**：只传 1 张 `replacement_image` 时，创建回执/冻结输入保持
  `version: 1`，指纹（`generation_request_sha256`、source/replacement 的 sha256、
  request digest）与 online 版逐位一致；幂等键行为不变。
- **多图新增**：同一 multipart 字段 `replacement_image` 重复最多 4 次；多于 1 张时
  冻结输入落 `version: 2` + `replacement_images` 数组（`app/main.py` 创建回执处）。
  旧客户端不传多图，行为零变化。
- **无内部开关泄露**：`MULTIREF_MODE`、`ARTIFACT_RETENTION`、`FAL_*` 均为服务端
  env，不出现在 capabilities、错误码或任何 API 响应里；对外合同字段无增删。
- public v1 文档 `docs/public-api-v1.md` 已同步多图与 `add_subtitles`，以它为准。

## 7. 上线前需同步（不阻塞切换，上线后尽快补齐）

`docs/minimal-frontend-backend-api-v1.md` 与代码现状不一致处：

1. §1 capabilities 示例缺 `defaults.add_subtitles: false` 和
   `replacement.max_images: 4`（实际负载见 `app/main.py` 的 `CAPABILITY`）。
2. §2 只描述单张 `replacement_image`；代码已支持同字段重复最多 4 张。
3. §4 断言 `creation_input.version` 必须严格等于 1；多图时实际落 `version: 2`
   + `replacement_images` 数组（单图仍为 1，逐位兼容）。
4. §5 错误码表写的 `invalid_replacement_image`（413/415）在代码中不存在；
   实际为 413 `upload_too_large`、415 `unsupported_replacement_image`、图片数超限
   422 `invalid_creation_form`（online 生产代码亦如此，属文档历史遗留，非本次回归）。

## 8. 上线后收尾

- 更新本树与主工作树 AGENTS.md 部署章节：3212 已切实验分支，去掉「今晚上线后切」
  的指向句，写明新的代码树与分支。
- 同步 §7 的文档不一致项。
- online 树保留不动，作为回滚锚点，至少保留到下次上线。

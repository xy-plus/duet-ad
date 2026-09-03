# duet-ad1 · 极简创建 v1 测试部署（3213 → 3214）

本 runbook 只更新极简前端测试入口所使用的后端进程：

- 已存在的 `duet-ad1-caddy.service` 在 `:3213` 接收 HTTPS，且仅把
  `/api/*` 反代到 `127.0.0.1:3214`；
- `duet-ad1-minimal-frontend-3213.service` 从本工作树启动单进程
  uvicorn，并只监听 `127.0.0.1:3214`；
- 测试实例固定使用独立的
  `/home/xy/duet-ad1/data/test-instances/minimal-frontend-3213/data`；
- 生产入口 `:3211 → 127.0.0.1:3212`、`duet-ad1.service`、生产
  `DATA_DIR=/home/xy/duet-ad1/data` 和生产静态文件均不在本次范围内。

唯一允许重启的服务是
`duet-ad1-minimal-frontend-3213.service`。不要 reload/restart Caddy，不要
restart `duet-ad1.service`，不要安装本工作树内的 Caddy 配置，也不要更新
3211 或 3213 的静态文件。若现有 Caddy 拓扑不满足上述断言，立即停止；本
runbook 不负责修复代理。

所有部署 smoke 仅使用 GET。不得运行
`/home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next/.deploy/smoke-h3.sh`，
不得创建项目，不得调用 submit/retry，也不得为了验收触发任何付费 POST。

## 1. 固定 unit 合同

本 runbook 假定以下 unit 已经由独立变更预置在
`/home/xy/.config/systemd/user/duet-ad1-minimal-frontend-3213.service`：

```ini
[Unit]
Description=duet-ad1 3213 minimal frontend preview backend
Wants=network-online.target
After=network-online.target duet-h3-gateway.service

[Service]
Type=simple
WorkingDirectory=/home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next
EnvironmentFile=/home/xy/.config/duet-ad1/service.env
ExecStart=/usr/bin/env DATA_DIR=/home/xy/duet-ad1/data/test-instances/minimal-frontend-3213/data ENABLE_PIPELINE=1 ENABLE_H3_SUBMIT=1 ENABLE_MINIMAL_CREATION=1 CODEX_HOME=/home/xy/.codex-accounts/acct2 /home/xy/duet-ad1/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 3214
Restart=on-failure
RestartSec=5s
TimeoutStopSec=45s
UMask=0077
NoNewPrivileges=true
LockPersonality=true
RestrictSUIDSGID=true
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK
StandardOutput=journal
StandardError=journal
SyslogIdentifier=duet-ad1-minimal-frontend-3213

[Install]
WantedBy=default.target
```

`EnvironmentFile` 只提供既有秘密和通用参数；`ExecStart` 必须在启动目标进程时
显式覆盖独立 `DATA_DIR`、三个能力开关和 `CODEX_HOME`。这样测试实例不会读取
3211 的项目，也不会与生产实例争用会话锁或恢复任务。不要输出、复制或提交
`/home/xy/.config/duet-ad1/service.env` 的值。

先验证磁盘上的 unit 和 systemd 当前已加载的合同。任何一项不一致都应停止，
不要在这次代码部署中顺带改 unit：

```bash
rtk proxy /usr/bin/test -r /home/xy/.config/systemd/user/duet-ad1-minimal-frontend-3213.service
rtk proxy /usr/bin/systemd-analyze --user verify /home/xy/.config/systemd/user/duet-ad1-minimal-frontend-3213.service
rtk rg -x -F 'WorkingDirectory=/home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next' /home/xy/.config/systemd/user/duet-ad1-minimal-frontend-3213.service
rtk rg -x -F 'EnvironmentFile=/home/xy/.config/duet-ad1/service.env' /home/xy/.config/systemd/user/duet-ad1-minimal-frontend-3213.service
rtk rg -x -F 'ExecStart=/usr/bin/env DATA_DIR=/home/xy/duet-ad1/data/test-instances/minimal-frontend-3213/data ENABLE_PIPELINE=1 ENABLE_H3_SUBMIT=1 ENABLE_MINIMAL_CREATION=1 CODEX_HOME=/home/xy/.codex-accounts/acct2 /home/xy/duet-ad1/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 3214' /home/xy/.config/systemd/user/duet-ad1-minimal-frontend-3213.service
rtk proxy /usr/bin/systemctl --user show duet-ad1-minimal-frontend-3213.service --property=FragmentPath --property=WorkingDirectory --property=ExecStart --no-pager
```

## 2. 只读拓扑与依赖检查

验证现行 Caddy，而不是本工作树内的候选 Caddy 文件。下面只读配置和状态，
不会 reload 或 restart 任何服务：

```bash
rtk proxy /home/xy/duet-ad1/.venv/bin/python - <<'PY'
import json
from pathlib import Path

config = json.loads(
    Path('/home/xy/duet-ad1/.deploy/caddy/config.json').read_text()
)
servers = config['apps']['http']['servers']

def api_upstream(name: str) -> str:
    route = next(
        item
        for item in servers[name]['routes']
        if item.get('match') == [{'path': ['/api/*']}]
    )
    return route['handle'][0]['upstreams'][0]['dial']

assert servers['srv3211']['listen'] == [':3211']
assert api_upstream('srv3211') == '127.0.0.1:3212'
assert servers['srv3213']['listen'] == [':3213']
assert api_upstream('srv3213') == '127.0.0.1:3214'
print('caddy-topology-ok')
PY
rtk proxy /usr/local/libexec/duet/caddy validate --config /home/xy/duet-ad1/.deploy/caddy/config.json
rtk proxy /usr/bin/systemctl --user is-active duet-ad1-caddy.service
rtk proxy /usr/bin/systemctl --user is-active duet-ad1.service
rtk proxy /usr/bin/systemctl --user is-active duet-h3-gateway.service
```

验证本地运行依赖、ASR/声学模型、DeepSeek 凭据文件和专用 Codex 认证目录。
这些命令不调用模型或视频供应商：

```bash
rtk proxy /usr/bin/test -x /home/xy/duet-ad1/.venv/bin/python
rtk proxy /usr/bin/test -x /home/xy/duet-ad1/.venv/bin/uvicorn
rtk proxy /usr/bin/test -x /home/xy/.local/bin/ffmpeg
rtk proxy /usr/bin/test -x /home/xy/.local/bin/ffprobe
rtk proxy /usr/bin/test -x /home/xy/.local/bin/codex
rtk proxy /usr/bin/test -x /usr/bin/bwrap
rtk proxy /usr/bin/test -x /home/xy/.local/share/duet-asr/whisper.cpp-1.9.2-src/build/bin/whisper-cli
rtk proxy /usr/bin/test -s /home/xy/.local/share/duet-asr/ggml-small.bin
rtk proxy /usr/bin/test -s /home/xy/duet-ad1/models/yamnet.tflite
rtk proxy /usr/bin/test -s /home/xy/.config/claude/deepseek.env
rtk proxy /usr/bin/test -s /home/xy/.codex-accounts/acct2/auth.json
rtk proxy /usr/bin/bash -c "printf '%s  %s\n' 1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b /home/xy/.local/share/duet-asr/ggml-small.bin | /usr/bin/sha256sum -c -"
rtk proxy /usr/bin/env CODEX_HOME=/home/xy/.codex-accounts/acct2 /home/xy/.local/bin/codex login status
rtk proxy /usr/bin/env PYTHONPATH=/home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next /home/xy/duet-ad1/.venv/bin/python -c 'import ai_edge_litert, cv2, fastapi, httpx, multipart, numpy, scenedetect, uvicorn; import app.config, app.minimal_creation, app.project_progress; print("python-dependencies-ok")'
```

极简 v1 capability 还依赖共享环境文件中的 `ACCESS_TOKEN`、
`AUTODL_ART_TOKEN`、`MINIMAX_API_KEY`、`ARK_API_KEY`、
`ENABLE_MEDIAKIT_ERASE=1` 和 `VOLC_MEDIAKIT_API_KEY`。只检查键是否非空，
不要打印值：

```bash
rtk proxy /home/xy/duet-ad1/.venv/bin/python - <<'PY'
from pathlib import Path

values = {}
for raw_line in Path('/home/xy/.config/duet-ad1/service.env').read_text().splitlines():
    line = raw_line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    key, value = line.split('=', 1)
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    values[key.strip()] = value

required = (
    'ACCESS_TOKEN',
    'AUTODL_ART_TOKEN',
    'MINIMAX_API_KEY',
    'ARK_API_KEY',
    'VOLC_MEDIAKIT_API_KEY',
)
missing = [key for key in required if not values.get(key)]
assert not missing, f'missing service.env keys: {missing}'
assert values.get('ENABLE_MEDIAKIT_ERASE', '').lower() in {'1', 'true', 'yes'}
print('minimal-runtime-settings-present')
PY
```

## 3. 数据兼容与测试门禁

复用并保留同一个测试 `DATA_DIR`；重复部署不得清空、改名或重新初始化它，也
不得把生产数据复制进来。先确保目录存在，再只读列出已有项目和恢复回执：

```bash
rtk proxy /usr/bin/install -d -m 0700 /home/xy/duet-ad1/data/test-instances/minimal-frontend-3213/data
rtk proxy /usr/bin/test -w /home/xy/duet-ad1/data/test-instances/minimal-frontend-3213/data
rtk proxy /usr/bin/find /home/xy/duet-ad1/data/test-instances/minimal-frontend-3213/data -mindepth 2 -maxdepth 2 -type f -name meta.json -print
rtk proxy /usr/bin/find /home/xy/duet-ad1/data/test-instances/minimal-frontend-3213/data -type f -name attempt.json -print
```

既有 minimal creation v1 项目的 `effective_request`、`input_receipt`、已生成
台词和供应商 attempt 都是恢复依据，不迁移、不回写。历史项目缺少 v1 冻结字段
时，list/detail 仍必须可读，detail 中相关字段返回 `null`，并由后端给出合法的
`project_progress`；不能伪造 hash，也不能回退到要求用户提交台词的旧流程。

部署前必须通过文法、依赖和兼容测试。测试不得访问真实供应商：

```bash
rtk proxy /usr/bin/git -C /home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next status --short
rtk proxy /usr/bin/git -C /home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next rev-parse HEAD
rtk proxy /usr/bin/git -C /home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next diff --check
rtk proxy /home/xy/duet-ad1/.venv/bin/python -m compileall -q /home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next/app
rtk proxy /usr/bin/env PYTHONPATH=/home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next /home/xy/duet-ad1/.venv/bin/python -m pytest -q /home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next/tests/test_minimal_creation_api.py /home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next/tests/test_minimal_creation_backend.py /home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next/tests/test_minimal_creation_storage.py /home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next/tests/test_minimal_frontend_contract.py /home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next/tests/test_minimal_pipeline_contract.py /home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next/tests/test_project_progress.py
rtk proxy /usr/bin/env PYTHONPATH=/home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next /home/xy/duet-ad1/.venv/bin/python -m pytest -q /home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next/tests
```

`ENABLE_PIPELINE=1` 和 `ENABLE_H3_SUBMIT=1` 意味着 restart 后会恢复这个独立
`DATA_DIR` 中已经持久化的任务。部署前必须确认这些任务属于测试实例；不要用新的
POST 作为健康检查，也不要因状态未知而重新提交。

## 4. 最小部署

代码、unit、代理和依赖检查全部通过后，只执行下列 restart：

```bash
rtk proxy /usr/bin/systemctl --user restart duet-ad1-minimal-frontend-3213.service
rtk proxy /usr/bin/systemctl --user is-active duet-ad1-minimal-frontend-3213.service
rtk proxy /usr/bin/systemctl --user show duet-ad1-minimal-frontend-3213.service --property=MainPID --property=ExecMainStatus --property=WorkingDirectory --property=ExecStart --property=NRestarts --no-pager
rtk proxy /usr/bin/journalctl --user -u duet-ad1-minimal-frontend-3213.service --since '-2 minutes' --no-pager
rtk proxy /usr/bin/ss -ltnp 'sport = :3214'
```

本次代码部署没有 unit 变更，因此不要执行 `systemctl --user daemon-reload`。
更不能执行 Caddy 的 reload/restart 或 `duet-ad1.service` 的 restart。3214 必须仍
仅绑定 `127.0.0.1`，不得暴露为公网监听。

## 5. GET-only smoke

下面脚本只发 GET：验证 3214 本机后端、3213 既有代理和静态入口、3211 生产
回归、完整 v1 capability，以及隔离目录内所有已有项目的 list/detail 读取兼容。
访问令牌由终端无回显输入，不写 shell history，也不打印响应中的项目内容。

```bash
rtk proxy /usr/bin/env PYTHONPATH=/home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next /home/xy/duet-ad1/.venv/bin/python - <<'PY'
import getpass

import httpx

token = getpass.getpass('3213 ACCESS_TOKEN: ')
assert token, 'ACCESS_TOKEN is required'
headers = {'Authorization': f'Bearer {token}'}

with httpx.Client(verify=False, timeout=30.0) as client:
    for url in (
        'http://127.0.0.1:3214/api/health',
        'https://127.0.0.1:3213/api/health',
        'https://127.0.0.1:3211/api/health',
    ):
        response = client.get(url)
        response.raise_for_status()
        assert response.json() == {'ok': True}

    frontend = client.get('https://127.0.0.1:3213/')
    frontend.raise_for_status()

    response = client.get(
        'https://127.0.0.1:3213/api/capabilities', headers=headers
    )
    response.raise_for_status()
    capability = response.json().get('minimal_creation')
    assert isinstance(capability, dict)
    assert capability.get('supported') is True
    assert capability.get('version') == 1
    assert capability.get('endpoint') == '/api/conversations'
    assert capability.get('encoding') == 'multipart/form-data'

    response = client.get(
        'https://127.0.0.1:3213/api/conversations', headers=headers
    )
    response.raise_for_status()
    projects = response.json()
    assert isinstance(projects, list)
    v1_projects = 0
    for project in projects:
        assert isinstance(project.get('has_video'), bool)
        progress = project.get('project_progress')
        assert isinstance(progress, dict)
        assert progress.get('status') in {
            'queued', 'running', 'succeeded', 'failed'
        }
        assert isinstance(progress.get('percent'), int)
        assert 0 <= progress['percent'] <= 100

        detail = client.get(
            'https://127.0.0.1:3213/api/conversations/' + project['id'],
            headers=headers,
        )
        detail.raise_for_status()
        payload = detail.json()
        assert payload.get('id') == project['id']
        assert isinstance(payload.get('has_video'), bool)
        assert isinstance(payload.get('project_progress'), dict)
        assert 'effective_request' in payload
        assert 'input_receipt' in payload
        effective_request = payload['effective_request']
        if (
            isinstance(effective_request, dict)
            and effective_request.get('version') == 1
        ):
            input_receipt = payload['input_receipt']
            assert isinstance(input_receipt, dict)
            assert input_receipt.get('version') == 1
            v1_projects += 1

print(f'get-smoke-ok projects={len(projects)} v1_projects={v1_projects}')
PY
```

成功标准：所有请求为 2xx，3214/3213/3211 health 均为 `{"ok":true}`，
`minimal_creation` 精确宣告 v1，且已有项目全部能通过 GET list/detail 读取。若失败，
不要用 POST 探测，也不要动 3211 或 Caddy；进入下一节关闭 capability。

## 6. 回滚：先关闭 capability

回滚的第一步不是切代码、删数据或回滚前端，而是停止接受新的 v1 创建：仅把
`/home/xy/.config/systemd/user/duet-ad1-minimal-frontend-3213.service` 的
`ExecStart` 中 `ENABLE_MINIMAL_CREATION=1` 改为
`ENABLE_MINIMAL_CREATION=0`。必须保留独立 `DATA_DIR`、`ENABLE_PIPELINE=1`、
`ENABLE_H3_SUBMIT=1` 和 `CODEX_HOME`；这样 capability 消失，新 v1 POST 被拒绝，
但已经接受的 v1 项目仍可读取和恢复。

该单行 unit 变更需先审核，再只重载 user manager 配置并重启同一个测试 service。
`daemon-reload` 不会 reload/restart Caddy，但仍不得夹带其他 unit 变更：

```bash
rtk proxy /usr/bin/systemd-analyze --user verify /home/xy/.config/systemd/user/duet-ad1-minimal-frontend-3213.service
rtk proxy /usr/bin/systemctl --user daemon-reload
rtk proxy /usr/bin/systemctl --user restart duet-ad1-minimal-frontend-3213.service
rtk proxy /usr/bin/systemctl --user is-active duet-ad1-minimal-frontend-3213.service
rtk proxy /usr/bin/journalctl --user -u duet-ad1-minimal-frontend-3213.service --since '-2 minutes' --no-pager
```

随后只用 GET 证明 capability 已关闭而历史项目仍可读：

```bash
rtk proxy /home/xy/duet-ad1/.venv/bin/python - <<'PY'
import getpass

import httpx

token = getpass.getpass('3213 ACCESS_TOKEN: ')
assert token, 'ACCESS_TOKEN is required'
headers = {'Authorization': f'Bearer {token}'}

with httpx.Client(verify=False, timeout=30.0) as client:
    health = client.get('https://127.0.0.1:3213/api/health')
    health.raise_for_status()
    assert health.json() == {'ok': True}

    capabilities = client.get(
        'https://127.0.0.1:3213/api/capabilities', headers=headers
    )
    capabilities.raise_for_status()
    assert 'minimal_creation' not in capabilities.json()

    projects = client.get(
        'https://127.0.0.1:3213/api/conversations', headers=headers
    )
    projects.raise_for_status()
    listed = projects.json()
    assert isinstance(listed, list)
    for project in listed:
        detail = client.get(
            'https://127.0.0.1:3213/api/conversations/' + project['id'],
            headers=headers,
        )
        detail.raise_for_status()
        payload = detail.json()
        assert 'effective_request' in payload
        assert 'input_receipt' in payload

print(f'capability-off-history-readable projects={len(listed)}')
PY
```

只有确认 capability 已消失后，才允许评估后端代码回滚；优先修复前进。任何回滚
候选都必须继续识别已落盘的 v1 `effective_request`、`input_receipt`、进度和恢复
回执，不能切到不理解 v1 数据的版本。不要删除或改指测试 `DATA_DIR`，不要恢复成
要求用户补台词/校对的流程。前端严格依赖 capability，应在 capability 缺失时自行
fail closed，因此本回滚不需要也不允许修改 Caddy 或 3211。

修复完成后，只有重新通过第 2、3、5 节，才可把同一行恢复为
`ENABLE_MINIMAL_CREATION=1`，执行一次 user `daemon-reload` 并仍只重启
`duet-ad1-minimal-frontend-3213.service`。

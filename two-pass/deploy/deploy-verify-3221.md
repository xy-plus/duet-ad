# verify 实例（3221/3222）部署说明

release 树的第三套实例，用于独立验证，与在跑的 3211/3212 实例完全隔离：

- 3221：caddy HTTPS 前端 + 对外 v1 API（复用同一 IP 证书，配置 `caddy-3221.json`）
- 3222：后端 uvicorn，监听 127.0.0.1（脚本 `run-backend-3222.sh`）
- 数据目录独立：`/home/xy/duet-ad1-h3-max-r2v-release/data-verify/`（unit 的 `Environment=DATA_DIR=` 与脚本内 source `.env` 之后的 re-export 双重保证；`.env` 里指向主树 data 的值不会生效）
- 积分台账随 DATA_DIR 独立：台账落在 `DATA_DIR/.public-api/credit-events/`，所以 verify 实例的余额与 3211 实例互不影响
- 客户端 key 认证文件复用 `/home/xy/.config/duet-ad1-h3-r2v/public-api-clients.json`（同一批 test-001..test-020 key 可以认证，但余额各自独立）
- `PUBLIC_BASE_URL=https://8.166.140.227:3221`：签名素材 URL 直接铸在 3221 端口上

## 安装 unit

```bash
systemctl --user link /home/xy/duet-ad1-h3-max-r2v-release/deploy/duet-ad1-h3-max-r2v-verify.service
systemctl --user link /home/xy/duet-ad1-h3-max-r2v-release/deploy/duet-ad1-h3-max-r2v-verify-caddy.service
systemctl --user daemon-reload
```

（也可以 `cp` 到 `~/.config/systemd/user/`；现有 3211/3212 unit 就是 `~/.config/systemd/user/` 下的普通文件。）

## 启动顺序

caddy unit 声明了 `Requires=`/`After=` 后端，所以只启动 caddy 即可拉起整套：

```bash
systemctl --user enable --now duet-ad1-h3-max-r2v-verify.service
systemctl --user enable --now duet-ad1-h3-max-r2v-verify-caddy.service
```

验证：

```bash
systemctl --user status duet-ad1-h3-max-r2v-verify.service duet-ad1-h3-max-r2v-verify-caddy.service
curl -k https://8.166.140.227:3221/api/capabilities   # 或带 Bearer 登录后的任一 GET
```

停止时先停 caddy 再停后端。注意：重启后端会掐断在飞项目（标 `process_restarted`），启动前确认 verify 实例没有在跑的项目。

## 初始化测试积分

verify 实例积分台账为空，测试账号需要先充值。以下命令由主代理或 E2E 代理在**服务启动后**执行（本文件只是模板，部署时未执行）：

```bash
cd /home/xy/duet-ad1-h3-max-r2v-release
DATA_DIR=/home/xy/duet-ad1-h3-max-r2v-release/data-verify \
/home/xy/duet-ad1/.venv/bin/python - <<'PY'
import csv, os
from pathlib import Path
from app.public_credits import adjust

data_dir = Path(os.environ["DATA_DIR"])
CREDITS = 100000  # 100 积分 = 1 CNY；按需调整
with open(
    "/home/xy/.config/duet-ad1/public-api-test-accounts/test-001-to-test-020-api-keys.csv",
    encoding="utf-8-sig",  # 文件带 BOM
) as f:
    for row in csv.DictReader(f):
        owner = row["username"].strip()
        adjust(
            data_dir,
            owner,
            CREDITS,
            reason="verify instance initial credits",
            idempotency_key=f"verify-init-{owner}",  # 幂等：重复执行不会重复充值
        )
        print(owner, "credited", CREDITS)
PY
```

说明：

- `adjust()` 带幂等键（8–64 位 `[A-Za-z0-9_-]`），同键重复执行返回 False、不重复入账，可安全重跑。
- 台账目录 `DATA_DIR/.public-api/` 由后端自己创建，权限 0700，不要手工建文件。
- 只给需要的账号充值时，把循环换成显式的 owner 列表即可。

## Picsart 素材回源（443 资产路由结论）

后端在 `app/main.py:547` 用 `{PUBLIC_BASE_URL}/internal/provider-assets/...` 铸造给 Picsart 的签名素材 URL。

现有 443 边缘 `/home/xy/.config/duet/Caddyfile` 里有两条签名素材通道：

- `/h3-r2v-assets/internal/provider-assets/*` → 127.0.0.1:3302（主工作区；3302 的 PUBLIC_BASE_URL 走共享 443 + 路径前缀，所以必须挂这条）
- `/h3-r2v-release-assets/internal/provider-assets/*` → 127.0.0.1:3212（release；作为兜底存在）

**结论：verify 实例直接用 3221 端口给 Picsart 供素材，不需要挂 443。** release 实例的 commit `7b34a09` 已经把 PUBLIC_BASE_URL 指到专用 3211 端口并经真实付费 E2E 验证——Picsart 可以回源非标准 HTTPS 端口（证书是同一 IP 证书，3221 的 caddy 配置原样复用）。3221 同理，素材 URL 落在 3221 的 caddy 上，整条链路不经过 443。

**例外**：只有在发现 Picsart 侧无法访问 3221（防火墙/供应商限制非标准端口）时，才需要 443 兜底。届时：

1. 在 `/home/xy/.config/duet/Caddyfile` 的 443 server 块里、`/h3-r2v-release-assets` 路由旁边新增（部署时未改原文件，片段备查）：

   ```
   # H3 R2V verify（3222）的供应商素材端点，走同一 443 签名素材通道。
   @h3_r2v_verify_assets path /h3-r2v-verify-assets/internal/provider-assets/*
   handle @h3_r2v_verify_assets {
       uri strip_prefix /h3-r2v-verify-assets
       reverse_proxy 127.0.0.1:3222
   }
   ```

2. 同时把 `run-backend-3222.sh` 的 `PUBLIC_BASE_URL` 改为 `https://8.166.140.227/h3-r2v-verify-assets`（无尾斜杠），重启 verify 后端，让新铸造的 URL 走 443 前缀。

   注意：切换 PUBLIC_BASE_URL 只影响之后铸造的 URL；已提交给 Picsart 的项目里的旧 URL 不会变。

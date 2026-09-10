# 3211 字幕与人物归并上线记录

2026-09-10，候选 `8fec854` 经独立本地回归 679 项通过后上线；不包含总体改编功能。

- 双亲整合提交 `4f56e5b` 保留旧 main 历史，其 tree 与候选一致（`4bfd9679db781f294b79370c87a1b83f96db7e79`）。新增部署线程配置后，实际启动应用提交为 `de0400b`。
- main 检出于 `/home/xy/duet-ad1-h3-max-r2v-production-candidate`；online 快进至 main。
- 原主目录 `/home/xy/duet-ad1-h3-max-r2v` 保留 `archive/main-before-production-20260910`（`a92673b`），旧线上保留引用 `archive/online-before-production-20260910`（`692e3de`）。3302 PID 139594 未变，磁盘树未变。
- 重启前生产 22 项目均成功，没有在飞任务。仅重启生产后端，Caddy 随依赖联动；新 3212/3211 PID 为 818478/818479。3232 PID 776918 未变。
- 后端 cwd 仍为 `/home/xy/duet-ad1-h3-max-r2v-online`；生产数据仍为 `/home/xy/duet-ad1-h3-max-r2v-release/data`，原客户端及供应商凭证路径不变。
- Python 3.12.3 / OpenCV 4.14.0 实际加载 YuNet、SFace 成功，阈值 0.73、算法指纹 `yunet-sface-compatible-track-groups-v2`；OpenCV 与数值库线程为 2。Noto Sans CJK SC 字体检测通过。
- SFace SHA256：`0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`；模型与许可证已部署在 online/models，本记录不纳入模型二进制。
- 公网 HTTPS 首页严格 TLS 校验 200；公开 OpenAPI 1.2.0，创建及报价均公开 `add_subtitles`，默认 false，无总体改编字段。HTTP 获取的页面与脚本确认字幕初始未选、没有总体改编入口；未另跑真实浏览器自动化。
- 无生成 smoke：1 秒视频报价 201、60 积分，报价 `vq_c243ac1a951a6bc3be20cfe727af20c8`；GET 200，`confirm:false` 422；未发送 `confirm:true`。
- 部署前后及报价后，22 个原项目状态和 `.public-api` 下账本等共 92 文件聚合 SHA256 均为 `bd780d2b14ced279813547db384750ce3deec3858be718d214eee9b28ee0f415`，项目和积分未变化。新建的未确认报价保留供审计。

本记录提交随后快进到 online；仅文档变化，无需再次重启。

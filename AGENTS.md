# 工作区规则

## Git 管理

任何改动完成后都做 git 提交（用户已长期授权本工作区的 git 变更操作，无需逐次确认）。

## 失败项目复测：从中断节点续跑，不要从头再来

项目失败后要复测时，优先使用断点续跑，不要直接新建项目：

- 失败发生在 Picsart submit 之前（无 submit 痕迹）→ `POST /api/conversations/{id}/resume-pre-picsart`（body `{"confirm": true}`），复用分段、抽帧、分析、翻译、索引、图版等全部前置产物。
- 已有冻结的 H3 前置产物 → `POST /api/conversations/{id}/resume-picsart`（`{"confirm": true}`），只重提 H3。
- `submission_unknown`（submit 结果未知）会挡住续跑：先与供应商人工对账，确认未建任务/未扣费后，把 `state.json` 里 `picsart_submission_receipts` 对应回执的 `status` 从 `unknown` 改为 `rejected`，再走 resume-picsart。

只有续跑门禁明确拒绝且无法对账时，才新建项目重跑。

## 部署

- 主工作区服务：3301（caddy 前端）→ 3302（后端），`systemctl --user restart duet-ad1-h3-max-r2v.service`。
- release 工作区 `/home/xy/duet-ad1-h3-max-r2v-release`（分支 `release/v1-compat-3211`）：3211（前端，含对外 v1 API）→ 3212（后端），服务 `duet-ad1-h3-max-r2v-release{,-caddy}.service`。
- 重启服务会掐断在飞项目（重启后标记失败），部署前先确认没有在跑的项目。

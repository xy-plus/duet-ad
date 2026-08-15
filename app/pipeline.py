"""处理流水线：queued → processing → done|failed。

步骤：extract_keyframes 抽 40 检查帧 → codex 沙箱选帧/写 prompt/dry-run →
后端白名单校验（不信任 agent 输出）→ ffmpeg 合成 15s 占位预览 → meta 落盘。
流水线复用 skills/seedance-cleaning-video-maker 的两个脚本，不重造。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from app import storage
from app.codex_runner import clean_stderr
from app.config import Settings

ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = ROOT / "skills" / "seedance-cleaning-video-maker"
EXTRACT_SCRIPT = SKILL_DIR / "scripts" / "extract_keyframes.py"
SEEDANCE_SCRIPT = SKILL_DIR / "scripts" / "seedance_task.py"
SKILL_MD = SKILL_DIR / "SKILL.md"
REF_COMPRESSION = SKILL_DIR / "references" / "prompt-and-compression.md"
REF_PATTERNS = SKILL_DIR / "references" / "proven-patterns.md"

MAX_PROMPT_BYTES = 32 * 1024
_SECRET_KEY_MARKERS = ("authorization", "token", "api_key", "secret")


class PipelineError(RuntimeError):
    """流水线单步失败（HTTP 层不感知，只进 meta.error）。"""


def _run_cmd(argv: list[str], *, timeout: int, step: str, cwd: Path | None = None) -> None:
    """argv 列表子进程；超时/找不到可执行/非零退出 → PipelineError（stderr 已清洗）。"""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:
        raise PipelineError(f"{step} timed out after {timeout}s") from None
    except FileNotFoundError as e:
        raise PipelineError(f"{step} executable not found: {e.filename}") from None
    if proc.returncode != 0:
        raise PipelineError(f"{step} exit {proc.returncode}: {clean_stderr(proc.stderr)}")


def _find_secret_key(node) -> str | None:
    """递归扫 JSON：任何层的字段名含 authorization/token/api_key/secret（不扫描值）。"""
    if isinstance(node, dict):
        for key, value in node.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in _SECRET_KEY_MARKERS):
                return str(key)
            found = _find_secret_key(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_secret_key(value)
            if found is not None:
                return found
    return None


def validate_work_dir(work: Path) -> tuple[list[str], str]:
    """产物白名单校验；返回 (关键帧文件名列表, prompt 文本)。任一不过 → PipelineError。"""
    frames = sorted(
        p.name
        for p in (work / "keyframes").glob("*.png")
        if "keyframe" in p.name
    ) if (work / "keyframes").is_dir() else []
    if not 1 <= len(frames) <= 9:
        raise PipelineError(f"keyframe count {len(frames)} not in 1..9")

    prompt_path = work / "seedance_prompt.txt"
    if not prompt_path.is_file():
        raise PipelineError("seedance_prompt.txt missing")
    raw = prompt_path.read_bytes()
    if not raw.strip():
        raise PipelineError("seedance_prompt.txt empty")
    if len(raw) > MAX_PROMPT_BYTES:
        raise PipelineError(f"seedance_prompt.txt exceeds {MAX_PROMPT_BYTES} bytes")
    prompt = raw.decode("utf-8", errors="replace")

    if not (work / "shot_timeline.md").is_file():
        raise PipelineError("shot_timeline.md missing")

    req_path = work / "api_request.json"
    try:
        req = json.loads(req_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PipelineError(f"api_request.json unparseable: {e}") from None
    content = req.get("content") if isinstance(req, dict) else None
    if not isinstance(content, list):
        raise PipelineError("api_request.json content must be a list")
    texts = [i for i in content if isinstance(i, dict) and i.get("type") == "text"]
    images = [i for i in content if isinstance(i, dict) and i.get("type") == "image_url"]
    if len(texts) != 1:
        raise PipelineError(f"api_request.json must contain exactly 1 text item, got {len(texts)}")
    if not 1 <= len(images) <= 9:
        raise PipelineError(f"api_request.json image_url count {len(images)} not in 1..9")
    if len(content) != len(texts) + len(images):
        raise PipelineError("api_request.json content must contain only text and image_url items")
    leaked = _find_secret_key(req)
    if leaked is not None:
        raise PipelineError(f"api_request.json contains forbidden field: {leaked}")
    return frames, prompt


def _codex_prompt(cdir: Path, source: Path) -> str:
    work = cdir / "work"
    return f"""你在受限沙箱中分析一支清洁类参考视频，产出 Seedance 2.0 素材包。工作目录：{cdir}

已有产物（后端已生成）：
- {source.name}：原始参考视频（只读，禁止修改）
- work/contact_sheet.jpg：40 帧等距检查拼图
- work/manifest.json：视频元数据与 40 帧时间戳
- work/NN_inspect_*.png：40 张检查帧

先阅读技能文档（只读；除这两个文件外不读取其他目录的文件）：
- {SKILL_MD}
- {REF_COMPRESSION}
- {REF_PATTERNS}

按以下步骤执行：
1. 查看 work/contact_sheet.jpg 与 work/manifest.json，必要时打开个别 inspect 帧确认细节；识别镜头边界，按技能规则把视频压缩为连贯的 15 秒结构（前态 → 工具介入 → 峰值动作 → 反应 → 擦除 → 最终结果）。
2. 选择不超过 9 个时间点（秒，升序），每个对应一个主导动作/状态；避开字幕、水印、模糊、转场混合帧。
3. 导出关键帧（keyframes/ 下同时生成 contact_sheet.jpg 与 manifest.json 属正常）：
   {sys.executable} {EXTRACT_SCRIPT} {source} --out-dir {work / "keyframes"} --times "t1,t2,..." --prefix keyframe --columns 3
4. 写 {work / "shot_timeline.md"}：源视频元数据、保留/删除的镜头、每张关键帧的角色、最终 15 秒时间分配。
5. 写 {work / "seedance_prompt.txt"}：中文，严格按技能 prompt 骨架（输出规格；图片1..N 角色；主体/环境连续性；合计 15.0 秒的时间线；物理因果；避免清单）。
6. 构建 dry-run 请求（禁止真实提交 Ark）：
   {sys.executable} {SEEDANCE_SCRIPT} create --dry-run --prompt-file {work / "seedance_prompt.txt"} --ref-images <按序关键帧绝对路径> --model doubao-seedance-2-0-260128 --ratio 9:16 --duration 15 --resolution 720p --generate-audio --no-watermark --payload-out {work / "api_request.json"}

硬性禁令：
- 只在 {cdir} 内创建/修改文件。
- 禁止联网（沙箱已断网，联网必然失败）。
- 禁止打印、读取或记录任何环境变量。
- 产物路径与文件名必须严格如上，不得改名。
"""


def _render_preview(work: Path, dest: Path) -> None:
    """选中的关键帧合成 15s 720x1280(9:16) 25fps 占位视频（等时长 slideshow）。"""
    frames = sorted((work / "keyframes").glob("*.png"))
    n = len(frames)
    _run_cmd(
        [
            "ffmpeg", "-y",
            "-framerate", f"{n}/15",
            "-pattern_type", "glob", "-i", "*.png",
            "-vf",
            "scale=720:1280:force_original_aspect_ratio=decrease,"
            "pad=720:1280:(ow-iw)/2:(oh-ih)/2:black,"
            "fps=25,tpad=stop_mode=clone:stop=2",
            "-t", "15", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(dest),
        ],
        timeout=120,
        cwd=work / "keyframes",
        step="preview",
    )


def run(settings: Settings, cid: str, runner) -> None:
    """后台任务入口；任何步骤失败 → status=failed + error，不抛异常。"""
    # data_dir 可能是相对路径（生产默认 "data"）：子进程带 cwd 时相对路径会错位，统一起点解析为绝对
    cdir = (settings.data_dir / cid).resolve()
    work = cdir / "work"
    try:
        if storage.update_meta(settings.data_dir, cid, status="processing", error=None) is None:
            return
        sources = sorted(cdir.glob("source.*"))
        if not sources:
            raise PipelineError("source video missing")
        source = sources[0]
        _run_cmd(
            [
                sys.executable, str(EXTRACT_SCRIPT), str(source),
                "--out-dir", str(work),
                "--sample-count", "40", "--prefix", "inspect",
            ],
            timeout=120,
            step="extract",
        )
        runner.run(cdir, _codex_prompt(cdir, source))
        keyframes, prompt = validate_work_dir(work)
        _render_preview(work, cdir / "preview.mp4")
        storage.update_meta(
            settings.data_dir, cid, status="done", keyframes=keyframes, prompt=prompt
        )
    except Exception as e:
        storage.update_meta(settings.data_dir, cid, status="failed", error=str(e)[:500])

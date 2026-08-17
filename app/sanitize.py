"""公共脱敏：seedance/seedream/postprocess 的报错输出先过 sanitize 再进日志/响应。

规则：剔除任何含 key/authorization 的行，就地抹除密钥字面值，最后截断。
"""

import os
import re

DETAIL_LIMIT = 300
LEAK_RE = re.compile(r"key|authorization", re.IGNORECASE)


def sanitize(text: str, limit: int = DETAIL_LIMIT) -> str:
    out = "\n".join(ln for ln in text.splitlines() if not LEAK_RE.search(ln)).strip()
    key = os.environ.get("ARK_API_KEY", "").strip()
    if key:
        out = out.replace(key, "***")
    return out[:limit]

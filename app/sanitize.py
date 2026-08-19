"""公共脱敏：provider 报错输出先过 sanitize 再进日志/响应。

规则：剔除任何含 key/authorization 的行，就地抹除密钥字面值，最后截断。
"""

import os
import re

DETAIL_LIMIT = 300
LEAK_RE = re.compile(r"key|authorization", re.IGNORECASE)
BEARER_RE = re.compile(r"\bBearer\s+[^\s,;]+", re.IGNORECASE)


def sanitize(text: str, limit: int = DETAIL_LIMIT, *, secrets=()) -> str:
    out = "\n".join(ln for ln in text.splitlines() if not LEAK_RE.search(ln)).strip()
    out = BEARER_RE.sub("Bearer ***", out)
    actual = list(secrets) + [
        os.environ.get("ARK_API_KEY", ""),
        os.environ.get("MINIMAX_API_KEY", ""),
        os.environ.get("AUTODL_ART_TOKEN", ""),
    ]
    for secret in actual:
        value = str(secret).strip()
        if value:
            out = out.replace(value, "***")
    return out[:limit]

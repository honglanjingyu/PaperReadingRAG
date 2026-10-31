# app/service/core/retrieval/cache_fingerprint.py
"""
检索缓存配置指纹（P1-4）

背景：缓存 key 原来只对「调用方传入的检索参数」做哈希，而影响检索结果的一批
环境开关不在参数里：

- ENABLE_TERM_EXPAND / BM25_TERM_EXPANSION_BOOST / TERM_MAP_PATH（方案 Q2 术语归一）；
- HYBRID_* 系列（单路 Top-1 保送、图片意图条件权重、RRF k、图片 boost 等）；
- RERANK_MODEL / RERANK_BASE_URL（换模型或换服务端，同 query 结果会变）。

后果：改了词表或 boost 之后，同样的 query 仍会命中旧缓存，评测结果与线上行为
都会骗人（大规模评测里「术语扩展命中 4 条 query 但指标没变」就有一半是这个原因）。

做法：把上面这批进程级检索配置压成一个短指纹，缓存 key 的哈希输入里带上它。
配置一改，指纹就变，旧缓存自然不命中。

本模块只依赖标准库，不 import 任何 retrieval / rag 内的模块，
避免 hybrid 与 rag.cached_search 之间形成循环导入。
"""

import hashlib
import json
import os
from typing import Dict, Optional
from urllib.parse import urlparse

# 默认词表就放在本模块同目录（与 rewriter.py 的默认值保持一致）
_DEFAULT_TERM_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "term_map.json")

# 参与指纹的环境变量前缀：新增同类开关时不用改代码，自动纳入
_FINGERPRINT_PREFIXES = (
    "HYBRID_",
    "ENABLE_TERM_EXPAND",
    "ENABLE_QUERY_REWRITE",
    "BM25_",
    "RERANK_",
    "TERM_MAP_PATH",
)

# 含这些片段的键一律跳过：既不该影响语义，也避免把密钥类配置卷进哈希
_SENSITIVE_MARKERS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "AUTH", "CREDENTIAL")

# 这些键只取 host（P1-4 原文：RERANK_BASE_URL 的 host），避免路径/尾斜杠差异
# 让同一台服务产生不同指纹。
_HOST_ONLY_SUFFIXES = ("_BASE_URL", "_URL", "_HOST")


def _is_relevant(key: str) -> bool:
    upper = key.upper()
    if any(m in upper for m in _SENSITIVE_MARKERS):
        return False
    return any(upper.startswith(p) for p in _FINGERPRINT_PREFIXES)


def _normalize_value(key: str, value: str) -> str:
    if any(key.upper().endswith(s) for s in _HOST_ONLY_SUFFIXES):
        try:
            netloc = urlparse(value).netloc
        except Exception:
            netloc = ""
        return netloc or value
    return value


def _term_map_digest(explicit_path: Optional[str] = None) -> str:
    """词表内容摘要。

    只把 TERM_MAP_PATH 写进指纹是不够的：常见做法是同一个路径换内容
    （改 term_map.json 加几条术语），路径没变但扩展词变了，必须一并失效。
    """
    path = explicit_path or os.getenv("TERM_MAP_PATH") or _DEFAULT_TERM_MAP
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:12]
    except OSError:
        # 词表不存在 / 不可读：记为 missing，与「存在但内容不同」区分开
        return "missing"


def retrieval_config_snapshot() -> Dict[str, str]:
    """当前进程里影响检索语义的配置快照（可直接打印进日志排查）。"""
    snapshot: Dict[str, str] = {}
    for key in sorted(os.environ):
        if _is_relevant(key):
            snapshot[key] = _normalize_value(key, os.environ.get(key, ""))
    # 词表内容摘要（含默认词表）：内容一变，指纹就变
    snapshot["__term_map_digest__"] = _term_map_digest()
    return snapshot


def retrieval_config_fingerprint() -> str:
    """检索配置指纹：配置不变则稳定，任一开关/词表内容变化即变。"""
    payload = json.dumps(retrieval_config_snapshot(), sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode()).hexdigest()[:12]

"""生成阶段的原图回填。

真一次 ANN 让图片块能被检索到，但如果生成阶段只把文本描述喂给 LLM，
图片的视觉信息仍然没有进入答案。这里负责把命中的图片块对应的原图
转成 OpenAI 多模态格式，塞进 user message。

约束（按 4GB 显存 / 计费友好设计）：
- 单次最多 4 张图
- 单张不超过 2 MB（原图）
- 总 base64 载荷设上限，避免撑爆上下文
- 任何一张图读失败都跳过，绝不影响正常回答
"""

import base64
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MAX_IMAGES = 4
DEFAULT_MAX_IMAGE_MB = 2.0
DEFAULT_MAX_TOTAL_MB = 6.0

_IMAGE_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _resolve_path(image_ref: str) -> Optional[str]:
    """把 image_ref 解析成真实存在的文件路径。

    image_ref 可能是绝对路径（入库时传的是 str(file_path)），
    也可能是相对仓库根目录 / uploads 的路径。逐个尝试。
    """
    if not image_ref:
        return None

    candidates: List[str] = []
    if os.path.isabs(image_ref):
        candidates.append(image_ref)
    else:
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )))
        )
        candidates.append(os.path.join(repo_root, image_ref))
        candidates.append(os.path.join(repo_root, "uploads", image_ref))
        candidates.append(image_ref)

    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _mime_of(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return _IMAGE_MIME.get(ext, "image/jpeg")


def _encode_image(path: str, max_bytes: int) -> Optional[Tuple[str, str]]:
    """读图并转 base64，返回 (mime, b64)。超限或失败返回 None。"""
    try:
        size = os.path.getsize(path)
        if size <= 0 or size > max_bytes:
            logger.info(
                f"跳过图片（大小 {size / 1024 / 1024:.2f} MB 超限 {max_bytes / 1024 / 1024:.1f} MB）: {path}"
            )
            return None
        with open(path, "rb") as f:
            raw = f.read()
        return _mime_of(path), base64.b64encode(raw).decode("ascii")
    except Exception as e:
        logger.warning(f"读取图片失败 {path}: {e}")
        return None


def collect_image_refs(results: List[Dict[str, Any]]) -> List[str]:
    """从检索结果里挑出图片块，按出现顺序去重。"""
    refs: List[str] = []
    seen = set()
    for r in results or []:
        ref = (r.get("image_ref") or "").strip()
        if not ref or ref in seen:
            continue
        # 只认图片块；老数据没有 chunk_type 时按有 image_ref 就收
        ctype = r.get("chunk_type") or ""
        if ctype and ctype != "image":
            continue
        seen.add(ref)
        refs.append(ref)
    return refs


def build_image_parts(
    results: List[Dict[str, Any]],
    max_images: int = None,
    max_image_mb: float = None,
    max_total_mb: float = None,
) -> List[Dict[str, Any]]:
    """把命中的图片块转成 OpenAI 多模态 image_url 片段列表。"""
    if os.getenv("MULTIMODAL_BACKFILL_ENABLED", "true").lower() != "true":
        return []

    max_images = max_images or _env_int("MULTIMODAL_MAX_IMAGES", DEFAULT_MAX_IMAGES)
    max_image_mb = max_image_mb or _env_float("MULTIMODAL_MAX_IMAGE_MB", DEFAULT_MAX_IMAGE_MB)
    max_total_mb = max_total_mb or _env_float("MULTIMODAL_MAX_TOTAL_MB", DEFAULT_MAX_TOTAL_MB)

    max_bytes = int(max_image_mb * 1024 * 1024)
    total_budget = int(max_total_mb * 1024 * 1024)

    refs = collect_image_refs(results)[:max_images]
    if not refs:
        return []

    parts: List[Dict[str, Any]] = []
    used = 0
    for ref in refs:
        path = _resolve_path(ref)
        if not path:
            logger.info(f"跳过图片（文件不存在）: {ref}")
            continue
        size = os.path.getsize(path)
        if used + size > total_budget:
            logger.info(f"跳过图片（总预算 {max_total_mb:.1f} MB 已用完）: {ref}")
            continue
        encoded = _encode_image(path, max_bytes)
        if not encoded:
            continue
        mime, b64 = encoded
        used += size
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })

    if parts:
        logger.info(f"原图回填: 命中 {len(refs)} 张，实际注入 {len(parts)} 张，共 {used / 1024:.0f} KB")
    return parts


def build_multimodal_user_content(
    text: str,
    results: List[Dict[str, Any]],
) -> Any:
    """构建 user message 的 content。

    没有可用图片时返回纯字符串，保持与原行为完全一致；
    有图片时返回 [{"type": "text"}, {"type": "image_url"}...]。
    """
    parts = build_image_parts(results)
    if not parts:
        return text
    return [{"type": "text", "text": text}] + parts


__all__ = [
    "collect_image_refs",
    "build_image_parts",
    "build_multimodal_user_content",
]

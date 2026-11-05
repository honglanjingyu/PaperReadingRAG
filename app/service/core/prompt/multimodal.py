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
import io
import logging
import os
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MAX_IMAGES = 4
DEFAULT_MAX_IMAGE_MB = 2.0
DEFAULT_MAX_TOTAL_MB = 6.0

# P0-4：图片注入的相似度门限。低于这个分数的图片块不注入，退回纯文本，
# 避免文本类问题命中一张无关插图也要多付一次图片载荷。
# 与 SIMILARITY_THRESHOLD 同级（.env 里是 0.3），可用 env MULTIMODAL_MIN_SCORE 覆盖。
DEFAULT_MIN_SCORE = 0.3

# P0-2：降采样与缓存默认值
def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v.strip() if v and v.strip() else default

DEFAULT_MAX_EDGE = 1280
DEFAULT_JPEG_QUALITY = 85
DEFAULT_CACHE_SIZE = 32
# 原图超过这个大小就不读盘（降采样后载荷很小，这里只防超大文件拖慢进程）
HARD_RAW_LIMIT_MB = 32.0

# 进程内 LRU：key=(abs_path, mtime, size, max_edge, quality) -> (mime, b64, payload_bytes)
_CACHE_MAX = 32
_cache: "OrderedDict[Tuple, Tuple[str, str, int]]" = OrderedDict()
_cache_lock = threading.Lock()
_cache_hit = 0
_cache_miss = 0


def _cache_get(key: Tuple) -> Optional[Tuple[str, str, int]]:
    global _cache_hit
    with _cache_lock:
        val = _cache.get(key)
        if val is not None:
            _cache.move_to_end(key)
            _cache_hit += 1
        return val


def _cache_put(key: Tuple, val: Tuple[str, str, int]) -> None:
    global _cache_miss
    with _cache_lock:
        _cache[key] = val
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
        _cache_miss += 1


def cache_stats() -> Dict[str, int]:
    with _cache_lock:
        return {"hit": _cache_hit, "miss": _cache_miss, "size": len(_cache), "capacity": _CACHE_MAX}


def clear_cache() -> None:
    global _cache_hit, _cache_miss
    with _cache_lock:
        _cache.clear()
        _cache_hit = 0
        _cache_miss = 0

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


def _shrink_to_jpeg(raw: bytes, max_edge: int, quality: int) -> Optional[bytes]:
    """把原始图片字节降采样成长边不超过 max_edge 的 JPEG 字节。

    - 透明通道（PNG/WebP）铺白底后再转 JPEG，否则透明区域会变黑
    - EXIF 方向先转正，避免手机拍的照片躺倒
    - 已经比 max_edge 小、且本来就是 JPEG 的，仍会重新压一遍（quality 统一）
    """
    try:
        from PIL import Image, ImageOps
    except ImportError:
        logger.warning("未安装 Pillow，跳过降采样，回退原始字节")
        return None

    try:
        with Image.open(io.BytesIO(raw)) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                img = img.convert("RGBA")
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")

            w, h = img.size
            long_edge = max(w, h)
            if long_edge > max_edge > 0:
                scale = max_edge / float(long_edge)
                new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
                img = img.resize(new_size, Image.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            return buf.getvalue()
    except Exception as e:
        logger.warning(f"降采样失败，回退原始字节: {e}")
        return None


def _encode_image(
    path: str,
    max_edge: int = None,
    quality: int = None,
) -> Optional[Tuple[str, str, int]]:
    """读图 → 降采样 → base64，返回 (mime, b64, payload_bytes)。失败返回 None。

    payload_bytes 是 base64 解码后的实际字节数，用于总预算统计。
    """
    max_edge = max_edge if max_edge is not None else _env_int("MULTIMODAL_MAX_EDGE", DEFAULT_MAX_EDGE)
    quality = quality if quality is not None else _env_int("MULTIMODAL_JPEG_QUALITY", DEFAULT_JPEG_QUALITY)

    try:
        st = os.stat(path)
        if st.st_size <= 0:
            logger.info(f"跳过图片（空文件）: {path}")
            return None
        if st.st_size > HARD_RAW_LIMIT_MB * 1024 * 1024:
            logger.info(
                f"跳过图片（原始文件 {st.st_size / 1024 / 1024:.1f} MB 超过硬上限 {HARD_RAW_LIMIT_MB:.0f} MB）: {path}"
            )
            return None

        key = (os.path.abspath(path), st.st_mtime_ns, st.st_size, max_edge, quality)
        cached = _cache_get(key)
        if cached is not None:
            return cached

        with open(path, "rb") as f:
            raw = f.read()

        shrunk = _shrink_to_jpeg(raw, max_edge, quality)
        # 小尺寸 PNG（线条图/示意图）本身压缩率极高，转 JPEG 反而会膨胀
        # （实测 3.7KB → 15.2KB）。只有确实更小才采用降采样结果，否则保留原图。
        if shrunk and len(shrunk) < len(raw):
            data, mime = shrunk, "image/jpeg"
        elif shrunk:
            logger.info(
                f"降采样反而变大（{len(raw) / 1024:.1f} KB → {len(shrunk) / 1024:.1f} KB），保留原图: {path}"
            )
            data, mime = raw, _mime_of(path)
        else:
            data, mime = raw, _mime_of(path)

        b64 = base64.b64encode(data).decode("ascii")
        val = (mime, b64, len(data))
        _cache_put(key, val)
        return val
    except Exception as e:
        logger.warning(f"读取图片失败 {path}: {e}")
        return None


def get_multimodal_min_score() -> float:
    """P0-4 图片注入相似度门限：env MULTIMODAL_MIN_SCORE，默认 0.3。

    设为 0 或负数即关闭门限（回到逢图必回填的旧行为，便于对照评测）。
    """
    return _env_float("MULTIMODAL_MIN_SCORE", DEFAULT_MIN_SCORE)


def collect_image_refs(
    results: List[Dict[str, Any]],
    min_score: Optional[float] = None,
) -> List[str]:
    """从检索结果里挑出图片块，按 score 降序、按 image_ref 去重。

    P0-4 改动：
    - 按 score 降序排序后再返回，调用方截断 top-N 时拿到的是最相关的几张，
      而不是「检索顺序里前 N 张」（后者可能被低分插图占满）。
    - min_score 门限：score 低于门限的图片块不返回。文本类问题命中一张
      无关插图时分数普遍偏低，这里直接退回纯文本，省掉一次图片载荷。
      门限来自 env MULTIMODAL_MIN_SCORE（默认 0.3），传参可覆盖。
    注意：没有 score 字段的老结果按 0.0 处理 —— 门限 > 0 时会被过滤掉，
    这是有意的（无分数无从判断相关性）；需要保留时把门限设为 0。
    """
    if min_score is None:
        min_score = get_multimodal_min_score()

    candidates: List[Tuple[float, int, str]] = []
    seen = set()
    for idx, r in enumerate(results or []):
        ref = (r.get("image_ref") or "").strip()
        if not ref or ref in seen:
            continue
        # 只认图片块；老数据没有 chunk_type 时按有 image_ref 就收
        ctype = r.get("chunk_type") or ""
        if ctype and ctype != "image":
            continue
        seen.add(ref)
        try:
            score = float(r.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        candidates.append((score, idx, ref))

    if min_score and min_score > 0:
        kept = [c for c in candidates if c[0] >= min_score]
        dropped = len(candidates) - len(kept)
        if dropped > 0:
            logger.info(
                f"P0-4 图片门限: {dropped}/{len(candidates)} 张 score < {min_score} 未注入，退回纯文本"
            )
        candidates = kept

    # score 降序；同分时保持检索原序（idx 升序），结果稳定可复现
    candidates.sort(key=lambda c: (-c[0], c[1]))
    return [ref for _, _, ref in candidates]


def build_image_parts_detailed(
    results: List[Dict[str, Any]],
    max_images: int = None,
    max_image_mb: float = None,
    max_total_mb: float = None,
    min_score: float = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """把命中的图片块转成 image_url 片段，并返回真正注入成功的 image_ref 列表。

    返回 (injected_refs, parts)。injected_refs 与 parts 一一对应，用于 P0-1
    只对成功注入的图片块去掉 caption（未注入的必须保留 caption，否则丢信息）。
    """
    if os.getenv("MULTIMODAL_BACKFILL_ENABLED", "true").lower() != "true":
        return [], []

    max_images = max_images or _env_int("MULTIMODAL_MAX_IMAGES", DEFAULT_MAX_IMAGES)
    max_image_mb = max_image_mb or _env_float("MULTIMODAL_MAX_IMAGE_MB", DEFAULT_MAX_IMAGE_MB)
    max_total_mb = max_total_mb or _env_float("MULTIMODAL_MAX_TOTAL_MB", DEFAULT_MAX_TOTAL_MB)

    max_bytes = int(max_image_mb * 1024 * 1024)
    total_budget = int(max_total_mb * 1024 * 1024)

    # P0-4：collect_image_refs 已按 score 降序 + 门限过滤，这里截 top-N 就是
    # 「最相关的 N 张」，而不是检索顺序里的前 N 张。
    refs = collect_image_refs(results, min_score=min_score)[:max_images]
    if not refs:
        return [], []

    max_edge = _env_int("MULTIMODAL_MAX_EDGE", DEFAULT_MAX_EDGE)
    quality = _env_int("MULTIMODAL_JPEG_QUALITY", DEFAULT_JPEG_QUALITY)

    parts: List[Dict[str, Any]] = []
    injected_refs: List[str] = []
    used = 0
    raw_total = 0
    for ref in refs:
        path = _resolve_path(ref)
        if not path:
            logger.info(f"跳过图片（文件不存在）: {ref}")
            continue
        encoded = _encode_image(path, max_edge, quality)
        if not encoded:
            continue
        mime, b64, payload = encoded
        # 单张降采样后仍超过 max_bytes（极端长图）就跳过
        if payload > max_bytes:
            logger.info(
                f"跳过图片（降采样后 {payload / 1024 / 1024:.2f} MB 仍超单张上限 {max_image_mb:.1f} MB）: {ref}"
            )
            continue
        if used + payload > total_budget:
            logger.info(f"跳过图片（总预算 {max_total_mb:.1f} MB 已用完）: {ref}")
            continue
        used += payload
        try:
            raw_total += os.path.getsize(path)
        except OSError:
            pass
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })
        injected_refs.append(ref)

    if parts:
        logger.info(
            f"原图回填: 命中 {len(refs)} 张，实际注入 {len(parts)} 张；"
            f"原图 {raw_total / 1024:.0f} KB → 降采样后 {used / 1024:.0f} KB"
            f"（长边≤{max_edge}，JPEG q={quality}）"
        )
    return injected_refs, parts


def build_image_parts(
    results: List[Dict[str, Any]],
    max_images: int = None,
    max_image_mb: float = None,
    max_total_mb: float = None,
    min_score: float = None,
) -> List[Dict[str, Any]]:
    """把命中的图片块转成 OpenAI 多模态 image_url 片段列表。"""
    _, parts = build_image_parts_detailed(
        results, max_images, max_image_mb, max_total_mb, min_score
    )
    return parts


def build_multimodal_user_content(
    text: str,
    results: List[Dict[str, Any]],
    parts: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    """构建 user message 的 content。

    没有可用图片时返回纯字符串，保持与原行为完全一致；
    有图片时返回 [{"type": "text"}, {"type": "image_url"}...]。

    parts：调用方已经算好的 image_url 片段（比如 P0-1 需要先知道哪几张
    注入成功才能决定去不去 caption）。传了就直接用，避免重复读盘和编码。
    """
    if parts is None:
        parts = build_image_parts(results)
    if not parts:
        return text
    return [{"type": "text", "text": text}] + parts


__all__ = [
    "collect_image_refs",
    "get_multimodal_min_score",
    "build_image_parts",
    "build_multimodal_user_content",
    "cache_stats",
    "clear_cache",
]

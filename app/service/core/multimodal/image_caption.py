"""图片语义描述（给 BM25 用的多模态文本）。

背景：真一次 ANN 让图片块有了图像向量，但 BM25 只认 content / content_with_weight
里的字面词。图片块原来的文字只有「[图片] 文件名 + OCR 文本」，示意图、照片这类
OCR 出不了字的图在 BM25 里等于空文档，永远搜不到。

这里用多模态大模型（OpenAI 兼容接口，走项目已有的 LLM_API_KEY / LLM_BASE_URL）
给图片生成一段中文描述，拼进图片块的 content，BM25 就能对描述里的名词、术语打分。

设计约束：
- 任何失败都只返回空串，绝不阻断入库
- 单图限制字节数，超限先缩图再送
- 结果按「路径 + 修改时间」缓存，重复导入不重复花钱
- 独立开关 IMAGE_CAPTION_ENABLED，默认开启
"""

import base64
import hashlib
import io
import json
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# 送进模型的原图上限（字节），超过先缩放
MAX_IMAGE_BYTES = 1_500_000
# 请求超时（秒）
REQUEST_TIMEOUT = 60

_PROMPT = (
    "请用中文详细描述这张图片，供全文检索使用。要求："
    "1) 先说明图片类型（照片/流程图/柱状图/折线图/散点图/表格截图/公式/示意图等）；"
    "2) 如实转述图中出现的所有文字、标题、坐标轴标签、图例、数字和单位；"
    "3) 描述图形结构、趋势、对比关系、关键结论；"
    "4) 如果图里包含文字说明，一并纳入；"
    "5) 只输出描述正文，不要加「好的」「以下是」这类前缀，不要重复图片文件名。"
)

# 报告 3.1 第 2 条：结构化 caption。在自由描述之外，再要求一句「形状/颜色/布局/数量」的
# 结构化标注，提升 BM25 对纯视觉 query（如「四根蓝色柱子」「三个方框用箭头连接」）的首命中率。
# 单独成段、固定前缀「结构化标注:」，便于回填脚本与评测按字段抽取，也不干扰自然语言描述。
_STRUCTURED_SUFFIX = (
    "\n\n另外，在描述最后另起一行，用固定前缀「结构化标注:」输出一句结构化摘要，"
    "按「布局方向、元素类型与数量、主色、形状」的顺序写，例如："
    "「结构化标注: 横向布局，3 个方框用箭头依次连接，主色蓝色，矩形。」"
    "数量与颜色不确定时写「约」或省略该项，不要编造。"
)

_cache_lock = threading.Lock()
_cache: dict = {}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def is_enabled() -> bool:
    """描述功能总开关。"""
    return _env_bool("IMAGE_CAPTION_ENABLED", True)


def _cache_key(image_path: str) -> str:
    try:
        mtime = os.path.getmtime(image_path)
        size = os.path.getsize(image_path)
    except OSError:
        mtime, size = 0, 0
    # 报告 3.1 第 2 条：把结构化开关状态并进 key，关掉开关后不会命中旧的
    # 「带结构化标注」描述（开关默认开，未设置时与改动前行为一致）。
    structured = "s1" if structured_enabled() else "s0"
    raw = f"{os.path.abspath(image_path)}|{mtime}|{size}|{structured}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _load_image_bytes(image_path: str) -> Optional[bytes]:
    """读原图；过大时用 Pillow 等比缩小，返回 JPEG/PNG 字节。"""
    try:
        with open(image_path, "rb") as f:
            data = f.read()
    except OSError as e:
        logger.warning(f"读取图片失败 {image_path}: {e}")
        return None

    if len(data) <= MAX_IMAGE_BYTES:
        return data

    try:
        from PIL import Image
    except ImportError:
        logger.warning("未安装 Pillow，超限图片无法缩放，直接跳过描述")
        return None

    try:
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        scale = 1.0
        while True:
            buf = io.BytesIO()
            w = max(1, int(img.width * scale))
            h = max(1, int(img.height * scale))
            resized = img.resize((w, h)) if scale < 1.0 else img
            resized.save(buf, format="JPEG", quality=85)
            out = buf.getvalue()
            if len(out) <= MAX_IMAGE_BYTES or scale < 0.2:
                return out
            scale *= 0.75
    except Exception as e:
        logger.warning(f"图片缩放失败 {image_path}: {e}")
        return None


def _mime_of(data: bytes, image_path: str) -> str:
    if data[:8].startswith(b"\x89PNG"):
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    ext = os.path.splitext(image_path)[1].lower()
    return {".png": "image/png", ".gif": "image/gif", ".bmp": "image/bmp",
            ".webp": "image/webp"}.get(ext, "image/jpeg")


def _cache_get(key: str) -> Optional[str]:
    with _cache_lock:
        return _cache.get(key)


def _cache_put(key: str, value: str):
    with _cache_lock:
        if len(_cache) > 4096:
            _cache.clear()
        _cache[key] = value


def structured_enabled() -> bool:
    """报告 3.1 第 2 条：是否要求模型额外出结构化标注。默认开启，可关掉做 A/B。"""
    return _env_bool("IMAGE_CAPTION_STRUCTURED", True)


def _build_messages(data: bytes, mime: str) -> list:
    b64 = base64.b64encode(data).decode("ascii")
    prompt = _PROMPT + _STRUCTURED_SUFFIX if structured_enabled() else _PROMPT
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }
    ]


def _call_model(data: bytes, mime: str) -> Optional[str]:
    """调用 OpenAI 兼容的多模态接口，返回描述文本；失败返回 None。"""
    api_key = os.getenv("LLM_API_KEY") or os.getenv("MODEL_API_KEY")
    base_url = (os.getenv("LLM_BASE_URL")
                or os.getenv("DASHSCOPE_BASE_URL"))
    model = os.getenv("IMAGE_CAPTION_MODEL") or os.getenv("LLM_MODEL", "qwen-turbo")
    if not api_key:
        logger.warning("未配置 LLM_API_KEY / MODEL_API_KEY，跳过图片描述")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("未安装 openai 包，跳过图片描述")
        return None

    kwargs = {"api_key": api_key, "timeout": REQUEST_TIMEOUT}
    if base_url:
        kwargs["base_url"] = base_url

    try:
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=model,
            messages=_build_messages(data, mime),
            temperature=0.2,
            max_tokens=512,
        )
        if not resp or not resp.choices:
            logger.warning(f"图片描述接口返回为空: model={model}")
            return None
        content = resp.choices[0].message.content
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("text"):
                    parts.append(p["text"])
                elif isinstance(p, str):
                    parts.append(p)
            content = "".join(parts)
        text = (content or "").strip()
        return text or None
    except Exception as e:
        logger.warning(f"调用多模态模型生成图片描述失败: {e}")
        return None


def describe_image(image_path: str) -> str:
    """为一张图片生成中文语义描述，供 BM25 索引。

    任何失败（未开启、读图失败、模型报错、返回空）都返回空串，
    调用方拼 content 时遇到空串就保持原样。
    """
    if not is_enabled():
        return ""
    if not image_path or not os.path.exists(image_path):
        return ""

    key = _cache_key(image_path)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    data = _load_image_bytes(image_path)
    if not data:
        return ""

    mime = _mime_of(data, image_path)
    text = _call_model(data, mime)
    text = (text or "").strip()
    if text:
        logger.info(f"图片描述生成成功: {os.path.basename(image_path)} ({len(text)} 字)")
    else:
        logger.info(f"图片描述为空，跳过: {os.path.basename(image_path)}")
    _cache_put(key, text)
    return text


def build_image_content(image_path: str, file_name: str, ocr_text: str = "") -> str:
    """拼图片块入库用的 content：原有 [图片] 前缀 + OCR 文本 + 模型描述。

    BM25 只对这些字面文字打分，把描述拼进来等于让图片块「可被文字检索」。
    """
    content, _ = build_image_payload(image_path, file_name, ocr_text)
    return content


def build_image_payload(image_path: str, file_name: str, ocr_text: str = "") -> tuple:
    """同时返回 (content, caption)。

    content 用于入库正文（BM25 的 content/content_with_weight 字段），
    caption 单独留一份，写进 ES 的 caption 字段单独加权，
    回填脚本也靠它判断哪些图片块还没描述过。
    没有描述时 caption 为空串，content 与改造前完全一致。
    """
    base = f"[图片] {file_name}"
    ocr = (ocr_text or "").strip()
    if ocr:
        base = f"{base}\n{ocr}"
    desc = describe_image(image_path)
    if desc:
        return f"{base}\n图片描述: {desc}", desc
    return base, ""


__all__ = [
    "is_enabled",
    "describe_image",
    "build_image_content",
    "build_image_payload",
]

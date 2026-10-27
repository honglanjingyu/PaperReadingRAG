"""中文 CLIP 多模态 Embedding 模型（真一次 ANN 的统一向量空间）

文本塔与图像塔输出同一空间、同一维度（768）的向量，因此文本 chunk 与图片 chunk
可以写入 Milvus 同一个集合的同一个 ``vector`` 字段，用一次 ANN 检索同时命中两者。

模型：OFA-Sys/chinese-clip-vit-large-patch14-336px（407M 参数，768 维，图像 336px）
"""

import os
import logging
from typing import List, Optional, Union

from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "OFA-Sys/chinese-clip-vit-large-patch14-336px"
DEFAULT_DIM = 768


def _as_tensor(out):
    """兼容 transformers 4.x（返回张量）与 5.x（返回 BaseModelOutputWithPooling）。"""
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0, :]
    return out


class ChineseClipEmbeddingModel(BaseEmbeddingModel):
    """中文 CLIP 编码器：文本与图片共用一套向量空间。"""

    _shared = {}

    def __init__(self, model_name: str = None, device: str = None, batch_size: int = 8):
        self._model_name = model_name or os.getenv("MM_EMBEDDING_MODEL", DEFAULT_MODEL)
        self._device_pref = device or os.getenv("MM_EMBEDDING_DEVICE", "auto")
        self._batch_size = int(os.getenv("MM_EMBEDDING_BATCH_SIZE", batch_size))
        self._dimension = int(os.getenv("MM_EMBEDDING_DIM", DEFAULT_DIM))
        self._model = None
        self._processor = None
        self._device = None
        self._load_failed = False

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    # ---------- 惰性加载 ----------

    def _ensure_loaded(self):
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:
            import torch
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor

            if self._device_pref == "cpu":
                device = "cpu"
            elif self._device_pref == "cuda":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # dtype：4GB 显存下 fp16 能把 large 模型的常驻显存从 1.53GB 压到 0.77GB。
            # auto = cuda 上用 fp16，cpu 上保持 fp32（CPU 跑 fp16 会明显变慢）。
            dtype_pref = os.getenv("MM_EMBEDDING_DTYPE", "auto").strip().lower()
            if dtype_pref == "float16" or (dtype_pref == "auto" and device == "cuda"):
                torch_dtype = torch.float16
            elif dtype_pref == "bfloat16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

            logger.info(
                f"加载多模态 embedding 模型 {self._model_name} "
                f"(device={device}, dtype={str(torch_dtype).replace('torch.', '')}) ..."
            )
            model = ChineseCLIPModel.from_pretrained(self._model_name, torch_dtype=torch_dtype)
            proc = ChineseCLIPProcessor.from_pretrained(self._model_name)
            model = model.to(device).eval()

            self._model = model
            self._processor = proc
            self._device = device
            try:
                self._dimension = int(model.config.projection_dim)
            except Exception:
                pass
            logger.info(
                f"多模态 embedding 就绪: {self._model_name} device={device} dim={self._dimension}"
            )
            return True
        except Exception as e:
            self._load_failed = True
            logger.error(f"多模态 embedding 加载失败: {e}", exc_info=True)
            return False

    @property
    def is_available(self) -> bool:
        return self._ensure_loaded()

    # ---------- BaseEmbeddingModel 接口 ----------

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        if not text:
            return None
        vecs = self.generate_embeddings([text])
        return vecs[0] if vecs else None

    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        if not texts:
            return []
        if not self._ensure_loaded():
            return [None] * len(texts)

        import torch

        results: List[Optional[List[float]]] = []
        bs = max(1, self._batch_size)
        for start in range(0, len(texts), bs):
            batch = [t if t else " " for t in texts[start:start + bs]]
            try:
                with torch.no_grad():
                    inputs = self._processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}
                    feats = _as_tensor(self._model.get_text_features(**inputs))
                    feats = torch.nn.functional.normalize(feats.float(), dim=-1)
                    results.extend(feats.cpu().numpy().astype("float32").tolist())
            except Exception as e:
                logger.error(f"文本编码失败: {e}")
                results.extend([None] * len(batch))
        return results

    def encode_image(self, image_path: str) -> Optional[List[float]]:
        vecs = self.encode_images([image_path])
        return vecs[0] if vecs else None

    def encode_images(self, image_paths: List[str]) -> List[Optional[List[float]]]:
        if not image_paths:
            return []
        if not self._ensure_loaded():
            return [None] * len(image_paths)

        import torch
        from PIL import Image

        results: List[Optional[List[float]]] = []
        bs = max(1, min(self._batch_size, 4))
        for start in range(0, len(image_paths), bs):
            batch_paths = image_paths[start:start + bs]
            images = []
            valid = []
            for p in batch_paths:
                try:
                    with Image.open(p) as im:
                        images.append(im.convert("RGB").copy())
                    valid.append(True)
                except Exception as e:
                    logger.warning(f"图片读取失败 {p}: {e}")
                    images.append(None)
                    valid.append(False)

            real_imgs = [im for im in images if im is not None]
            if not real_imgs:
                results.extend([None] * len(batch_paths))
                continue

            try:
                with torch.no_grad():
                    inputs = self._processor(images=real_imgs, return_tensors="pt")
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}
                    feats = _as_tensor(self._model.get_image_features(**inputs))
                    feats = torch.nn.functional.normalize(feats.float(), dim=-1)
                    feats_list = feats.cpu().numpy().astype("float32").tolist()
                it = iter(feats_list)
                for ok in valid:
                    results.append(next(it) if ok else None)
            except Exception as e:
                logger.error(f"图像编码失败: {e}")
                results.extend([None] * len(batch_paths))
        return results

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

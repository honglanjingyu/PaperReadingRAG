"""VISTA 多模态 Embedding（1024 维，真一次 ANN）

文本塔 bge-m3 + 视觉塔 EVA02-CLIP-L-14，文本与图片输出同一空间、同一 1024 维向量，
因此可直接写同一个 Milvus 集合的同一个 vector 字段，query 一次编码一次 ANN。

模型：BAAI/bge-visualized 的 Visualized_m3.pth（VISTA，arXiv:2406.04292，ACL 2024）

【为什么是子进程】
visual_bge/modeling.py 依赖 transformers.file_utils，该模块在 transformers 5.x 已移除；
主 rag 环境是 5.20.2，无法直接 import。所以模型跑在独立 conda 环境 vista
（transformers 4.44.2 + torch 2.6.0+cu124）的常驻子进程里，本类通过 JSONL 协议调用。
这样主环境的 transformers 一行都不用动，Chinese-CLIP 那条路仍可随时切回。

相关环境变量：
  MM_EMBEDDING_MODEL       固定为 vista（由 service.py 按 provider 选择）
  MM_EMBEDDING_DIM         VISTA 的输出维度，默认 1024
  VISTA_PYTHON             子进程解释器，默认 /root/miniconda3/envs/vista/bin/python
  VISTA_SERVICE_SCRIPT     服务脚本，默认 <repo>/work/vista_service.py
  VISTA_PKG_DIR            visual_bge 源码目录，默认 /root/visual_bge_pkg
  VISTA_WEIGHT             权重文件，默认 /root/models/vista/Visualized_m3.pth
  VISTA_BGE_NAME           文本塔，默认 BAAI/bge-m3
  VISTA_START_TIMEOUT      模型加载超时秒数，默认 300
"""

import os
import sys
import json
import uuid
import logging
import threading
import subprocess
from typing import List, Optional

from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "vista-bge-visualized-m3"
DEFAULT_DIM = 1024


def _repo_root() -> str:
    # app/service/core/embedding/vista.py -> 上溯 4 层
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


class _VistaProcess:
    """常驻子进程封装（进程内单例，带锁）。"""

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._dim = None
        self._broken = False

    @classmethod
    def get(cls) -> "_VistaProcess":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ---------- 进程管理 ----------

    def _python(self) -> str:
        return os.getenv("VISTA_PYTHON", "/root/miniconda3/envs/vista/bin/python")

    def _script(self) -> str:
        # 服务端脚本随代码一起提交（放在包内），不依赖被 .gitignore 忽略的 work/ 目录
        default_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vista_service.py")
        return os.getenv("VISTA_SERVICE_SCRIPT", default_script)

    def _env(self) -> dict:
        env = os.environ.copy()
        env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _spawn(self):
        py = self._python()
        script = self._script()
        if not os.path.exists(py):
            raise RuntimeError(
                "找不到 VISTA 子进程解释器 %s；请先按 work/mm_vista_env.sh / "
                "work/mm_vista_gpu_install.sh / work/mm_vista_tf4.sh 建好 vista 环境" % py
            )
        if not os.path.exists(script):
            raise RuntimeError("找不到 VISTA 服务脚本 %s" % script)

        logger.info("启动 VISTA 编码子进程: %s %s", py, script)
        proc = subprocess.Popen(
            [py, "-u", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=self._env(),
        )

        timeout = float(os.getenv("VISTA_START_TIMEOUT", "300"))
        ready_line = self._readline_with_timeout(proc, timeout)
        if ready_line is None:
            proc.kill()
            raise RuntimeError("VISTA 子进程 %.0fs 内未就绪" % timeout)

        ready_line = ready_line.strip()
        if not ready_line.startswith("READY"):
            proc.kill()
            raise RuntimeError("VISTA 子进程未返回 READY：%r" % ready_line[:200])

        parts = ready_line.split()
        self._dim = int(parts[1]) if len(parts) > 1 else DEFAULT_DIM
        self._proc = proc
        logger.info("VISTA 编码子进程就绪: dim=%s pid=%s", self._dim, proc.pid)

    @staticmethod
    def _readline_with_timeout(proc: subprocess.Popen, timeout: float) -> Optional[str]:
        """带超时读一行（子进程加载模型要 30s 以上，不能直接阻塞 readline）。"""
        result = {}

        def _reader():
            try:
                result["line"] = proc.stdout.readline()
            except Exception as e:  # pragma: no cover
                result["err"] = e

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            return None
        return result.get("line")

    def _ensure(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        if self._broken:
            raise RuntimeError("VISTA 子进程已标记不可用，需重启进程")
        self._spawn()

    def restart(self):
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.kill()
                except Exception:
                    pass
                self._proc = None
            self._broken = False
            self._spawn()

    @property
    def dim(self) -> int:
        return self._dim or DEFAULT_DIM

    # ---------- 请求 ----------

    def encode(self, texts: List[str], images: List[str]):
        """返回 (text_vecs, image_vecs)，失败条目为 None。"""
        if not texts and not images:
            return [], []

        payload = {
            "id": uuid.uuid4().hex[:12],
            "texts": list(texts or []),
            "images": list(images or []),
        }
        line = json.dumps(payload, ensure_ascii=False)

        with self._lock:
            for attempt in (1, 2):
                try:
                    self._ensure()
                    self._proc.stdin.write(line + "\n")
                    self._proc.stdin.flush()
                    out = self._proc.stdout.readline()
                    if not out:
                        raise RuntimeError("VISTA 子进程无响应（可能已退出）")
                    resp = json.loads(out)
                    if resp.get("error"):
                        raise RuntimeError(resp["error"])
                    t = resp.get("texts") or [None] * len(payload["texts"])
                    i = resp.get("images") or [None] * len(payload["images"])
                    if resp.get("dim"):
                        self._dim = int(resp["dim"])
                    return t, i
                except Exception as e:
                    logger.warning("VISTA 编码失败（第 %d 次）：%s", attempt, e)
                    if attempt == 1:
                        try:
                            self._proc.kill()
                        except Exception:
                            pass
                        self._proc = None
                        continue
                    self._broken = True
                    raise
        return [], []

    def shutdown(self):
        with self._lock:
            if self._proc is None:
                return
            try:
                self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            finally:
                self._proc = None


class VistaEmbeddingModel(BaseEmbeddingModel):
    """VISTA 编码器：文本与图片共用一套 1024 维向量空间。"""

    def __init__(self, model_name: str = None, batch_size: int = 8):
        # 注意：不要读 MM_EMBEDDING_MODEL，那是 Chinese-CLIP 的配置项。
        # VISTA 的模型标识由 VISTA_MODEL_NAME 单独指定。
        self._model_name = model_name or os.getenv("VISTA_MODEL_NAME", DEFAULT_MODEL)
        self._dimension = int(os.getenv("MM_EMBEDDING_DIM", DEFAULT_DIM))
        self._batch_size = int(os.getenv("MM_EMBEDDING_BATCH_SIZE", batch_size))
        self._svc = _VistaProcess.get()

    # ---------- BaseEmbeddingModel ----------

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        if not text:
            return None
        vecs = self.generate_embeddings([text])
        return vecs[0] if vecs else None

    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        if not texts:
            return []
        results: List[Optional[List[float]]] = []
        bs = max(1, self._batch_size)
        for start in range(0, len(texts), bs):
            batch = [t if t else " " for t in texts[start:start + bs]]
            try:
                vecs, _ = self._svc.encode(texts=batch, images=[])
                if len(vecs) != len(batch):
                    vecs = (vecs + [None] * len(batch))[:len(batch)]
                results.extend(vecs)
            except Exception as e:
                logger.error("VISTA 文本编码失败: %s", e)
                results.extend([None] * len(batch))
        return results

    def encode_image(self, image_path: str) -> Optional[List[float]]:
        vecs = self.encode_images([image_path])
        return vecs[0] if vecs else None

    def encode_images(self, image_paths: List[str]) -> List[Optional[List[float]]]:
        if not image_paths:
            return []
        # 4GB 显存，图片单张编码，批量只做很小的分组
        results: List[Optional[List[float]]] = []
        bs = max(1, min(self._batch_size, 4))
        for start in range(0, len(image_paths), bs):
            batch = image_paths[start:start + bs]
            try:
                _, vecs = self._svc.encode(texts=[], images=batch)
                if len(vecs) != len(batch):
                    vecs = (vecs + [None] * len(batch))[:len(batch)]
                results.extend(vecs)
            except Exception as e:
                logger.error("VISTA 图片编码失败: %s", e)
                results.extend([None] * len(batch))
        return results

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    # ---------- 生命周期 ----------

    def shutdown(self):
        self._svc.shutdown()

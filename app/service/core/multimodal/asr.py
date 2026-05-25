"""ASR 服务 - 语音转文字"""

import os
import logging
import tempfile
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)

# ASR 引擎类型
ASR_TYPE = os.getenv("ASR_TYPE", "whisper")  # whisper, funasr, cloud


class ASRService:
    """ASR 服务 - 语音转文字"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._model = None
        self._asr_type = ASR_TYPE
        self._init_model()

    def _init_model(self):
        """初始化 ASR 模型"""
        if self._asr_type == "whisper":
            self._init_whisper()
        elif self._asr_type == "funasr":
            self._init_funasr()
        elif self._asr_type == "cloud":
            self._init_cloud()
        else:
            logger.warning(f"未知的 ASR 类型: {self._asr_type}，使用 Whisper")
            self._init_whisper()

    def _init_whisper(self):
        try:
            import whisper

            model_size = os.getenv("WHISPER_MODEL", "base")
            device = os.getenv("WHISPER_DEVICE", "cpu")  # 从环境变量读取

            self._model = whisper.load_model(model_size, device=device)
            logger.info(f"Whisper 模型加载成功: {model_size} (device={device})")
        except Exception as e:
            logger.error(f"Whisper 加载失败: {e}")
            self._model = None

    def _init_funasr(self):
        """初始化 FunASR"""
        try:
            from funasr import AutoModel

            model_dir = os.getenv("FUNASR_MODEL_DIR", "iic/SenseVoiceSmall")
            self._model = AutoModel(model=model_dir, disable_update=True)
            logger.info(f"FunASR 模型加载成功: {model_dir}")
        except ImportError:
            logger.error("FunASR 未安装")
            self._model = None

    def _init_cloud(self):
        """初始化云端 ASR"""
        self._cloud_provider = os.getenv("CLOUD_ASR_PROVIDER", "aliyun")
        self._cloud_api_key = os.getenv("CLOUD_ASR_API_KEY")
        self._cloud_secret = os.getenv("CLOUD_ASR_SECRET")
        logger.info(f"云端 ASR 初始化: provider={self._cloud_provider}")

    def transcribe(self, audio_path: str, language: str = "zh") -> Tuple[str, float, Dict]:
        """
        语音转文字

        Args:
            audio_path: 音频文件路径
            language: 语言代码 (zh, en, ja 等)

        Returns:
            (转录文字, 置信度, 详情)
        """
        if self._asr_type == "whisper":
            return self._transcribe_whisper(audio_path, language)
        elif self._asr_type == "funasr":
            return self._transcribe_funasr(audio_path)
        elif self._asr_type == "cloud":
            return self._transcribe_cloud(audio_path)
        else:
            return "", 0.0, {}

    def _transcribe_whisper(self, audio_path: str, language: str) -> Tuple[str, float, Dict]:
        """使用 Whisper 转录"""
        if self._model is None:
            return "", 0.0, {}

        try:
            result = self._model.transcribe(
                audio_path,
                language=language if language != "zh" else "zh",
                task="transcribe",
                verbose=False
            )

            text = result.get("text", "").strip()
            segments = result.get("segments", [])

            # 计算平均置信度
            confidence = 0.0
            if segments:
                avg_confidence = sum(s.get("confidence", 0) for s in segments) / len(segments)
                confidence = avg_confidence

            logger.info(f"Whisper 转录完成: 文本长度={len(text)}, 置信度={confidence:.2f}")
            return text, confidence, {"segments": segments, "language": result.get("language")}

        except Exception as e:
            logger.error(f"Whisper 转录失败: {e}")
            return "", 0.0, {}

    def _transcribe_funasr(self, audio_path: str) -> Tuple[str, float, Dict]:
        """使用 FunASR 转录"""
        if self._model is None:
            return "", 0.0, {}

        try:
            result = self._model.generate(input=audio_path, cache={})

            if result and len(result) > 0:
                text = result[0].get("text", "").strip()
                confidence = result[0].get("confidence", 0.8)
                return text, confidence, {"raw_result": result}

            return "", 0.0, {}

        except Exception as e:
            logger.error(f"FunASR 转录失败: {e}")
            return "", 0.0, {}

    def _transcribe_cloud(self, audio_path: str) -> Tuple[str, float, Dict]:
        """使用云端 ASR API"""
        # 实现云端 API 调用
        return "", 0.0, {}

    def transcribe_bytes(self, audio_bytes: bytes, format: str = "wav") -> Tuple[str, float, Dict]:
        """从字节数据转录"""
        import tempfile

        suffix = f".{format}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return self.transcribe(tmp_path)
        finally:
            os.unlink(tmp_path)


# 全局单例
_asr_service = None


def get_asr_service() -> ASRService:
    global _asr_service
    if _asr_service is None:
        _asr_service = ASRService()
    return _asr_service
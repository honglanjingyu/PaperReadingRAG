# app/service/core/multimodal/ocr.py (修复版)

"""OCR 服务 - 支持多种 OCR 引擎"""

import os
import logging
import base64
from typing import Optional, List, Tuple
from pathlib import Path

# ========== 设置环境变量 ==========
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

logger = logging.getLogger(__name__)

# OCR 引擎类型
OCR_TYPE = os.getenv("OCR_TYPE", "paddle")  # paddle, tesseract, easyocr, cloud


class OCRService:
    """OCR 服务 - 从图片中提取文字"""

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
        self._engine = None
        self._ocr_type = OCR_TYPE
        self._init_engine()

    def _init_engine(self):
        """初始化 OCR 引擎"""
        if self._ocr_type == "paddle":
            self._init_paddle()
        elif self._ocr_type == "tesseract":
            self._init_tesseract()
        elif self._ocr_type == "easyocr":
            self._init_easyocr()
        elif self._ocr_type == "cloud":
            self._init_cloud()
        else:
            logger.warning(f"未知的 OCR 类型: {self._ocr_type}，使用 PaddleOCR")
            self._init_paddle()

    def _init_paddle(self):
        """初始化 PaddleOCR - 修复参数问题"""
        try:
            from paddleocr import PaddleOCR

            # 抑制日志
            import logging
            for logger_name in ['ppocr', 'ppocr.data', 'ppocr.modeling', 'ppocr.postprocess']:
                logging.getLogger(logger_name).setLevel(logging.ERROR)

            # 使用最简参数
            self._engine = PaddleOCR(
                use_angle_cls=True,
                lang='ch'
            )
            logger.info("PaddleOCR 初始化成功")
        except ImportError:
            logger.error("PaddleOCR 未安装，请运行: pip install paddleocr")
            self._engine = None
        except Exception as e:
            logger.error(f"PaddleOCR 初始化失败: {e}")
            self._engine = None

    def _init_easyocr(self):
        """初始化 EasyOCR"""
        try:
            import easyocr
            self._engine = easyocr.Reader(['ch_sim', 'en'], gpu=False)
            logger.info("EasyOCR 初始化成功")
        except ImportError:
            logger.error("EasyOCR 未安装，请运行: pip install easyocr")
            self._engine = None

    def _init_tesseract(self):
        """初始化 Tesseract"""
        try:
            import pytesseract
            from PIL import Image
            self._engine = "tesseract"
            logger.info("Tesseract 初始化成功")
        except ImportError:
            logger.error("pytesseract 未安装")
            self._engine = None

    def _init_cloud(self):
        """初始化云端 OCR（阿里云/腾讯云）"""
        self._cloud_provider = os.getenv("CLOUD_OCR_PROVIDER", "aliyun")
        self._cloud_api_key = os.getenv("CLOUD_OCR_API_KEY")
        self._cloud_secret = os.getenv("CLOUD_OCR_SECRET")
        self._engine = "cloud"
        logger.info(f"云端 OCR 初始化: provider={self._cloud_provider}")

    def extract_text(self, image_path: str) -> Tuple[str, float, List[Dict]]:
        """
        从图片提取文字

        Args:
            image_path: 图片路径或 base64 字符串

        Returns:
            (提取的文字, 置信度, 识别详情)
        """
        if self._ocr_type == "paddle":
            return self._extract_with_paddle(image_path)
        elif self._ocr_type == "easyocr":
            return self._extract_with_easyocr(image_path)
        elif self._ocr_type == "tesseract":
            return self._extract_with_tesseract(image_path)
        elif self._ocr_type == "cloud":
            return self._extract_with_cloud(image_path)
        else:
            return "", 0.0, []

    def _extract_with_paddle(self, image_path: str) -> Tuple[str, float, List[Dict]]:
        """使用 PaddleOCR 提取文字"""
        if self._engine is None:
            return "", 0.0, []

        try:
            result = self._engine.ocr(image_path, cls=True)

            if not result or not result[0]:
                return "", 0.0, []

            texts = []
            confidences = []
            details = []

            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    confidence = line[1][1]
                    bbox = line[0]

                    texts.append(text)
                    confidences.append(confidence)
                    details.append({
                        "text": text,
                        "confidence": confidence,
                        "bbox": bbox
                    })

            full_text = "\n".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            logger.info(f"OCR 提取完成: {len(texts)} 行文字, 置信度={avg_confidence:.2f}")
            return full_text, avg_confidence, details

        except Exception as e:
            logger.error(f"PaddleOCR 提取失败: {e}")
            return "", 0.0, []

    def _extract_with_easyocr(self, image_path: str) -> Tuple[str, float, List[Dict]]:
        """使用 EasyOCR 提取文字"""
        if self._engine is None:
            return "", 0.0, []

        try:
            result = self._engine.readtext(image_path)

            if not result:
                return "", 0.0, []

            texts = []
            confidences = []
            details = []

            for bbox, text, confidence in result:
                texts.append(text)
                confidences.append(confidence)
                details.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": bbox
                })

            full_text = "\n".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return full_text, avg_confidence, details

        except Exception as e:
            logger.error(f"EasyOCR 提取失败: {e}")
            return "", 0.0, []

    def _extract_with_tesseract(self, image_path: str) -> Tuple[str, float, List[Dict]]:
        """使用 Tesseract 提取文字"""
        try:
            from PIL import Image
            import pytesseract

            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang='chi_sim+eng')

            # Tesseract 不直接提供置信度，使用近似值
            confidence = 0.8 if len(text) > 10 else 0.5

            return text.strip(), confidence, [{"text": text[:500], "confidence": confidence}]

        except Exception as e:
            logger.error(f"Tesseract 提取失败: {e}")
            return "", 0.0, []

    def _extract_with_cloud(self, image_path: str) -> Tuple[str, float, List[Dict]]:
        """使用云端 OCR API 提取文字"""
        if self._cloud_provider == "aliyun":
            return self._extract_with_aliyun(image_path)
        elif self._cloud_provider == "tencent":
            return self._extract_with_tencent(image_path)
        else:
            return "", 0.0, []

    def _extract_with_aliyun(self, image_path: str) -> Tuple[str, float, List[Dict]]:
        """阿里云 OCR"""
        try:
            import requests

            # 读取图片并转为 base64
            with open(image_path, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode()

            url = "https://ocr.cn-hangzhou.aliyuncs.com/pop/RecognizeGeneral"

            headers = {
                "Authorization": f"APPCODE {self._cloud_api_key}",
                "Content-Type": "application/json"
            }

            data = {"image": img_base64}

            response = requests.post(url, json=data, headers=headers, timeout=30)

            if response.status_code == 200:
                result = response.json()
                text = result.get("data", {}).get("content", "")
                confidence = result.get("data", {}).get("confidence", 0.8)
                return text, confidence, []
            else:
                logger.error(f"阿里云 OCR 失败: {response.status_code}")
                return "", 0.0, []

        except Exception as e:
            logger.error(f"阿里云 OCR 异常: {e}")
            return "", 0.0, []

    def _extract_with_tencent(self, image_path: str) -> Tuple[str, float, List[Dict]]:
        """腾讯云 OCR"""
        # 类似实现
        return "", 0.0, []

    def extract_from_bytes(self, image_bytes: bytes, filename: str = "image.jpg") -> Tuple[str, float, List[Dict]]:
        """从字节数据提取文字"""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            return self.extract_text(tmp_path)
        finally:
            os.unlink(tmp_path)


# 全局单例
_ocr_service = None


def get_ocr_service() -> OCRService:
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service
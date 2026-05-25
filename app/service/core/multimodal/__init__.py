# app/service/core/multimodal/__init__.py (修改版)
"""
多模态解析器 - 将图片/音频/视频转换为文本
图片解析使用 MinerU API
"""

import os
import logging
import tempfile
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

# ========== 在模块加载时设置环境变量 ==========
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['PADDLE_DEBUG'] = '0'

logger = logging.getLogger(__name__)


class MediaType(str, Enum):
    DOCUMENT = "document"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class ExtractedContent:
    """提取的内容"""
    media_type: MediaType
    original_filename: str
    text_content: str = ""
    metadata: Dict = field(default_factory=dict)
    success: bool = True
    error: str = None

    # 图片特有
    ocr_confidence: float = 0.0

    # 音频/视频特有
    duration_seconds: float = 0.0
    transcript_confidence: float = 0.0


class MinerUImageParser:
    """MinerU 图片解析器 - 使用 MinerU API 解析图片"""

    def __init__(self, api_token: str = None):
        """
        初始化 MinerU 图片解析器

        Args:
            api_token: MinerU API Token
        """
        self.api_token = api_token or os.getenv("PARSE_API_TOKEN")
        self.base_url = "https://mineru.net"

        if not self.api_token:
            logger.warning("PARSE_API_TOKEN 未配置，MinerU 图片解析不可用")

    def is_available(self) -> bool:
        """检查 MinerU 是否可用"""
        return bool(self.api_token)

    def parse_image(self, image_path: str, image_name: str) -> ExtractedContent:
        """
        使用 MinerU API 解析图片

        Args:
            image_path: 图片文件路径
            image_name: 图片文件名

        Returns:
            ExtractedContent 对象
        """
        content = ExtractedContent(
            media_type=MediaType.IMAGE,
            original_filename=image_name
        )

        if not self.is_available():
            content.success = False
            content.error = "MinerU API Token 未配置"
            return content

        try:
            import requests
            import time
            import zipfile
            from io import BytesIO

            # 1. 上传图片获取 batch_id
            upload_result = self._upload_image(image_path)
            if not upload_result:
                content.success = False
                content.error = "上传图片失败"
                return content

            batch_id = upload_result

            # 2. 等待解析完成
            zip_url = self._wait_for_result(batch_id, timeout=120)
            if not zip_url:
                content.success = False
                content.error = "图片解析超时或失败"
                return content

            # 3. 下载并提取 Markdown 内容
            markdown_content = self._download_markdown(zip_url)
            if markdown_content is None:
                content.success = False
                content.error = "下载解析结果失败"
                return content

            content.text_content = markdown_content
            content.ocr_confidence = 0.9  # MinerU 置信度较高
            content.metadata = {
                "parser": "mineru",
                "file_path": image_path,
                "file_size": os.path.getsize(image_path),
                "api_version": "v4"
            }

            logger.info(f"MinerU 图片解析完成: {image_name}, 文字长度={len(markdown_content)}")

        except Exception as e:
            logger.error(f"MinerU 图片解析失败: {image_name}, {e}")
            content.success = False
            content.error = str(e)

        return content

    def _upload_image(self, file_path: str) -> Optional[str]:
        """上传图片到 MinerU，返回 batch_id"""
        try:
            import requests

            url = f"{self.base_url}/api/v4/file-urls/batch"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_token}"
            }

            file_name = os.path.basename(file_path)
            data = {
                "files": [{"name": file_name}],
                "model_version": "vlm"
            }

            logger.info(f"申请上传URL: {file_name}")
            response = requests.post(url, headers=headers, json=data, timeout=30)

            if response.status_code != 200:
                logger.error(f"申请上传URL失败: {response.status_code}")
                return None

            result = response.json()
            if result.get("code") != 0:
                logger.error(f"申请上传URL失败: {result.get('msg')}")
                return None

            batch_id = result["data"]["batch_id"]
            file_url = result["data"]["file_urls"][0]

            logger.info(f"获取到上传URL, batch_id={batch_id}")

            with open(file_path, 'rb') as f:
                upload_response = requests.put(file_url, data=f, timeout=60)

            if upload_response.status_code not in (200, 201):
                logger.error(f"文件上传失败: {upload_response.status_code}")
                return None

            logger.info("文件上传成功")
            return batch_id

        except Exception as e:
            logger.error(f"上传图片失败: {e}")
            return None

    def _wait_for_result(self, batch_id: str, timeout: int = 120) -> Optional[str]:
        """等待解析完成并返回结果 ZIP URL"""
        try:
            import requests
            import time

            url = f"{self.base_url}/api/v4/extract-results/batch/{batch_id}"
            headers = {"Authorization": f"Bearer {self.api_token}"}

            start_time = time.time()
            interval = 3

            while time.time() - start_time < timeout:
                response = requests.get(url, headers=headers, timeout=30)

                if response.status_code != 200:
                    time.sleep(interval)
                    continue

                result = response.json()
                if result.get("code") != 0:
                    time.sleep(interval)
                    continue

                extract_results = result["data"].get("extract_result", [])
                if not extract_results:
                    time.sleep(interval)
                    continue

                file_result = extract_results[0]
                state = file_result.get("state")

                if state == "done":
                    zip_url = file_result.get("full_zip_url")
                    logger.info("解析完成")
                    return zip_url
                elif state == "failed":
                    err_msg = file_result.get("err_msg", "未知错误")
                    logger.error(f"解析失败: {err_msg}")
                    return None
                else:
                    elapsed = int(time.time() - start_time)
                    logger.info(f"解析中... 状态={state}, 已等待{elapsed}秒")
                    time.sleep(interval)

            logger.error(f"等待超时 ({timeout}秒)")
            return None

        except Exception as e:
            logger.error(f"查询结果异常: {e}")
            return None

    def _download_markdown(self, zip_url: str) -> Optional[str]:
        """下载 ZIP 包并提取 Markdown 内容"""
        try:
            import requests
            import zipfile
            from io import BytesIO

            response = requests.get(zip_url, timeout=60)
            if response.status_code != 200:
                logger.error(f"下载ZIP失败: {response.status_code}")
                return None

            zip_data = BytesIO(response.content)

            with zipfile.ZipFile(zip_data, 'r') as zip_ref:
                for file_name in zip_ref.namelist():
                    if file_name.endswith('full.md') or file_name.endswith('.md'):
                        with zip_ref.open(file_name) as md_file:
                            content = md_file.read().decode('utf-8')
                            logger.info(f"找到 Markdown 文件: {file_name}, 长度: {len(content)}")
                            return content

            logger.error("ZIP包中未找到 Markdown 文件")
            return None

        except Exception as e:
            logger.error(f"下载解析结果失败: {e}")
            return None


class MultimodalParser:
    """多模态解析器 - 将图片/音频/视频转换为文本"""

    # 图片格式列表
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    DOCUMENT_EXTENSIONS = {'.pdf', '.docx'}
    TEXT_EXTENSIONS = {'.txt', '.md', '.markdown'}

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

        # 初始化 MinerU 图片解析器
        self._mineru_parser = MinerUImageParser()

        # 初始化 ASR 和视频处理器（懒加载）
        self._asr_service = None
        self._video_processor = None

        self._asr_type = os.getenv("ASR_TYPE", "whisper")

        logger.info(f"MultimodalParser 初始化完成: 图片解析=MinerU API, ASR={self._asr_type}")

    def _get_asr_service(self):
        """懒加载 ASR 服务"""
        if self._asr_service is None:
            self._asr_service = self._init_asr()
        return self._asr_service

    def _init_asr(self):
        """初始化 ASR 服务"""
        if self._asr_type == "whisper":
            return self._init_whisper()
        elif self._asr_type == "funasr":
            return self._init_funasr()
        else:
            logger.warning(f"未知 ASR 类型: {self._asr_type}，使用 Whisper")
            return self._init_whisper()

    def _init_whisper(self):
        """初始化 Whisper"""
        try:
            import whisper
            model_size = os.getenv("WHISPER_MODEL", "base")
            model = whisper.load_model(model_size)
            logger.info(f"Whisper 加载成功: {model_size}")
            return model
        except ImportError:
            logger.error("Whisper 未安装，请运行: pip install openai-whisper")
            return None

    def _init_funasr(self):
        """初始化 FunASR"""
        try:
            from funasr import AutoModel
            model = AutoModel(model="iic/SenseVoiceSmall", disable_update=True)
            logger.info("FunASR 初始化成功")
            return model
        except ImportError:
            logger.error("FunASR 未安装")
            return None

    def _get_video_processor(self):
        """懒加载视频处理器"""
        if self._video_processor is None:
            self._video_processor = self._init_video_processor()
        return self._video_processor

    def _init_video_processor(self):
        """初始化视频处理器"""
        return VideoProcessor()

    def is_mineru_available(self) -> bool:
        """检查 MinerU 图片解析是否可用"""
        return self._mineru_parser.is_available()

    def parse_image(self, file_path: str, file_name: str) -> ExtractedContent:
        """
        解析图片 - 使用 MinerU API

        Args:
            file_path: 图片文件路径
            file_name: 图片文件名

        Returns:
            ExtractedContent 对象
        """
        return self._mineru_parser.parse_image(file_path, file_name)

    def parse_audio(self, file_path: str, file_name: str) -> ExtractedContent:
        """解析音频 -> 转文字"""
        content = ExtractedContent(
            media_type=MediaType.AUDIO,
            original_filename=file_name
        )

        asr = self._get_asr_service()
        if asr is None:
            content.success = False
            content.error = "ASR 服务不可用"
            return content

        try:
            if self._asr_type == "whisper":
                result = asr.transcribe(file_path, language="zh")
                content.text_content = result.get("text", "").strip()

                # 计算平均置信度
                segments = result.get("segments", [])
                if segments:
                    content.transcript_confidence = sum(s.get("confidence", 0) for s in segments) / len(segments)
                else:
                    content.transcript_confidence = 0.8 if content.text_content else 0

                content.duration_seconds = result.get("segments", [{}])[-1].get("end", 0) if segments else 0

            elif self._asr_type == "funasr":
                result = asr.generate(input=file_path, cache={})
                if result and len(result) > 0:
                    content.text_content = result[0].get("text", "").strip()
                    content.transcript_confidence = result[0].get("confidence", 0.8)

            # 获取音频时长
            try:
                import mutagen
                audio_info = mutagen.File(file_path)
                if audio_info and hasattr(audio_info.info, 'length'):
                    content.duration_seconds = audio_info.info.length
            except ImportError:
                pass

            content.metadata = {
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "asr_engine": self._asr_type,
                "duration": content.duration_seconds
            }

            logger.info(f"音频解析完成: {file_name}, 文字长度={len(content.text_content)}")

        except Exception as e:
            logger.error(f"音频解析失败: {file_name}, {e}")
            content.success = False
            content.error = str(e)

        return content

    def parse_video(self, file_path: str, file_name: str) -> ExtractedContent:
        """解析视频 -> 提取音频转文字 + 关键帧 OCR（使用 MinerU）"""
        content = ExtractedContent(
            media_type=MediaType.VIDEO,
            original_filename=file_name
        )

        processor = self._get_video_processor()
        if processor is None:
            content.success = False
            content.error = "视频处理器不可用"
            return content

        temp_files = []
        all_texts = []

        try:
            # 1. 提取音频并转文字
            audio_path = processor.extract_audio(file_path)
            if audio_path:
                temp_files.append(audio_path)
                audio_content = self.parse_audio(audio_path, f"{file_name}_audio")
                if audio_content.success and audio_content.text_content:
                    all_texts.append(f"[视频音频内容]\n{audio_content.text_content}")
                    content.transcript_confidence = audio_content.transcript_confidence

            # 2. 提取关键帧并使用 MinerU 进行 OCR
            frame_paths = processor.extract_keyframes(file_path, num_frames=5)
            temp_files.extend(frame_paths)

            frame_texts = []
            for i, frame_path in enumerate(frame_paths):
                frame_content = self.parse_image(frame_path, f"{file_name}_frame_{i}")
                if frame_content.success and frame_content.text_content:
                    frame_texts.append(f"[视频画面 {i + 1}]\n{frame_content.text_content}")

            if frame_texts:
                all_texts.append("\n\n".join(frame_texts))

            content.text_content = "\n\n".join(all_texts)
            content.duration_seconds = processor.get_duration(file_path)
            content.metadata = {
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "duration": content.duration_seconds,
                "keyframe_count": len(frame_paths),
                "has_audio": audio_path is not None
            }

            logger.info(f"视频解析完成: {file_name}, 文字长度={len(content.text_content)}")

        except Exception as e:
            logger.error(f"视频解析失败: {file_name}, {e}")
            content.success = False
            content.error = str(e)

        finally:
            # 清理临时文件
            processor.cleanup(temp_files)

        return content

    def parse_document(self, file_path: str, file_name: str) -> ExtractedContent:
        """解析文档（使用原有的 DocumentParser）"""
        from app.service.core.deepdoc import DocumentParser

        content = ExtractedContent(
            media_type=MediaType.DOCUMENT,
            original_filename=file_name
        )

        try:
            parser = DocumentParser()
            parsed = parser.parse(file_path, enable_cleaning=True)

            content.text_content = parsed.cleaned_text or ""
            content.metadata = {
                "total_pages": parsed.total_pages,
                "file_type": parsed.file_type,
                "file_size": os.path.getsize(file_path)
            }

            logger.info(f"文档解析完成: {file_name}, 文字长度={len(content.text_content)}")

        except Exception as e:
            logger.error(f"文档解析失败: {file_name}, {e}")
            content.success = False
            content.error = str(e)

        return content

    def parse_text(self, file_path: str, file_name: str) -> ExtractedContent:
        """解析文本文件"""
        content = ExtractedContent(
            media_type=MediaType.DOCUMENT,
            original_filename=file_name
        )

        try:
            # 尝试 UTF-8
            with open(file_path, 'r', encoding='utf-8') as f:
                content.text_content = f.read()
            content.metadata = {"encoding": "utf-8"}
        except UnicodeDecodeError:
            try:
                # 尝试 GBK
                with open(file_path, 'r', encoding='gbk') as f:
                    content.text_content = f.read()
                content.metadata = {"encoding": "gbk"}
            except Exception as e:
                content.success = False
                content.error = str(e)

        if content.success:
            logger.info(f"文本解析完成: {file_name}, 文字长度={len(content.text_content)}")

        return content

    def parse(self, file_path: str, file_name: str) -> ExtractedContent:
        """统一解析入口"""
        ext = os.path.splitext(file_name)[1].lower()

        if ext in self.IMAGE_EXTENSIONS:
            return self.parse_image(file_path, file_name)
        elif ext in self.AUDIO_EXTENSIONS:
            return self.parse_audio(file_path, file_name)
        elif ext in self.VIDEO_EXTENSIONS:
            return self.parse_video(file_path, file_name)
        elif ext in self.DOCUMENT_EXTENSIONS:
            return self.parse_document(file_path, file_name)
        elif ext in self.TEXT_EXTENSIONS:
            return self.parse_text(file_path, file_name)
        else:
            # 未知类型，当作文本处理
            logger.warning(f"未知文件类型: {ext}, 当作文本处理")
            return self.parse_text(file_path, file_name)


class VideoProcessor:
    """视频处理器 - 提取音频和关键帧"""

    def __init__(self):
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        import subprocess
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            self._ffmpeg_available = True
            logger.info("ffmpeg 可用")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self._ffmpeg_available = False
            logger.warning("ffmpeg 不可用，视频处理将受限")

    def extract_audio(self, video_path: str) -> Optional[str]:
        """从视频提取音频"""
        if not self._ffmpeg_available:
            return None

        import subprocess

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name

        try:
            cmd = [
                "ffmpeg", "-i", video_path, "-vn",
                "-acodec", "pcm_s16le", "-ar", "16000",
                "-ac", "1", "-y", audio_path
            ]
            subprocess.run(cmd, capture_output=True, timeout=120, check=True)
            return audio_path
        except Exception as e:
            logger.error(f"音频提取失败: {e}")
            return None

    def extract_keyframes(self, video_path: str, num_frames: int = 5) -> List[str]:
        """提取关键帧"""
        if not self._ffmpeg_available:
            return []

        import subprocess

        duration = self.get_duration(video_path)
        if duration <= 0:
            return []

        frame_paths = []
        time_points = [duration * (i + 0.5) / num_frames for i in range(num_frames)]

        for i, time_point in enumerate(time_points):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                frame_path = tmp.name

            try:
                cmd = [
                    "ffmpeg", "-i", video_path, "-ss", str(time_point),
                    "-vframes", "1", "-q:v", "2", "-y", frame_path
                ]
                subprocess.run(cmd, capture_output=True, timeout=30, check=True)

                if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                    frame_paths.append(frame_path)
                else:
                    os.unlink(frame_path)
            except Exception as e:
                logger.error(f"关键帧提取失败: {e}")

        return frame_paths

    def get_duration(self, video_path: str) -> float:
        """获取视频时长"""
        if not self._ffmpeg_available:
            return 0.0

        import subprocess
        import json

        try:
            cmd = [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "json", video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data.get("format", {}).get("duration", 0))
        except Exception:
            pass
        return 0.0

    def cleanup(self, paths: List[str]):
        """清理临时文件"""
        for path in paths:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass


# 全局单例
_parser = None


def get_multimodal_parser() -> MultimodalParser:
    """获取多模态解析器实例"""
    global _parser
    if _parser is None:
        _parser = MultimodalParser()
    return _parser


def is_mineru_available() -> bool:
    """检查 MinerU 图片解析是否可用"""
    return get_multimodal_parser().is_mineru_available()


__all__ = ['MultimodalParser', 'get_multimodal_parser', 'ExtractedContent', 'MediaType', 'is_mineru_available']
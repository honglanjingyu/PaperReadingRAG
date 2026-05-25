# app/service/core/multimodal/parser.py
"""
多模态解析器 - 将图片/音频/视频转换为文本
- 图片处理：使用 MinerU API
- 音频处理：使用 ASR（Whisper/FunASR）
- 视频处理：提取关键帧复用图片处理 + 提取音频复用音频处理
"""

import os
import logging
import tempfile
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .models import ExtractedContent, MediaType, ExtractStatus
from .asr import get_asr_service

logger = logging.getLogger(__name__)


class MinerUImageParser:
    """MinerU 图片解析器 - 使用 MinerU API 解析图片"""

    def __init__(self, api_token: str = None):
        self.api_token = api_token or os.getenv("PARSE_API_TOKEN")
        self.base_url = "https://mineru.net"
        self.model_version = os.getenv("MINERU_MODEL_VERSION", "vlm")

        if not self.api_token:
            logger.warning("PARSE_API_TOKEN 未配置，MinerU 图片解析不可用")

    def is_available(self) -> bool:
        return bool(self.api_token)

    def parse_image(self, image_path: str, image_name: str) -> ExtractedContent:
        """使用 MinerU API 解析图片"""
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
            batch_id = self._upload_image(image_path)
            if not batch_id:
                content.success = False
                content.error = "上传图片失败"
                return content

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

            # 清理 Markdown 格式，提取纯文本
            text_content = self._clean_markdown(markdown_content)

            content.text_content = text_content
            content.ocr_confidence = 0.9
            content.success = True
            content.metadata = {
                "parser": "mineru",
                "file_path": image_path,
                "file_size": os.path.getsize(image_path),
                "api_version": "v4"
            }

            logger.info(f"MinerU 图片解析完成: {image_name}, 文字长度={len(text_content)}")

        except Exception as e:
            logger.error(f"MinerU 图片解析失败: {image_name}, {e}")
            content.success = False
            content.error = str(e)

        return content

    def _upload_image(self, file_path: str) -> Optional[str]:
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
                "model_version": self.model_version
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

    def _clean_markdown(self, markdown_content: str) -> str:
        """清理 Markdown 格式，提取纯文本"""
        if not markdown_content:
            return ""

        import re

        # 移除图片链接
        text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', markdown_content)

        # 移除链接，保留链接文字
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # 移除粗体和斜体标记
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)

        # 移除行内代码和代码块
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'```[\s\S]*?```', '', text)

        # 移除标题标记
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

        # 移除水平线
        text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)

        # 移除引用标记
        text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)

        # 移除列表标记
        text = re.sub(r'^[\*\-\+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)

        # 标准化空白字符
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        return text.strip()


class MultimodalParser:
    """多模态解析器 - 将图片/音频/视频转换为文本"""

    # 文件类型分组
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

        # 初始化 ASR 服务（懒加载）
        self._asr_service = None

        self._asr_type = os.getenv("ASR_TYPE", "whisper")

        logger.info(f"MultimodalParser 初始化完成: 图片解析=MinerU API, ASR={self._asr_type}")

    def _get_asr_service(self):
        """懒加载 ASR 服务"""
        if self._asr_service is None:
            self._asr_service = get_asr_service()
        return self._asr_service

    def _get_video_processor(self):
        """懒加载视频处理器"""
        return VideoProcessor()

    def is_mineru_available(self) -> bool:
        return self._mineru_parser.is_available()

    def parse_image(self, file_path: str, file_name: str) -> ExtractedContent:
        """解析图片 - 使用 MinerU API"""
        return self._mineru_parser.parse_image(file_path, file_name)

    def parse_audio(self, file_path: str, file_name: str) -> ExtractedContent:
        """解析音频 -> 转文字（使用 ASR）"""
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
            # 使用 ASR 服务转写
            text, confidence, details = asr.transcribe(file_path)

            content.text_content = text
            content.transcript_confidence = confidence
            content.success = bool(text)

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

            logger.info(f"音频解析完成: {file_name}, 文字长度={len(text)}")

        except Exception as e:
            logger.error(f"音频解析失败: {file_name}, {e}")
            content.success = False
            content.error = str(e)

        return content

    def parse_video(self, file_path: str, file_name: str) -> ExtractedContent:
        """
        解析视频 -> 复用图片和音频处理流程
        1. 提取关键帧 -> 复用图片处理（MinerU API OCR）
        2. 提取音频 -> 复用音频处理（ASR转文字）
        """
        content = ExtractedContent(
            media_type=MediaType.VIDEO,
            original_filename=file_name,
            success=True
        )

        processor = self._get_video_processor()
        if not processor.is_available():
            content.success = False
            content.error = "视频处理器不可用（需要 ffmpeg）"
            return content

        temp_files = []
        all_texts = []
        frame_texts = []

        try:
            # ========== 1. 提取音频并复用音频处理 ==========
            audio_path = processor.extract_audio(file_path)
            if audio_path:
                temp_files.append(audio_path)
                # 复用音频处理
                audio_content = self.parse_audio(audio_path, f"{file_name}_audio")
                if audio_content.success and audio_content.text_content:
                    all_texts.append(f"[视频音频内容]\n{audio_content.text_content}")
                    content.transcript_confidence = audio_content.transcript_confidence
                    content.duration_seconds = audio_content.duration_seconds
                    logger.info(f"视频音频提取完成: {file_name}, 文字长度={len(audio_content.text_content)}")
                else:
                    logger.warning(f"视频音频提取失败: {file_name}, error={audio_content.error}")

            # ========== 2. 提取关键帧并复用图片处理 ==========
            frame_paths = processor.extract_keyframes(file_path, num_frames=5)
            temp_files.extend(frame_paths)

            if frame_paths:
                logger.info(f"视频关键帧提取完成: {file_name}, 共 {len(frame_paths)} 帧")

                for i, frame_path in enumerate(frame_paths):
                    # 复用图片处理（使用 MinerU API）
                    frame_content = self.parse_image(frame_path, f"{file_name}_frame_{i}")
                    if frame_content.success and frame_content.text_content:
                        frame_texts.append(f"[视频画面 {i + 1}]\n{frame_content.text_content}")
                        # 收集 OCR 置信度
                        if frame_content.ocr_confidence > 0:
                            content.ocr_confidence = max(content.ocr_confidence, frame_content.ocr_confidence)
                        logger.info(f"关键帧 {i + 1} OCR 完成: 文字长度={len(frame_content.text_content)}")
                    else:
                        logger.warning(f"关键帧 {i + 1} OCR 失败: {frame_content.error}")

                if frame_texts:
                    all_texts.append("\n\n".join(frame_texts))
            else:
                logger.warning(f"视频关键帧提取失败: {file_name}")

            # 合并所有提取的文字
            content.text_content = "\n\n".join(all_texts)

            # 获取视频时长（如果没有从音频获取到）
            if content.duration_seconds == 0:
                content.duration_seconds = processor.get_duration(file_path)

            # 判断处理是否成功
            if not content.text_content:
                content.success = False
                content.error = "未能从视频中提取任何文字内容（音频转录和关键帧OCR均失败）"
            else:
                content.success = True

            content.metadata = {
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "duration": content.duration_seconds,
                "keyframe_count": len(frame_paths),
                "has_audio": audio_path is not None,
                "processor": "video_multimodal",
                "keyframe_ocr_success": len(frame_texts) > 0,
                "audio_transcript_success": audio_path is not None and bool(content.transcript_confidence > 0)
            }

            logger.info(f"视频解析完成: {file_name}, 文字总长度={len(content.text_content)}, "
                        f"关键帧数={len(frame_paths)}, 音频转录={audio_path is not None}, "
                        f"成功={content.success}")

        except Exception as e:
            logger.error(f"视频解析失败: {file_name}, {e}", exc_info=True)
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
            media_type=MediaType.PDF if file_name.lower().endswith('.pdf') else MediaType.DOCX,
            original_filename=file_name
        )

        try:
            parser = DocumentParser()
            parsed = parser.parse(file_path, enable_cleaning=True)

            content.text_content = parsed.cleaned_text or ""
            content.success = bool(content.text_content)
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
            media_type=MediaType.TEXT,
            original_filename=file_name
        )

        try:
            # 尝试 UTF-8
            with open(file_path, 'r', encoding='utf-8') as f:
                content.text_content = f.read()
            content.metadata = {"encoding": "utf-8"}
            content.success = True
        except UnicodeDecodeError:
            try:
                # 尝试 GBK
                with open(file_path, 'r', encoding='gbk') as f:
                    content.text_content = f.read()
                content.metadata = {"encoding": "gbk"}
                content.success = True
            except Exception as e:
                content.success = False
                content.error = str(e)

        if content.success:
            logger.info(f"文本解析完成: {file_name}, 文字长度={len(content.text_content)}")

        return content

    def parse(self, file_path: str, file_name: str = None) -> ExtractedContent:
        """统一解析入口"""
        if file_name is None:
            file_name = os.path.basename(file_path)

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
            logger.warning(f"未知文件类型: {ext}, 当作文本处理")
            return self.parse_text(file_path, file_name)

    def parse_bytes(self, file_bytes: bytes, file_name: str) -> ExtractedContent:
        """从字节数据解析文件"""
        ext = os.path.splitext(file_name)[1].lower()

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            return self.parse(tmp_path, file_name)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# app/service/core/multimodal/parser.py
# 找到 VideoProcessor 类，修改 extract_keyframes 方法

class VideoProcessor:
    """视频处理器 - 提取音频和关键帧"""

    def __init__(self):
        self._ffmpeg_available = False
        self._check_ffmpeg()
        # 添加：从环境变量读取关键帧数量配置
        self.default_keyframe_count = int(os.getenv("VIDEO_KEYFRAME_COUNT", "5"))
        self.keyframe_quality = int(os.getenv("VIDEO_KEYFRAME_QUALITY", "2"))  # 2 = 高质量
        self.keyframe_extract_mode = os.getenv("VIDEO_KEYFRAME_MODE", "uniform")  # uniform, time
        self.keyframe_min_interval = float(os.getenv("VIDEO_KEYFRAME_MIN_INTERVAL", "2.0"))  # 最小间隔（秒）

    def _check_ffmpeg(self):
        """检查 ffmpeg 是否可用"""
        import subprocess

        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            self._ffmpeg_available = result.returncode == 0
            if self._ffmpeg_available:
                logger.info("ffmpeg 可用")
                logger.info(f"关键帧配置: 数量={self.default_keyframe_count}, 模式={self.keyframe_extract_mode}, 质量={self.keyframe_quality}")
            else:
                logger.warning("ffmpeg 不可用，视频处理将受限")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            self._ffmpeg_available = False
            logger.warning("ffmpeg 未安装或不可用，请安装 ffmpeg")

    def is_available(self) -> bool:
        return self._ffmpeg_available

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
            logger.info(f"音频提取成功: {audio_path}")
            return audio_path
        except Exception as e:
            logger.error(f"音频提取失败: {e}")
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            return None

    def extract_keyframes(
        self,
        video_path: str,
        num_frames: int = None,
        quality: int = None,
        mode: str = None
    ) -> List[str]:
        """
        提取关键帧

        Args:
            video_path: 视频文件路径
            num_frames: 提取的关键帧数量（None 则使用环境变量配置）
            quality: 图片质量 (1-31, 越小质量越高，默认 2)
            mode: 提取模式 (uniform: 均匀分布, time: 按时间间隔)

        Returns:
            关键帧图片路径列表
        """
        if not self._ffmpeg_available:
            return []

        import subprocess

        # 使用参数或环境变量配置
        num_frames = num_frames or self.default_keyframe_count
        quality = quality or self.keyframe_quality
        mode = mode or self.keyframe_extract_mode

        # 获取视频时长
        duration = self.get_duration(video_path)
        if duration <= 0:
            logger.warning(f"无法获取视频时长: {video_path}")
            return []

        # 根据模式计算时间点
        if mode == "time":
            # 按固定时间间隔提取
            interval = self.keyframe_min_interval
            time_points = []
            t = 0
            while t < duration and len(time_points) < num_frames:
                time_points.append(t + interval / 2)  # 取间隔中点
                t += interval
            if len(time_points) < num_frames:
                # 如果时间点不够，补充均匀分布
                remaining = num_frames - len(time_points)
                step = duration / (remaining + 1)
                for i in range(remaining):
                    time_points.append(step * (i + 1))
            time_points = sorted(time_points)[:num_frames]
        else:
            # 默认：均匀分布（取每个区间的中点）
            time_points = [duration * (i + 0.5) / num_frames for i in range(num_frames)]

        frame_paths = []

        for i, time_point in enumerate(time_points):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                frame_path = tmp.name

            try:
                cmd = [
                    "ffmpeg", "-i", video_path, "-ss", str(time_point),
                    "-vframes", "1", "-q:v", str(quality), "-y", frame_path
                ]
                subprocess.run(cmd, capture_output=True, timeout=30, check=True)

                if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                    frame_paths.append(frame_path)
                    logger.debug(f"关键帧提取成功: 时间点={time_point:.2f}s")
                else:
                    os.unlink(frame_path)
            except Exception as e:
                logger.error(f"关键帧提取失败: {e}")
                if os.path.exists(frame_path):
                    os.unlink(frame_path)

        logger.info(f"关键帧提取完成: {len(frame_paths)}/{num_frames} 帧 (模式={mode})")
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
        except Exception as e:
            logger.warning(f"获取视频时长失败: {e}")

        return 0.0

    def cleanup(self, paths: List[str]):
        """清理临时文件"""
        for path in paths:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.debug(f"清理临时文件: {path}")
                except Exception as e:
                    logger.warning(f"清理失败: {path}, {e}")


# 全局单例
_parser = None


def get_multimodal_parser() -> MultimodalParser:
    """获取多模态解析器实例"""
    global _parser
    if _parser is None:
        _parser = MultimodalParser()
    return _parser


def parse_multimodal_file(file_path: str, file_name: str = None) -> ExtractedContent:
    """便捷函数：解析多模态文件"""
    return get_multimodal_parser().parse(file_path, file_name)


def is_mineru_available() -> bool:
    """检查 MinerU 图片解析是否可用"""
    return get_multimodal_parser().is_mineru_available()


__all__ = [
    'MultimodalParser',
    'get_multimodal_parser',
    'parse_multimodal_file',
    'is_mineru_available',
    'ExtractedContent',
    'MediaType',
    'ExtractStatus'
]
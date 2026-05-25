"""视频处理器 - 提取关键帧和音频"""

import os
import logging
import tempfile
from typing import List, Tuple, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoProcessor:
    """视频处理器 - 提取关键帧和音频"""

    def __init__(self):
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """检查 ffmpeg 是否可用"""
        import subprocess

        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True
            )
            self._ffmpeg_available = result.returncode == 0
            if self._ffmpeg_available:
                logger.info("ffmpeg 可用")
            else:
                logger.warning("ffmpeg 不可用，视频处理功能受限")
        except FileNotFoundError:
            self._ffmpeg_available = False
            logger.warning("ffmpeg 未安装，请安装 ffmpeg")

    def extract_audio(self, video_path: str, output_format: str = "wav") -> Optional[str]:
        """
        从视频中提取音频

        Args:
            video_path: 视频文件路径
            output_format: 输出格式 (wav, mp3, m4a)

        Returns:
            音频文件路径
        """
        if not self._ffmpeg_available:
            return None

        import subprocess

        # 创建临时音频文件
        with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as tmp:
            audio_path = tmp.name

        try:
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-vn",  # 不处理视频
                "-acodec", "pcm_s16le" if output_format == "wav" else "libmp3lame",
                "-ar", "16000",  # 采样率 16kHz
                "-ac", "1",  # 单声道
                "-y",  # 覆盖输出文件
                audio_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                logger.info(f"音频提取成功: {audio_path}")
                return audio_path
            else:
                logger.error(f"音频提取失败: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"音频提取异常: {e}")
            return None

    def extract_keyframes(self, video_path: str, num_frames: int = 5) -> List[str]:
        """
        提取视频关键帧

        Args:
            video_path: 视频文件路径
            num_frames: 提取的关键帧数量

        Returns:
            关键帧图片路径列表
        """
        if not self._ffmpeg_available:
            return []

        import subprocess

        # 获取视频时长
        duration = self._get_video_duration(video_path)
        if duration <= 0:
            return []

        # 计算时间点（均匀分布）
        time_points = [duration * (i + 0.5) / num_frames for i in range(num_frames)]

        frame_paths = []

        for i, time_point in enumerate(time_points):
            with tempfile.NamedTemporaryFile(suffix=f".jpg", delete=False) as tmp:
                frame_path = tmp.name

            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-ss", str(time_point),
                "-vframes", "1",
                "-q:v", "2",
                "-y",
                frame_path
            ]

            try:
                subprocess.run(cmd, capture_output=True, timeout=30)

                if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                    frame_paths.append(frame_path)
                    logger.info(f"关键帧提取成功: {frame_path}")
                else:
                    os.unlink(frame_path)

            except Exception as e:
                logger.error(f"关键帧提取失败: {e}")
                if os.path.exists(frame_path):
                    os.unlink(frame_path)

        return frame_paths

    def _get_video_duration(self, video_path: str) -> float:
        """获取视频时长"""
        if not self._ffmpeg_available:
            return 0.0

        import subprocess
        import json

        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                video_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                data = json.loads(result.stdout)
                duration = float(data.get("format", {}).get("duration", 0))
                return duration

        except Exception as e:
            logger.error(f"获取视频时长失败: {e}")

        return 0.0

    def process_video(self, video_path: str, extract_frames: bool = True) -> Dict:
        """
        处理视频：提取音频和关键帧

        Args:
            video_path: 视频文件路径
            extract_frames: 是否提取关键帧

        Returns:
            处理结果字典
        """
        result = {
            "audio_path": None,
            "frame_paths": [],
            "duration": self._get_video_duration(video_path)
        }

        # 提取音频
        audio_path = self.extract_audio(video_path)
        if audio_path:
            result["audio_path"] = audio_path

        # 提取关键帧
        if extract_frames:
            result["frame_paths"] = self.extract_keyframes(video_path, num_frames=5)

        return result

    def cleanup_temp_files(self, paths: List[str]):
        """清理临时文件"""
        for path in paths:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.debug(f"清理临时文件: {path}")
                except Exception as e:
                    logger.warning(f"清理失败: {path}, {e}")


# 全局单例
_video_processor = None


def get_video_processor() -> VideoProcessor:
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor()
    return _video_processor
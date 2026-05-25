# scripts/generate_test_video.py
"""
生成测试视频文件脚本
将多个图片和音频文件融合成一个视频文件
用于测试 RAG 系统的视频处理能力

用法:
    python scripts/generate_test_video.py

要求:
    pip install opencv-python pillow moviepy numpy
"""

import os
import sys
from pathlib import Path

# 获取当前脚本所在目录和 tmp 目录
SCRIPT_DIR = Path(__file__)
PROJECT_ROOT = SCRIPT_DIR.parent
TMP_DIR = PROJECT_ROOT / "tmp"

# 默认文件路径
DEFAULT_IMAGES = [
    TMP_DIR / "架构图.jpeg",
    TMP_DIR / "流程图.png",
]
DEFAULT_AUDIO = TMP_DIR / "汇报.mp3"
DEFAULT_OUTPUT = TMP_DIR / "test_video.mp4"


def check_files_exist(image_paths: list, audio_path: Path) -> bool:
    """检查输入文件是否存在"""
    missing_files = []

    for img_path in image_paths:
        if not img_path.exists():
            missing_files.append(str(img_path))

    if not audio_path.exists():
        missing_files.append(str(audio_path))

    if missing_files:
        print("❌ 以下文件不存在:")
        for f in missing_files:
            print(f"   - {f}")
        return False

    print("✅ 所有输入文件存在")
    return True


def generate_video_with_moviepy(
        image_paths: list,
        audio_path: Path,
        output_path: Path,
        duration_per_image: float = 3.0,
        transition_duration: float = 0.5,
        resolution: tuple = (1280, 720),
        fps: int = 24,
        add_title: bool = True
):
    """
    使用 moviepy 生成视频（推荐）

    Args:
        image_paths: 图片路径列表
        audio_path: 音频文件路径
        output_path: 输出视频路径
        duration_per_image: 每张图片显示时长（秒）
        transition_duration: 转场过渡时长（秒）
        resolution: 视频分辨率 (width, height)
        fps: 帧率
        add_title: 是否添加标题
    """
    from moviepy.editor import (
        ImageClip, AudioFileClip,
        CompositeVideoClip, concatenate_videoclips,
        TextClip, ColorClip
    )
    from PIL import Image
    import numpy as np

    print(f"\n🎬 开始生成测试视频...")
    print(f"  图片数量: {len(image_paths)}")
    print(f"  音频文件: {audio_path.name}")
    print(f"  输出路径: {output_path}")
    print(f"  每张图片时长: {duration_per_image}秒")
    print(f"  分辨率: {resolution[0]}x{resolution[1]}")
    print("-" * 50)

    # 1. 加载音频
    print("📀 加载音频...")
    try:
        audio_clip = AudioFileClip(str(audio_path))
        audio_duration = audio_clip.duration
        print(f"  音频时长: {audio_duration:.2f}秒")
    except Exception as e:
        print(f"  ❌ 音频加载失败: {e}")
        return None

    # 2. 计算每张图片的实际显示时长
    total_image_duration = len(image_paths) * duration_per_image
    if total_image_duration < audio_duration:
        adjusted_duration = audio_duration / len(image_paths)
        print(f"  ⚠️ 音频时长超过图片总时长，调整每张图片时长为: {adjusted_duration:.2f}秒")
        duration_per_image = adjusted_duration

    # 3. 处理每张图片
    print("🖼️ 处理图片...")
    clips = []
    temp_files = []

    for i, img_path in enumerate(image_paths, 1):
        print(f"  [{i}/{len(image_paths)}] {img_path.name}")

        try:
            # 加载图片
            img = Image.open(img_path)
            original_size = img.size

            # 计算缩放比例（保持宽高比，居中显示）
            img_ratio = img.width / img.height
            target_ratio = resolution[0] / resolution[1]

            if img_ratio > target_ratio:
                # 图片更宽，按宽度缩放
                new_width = resolution[0]
                new_height = int(resolution[0] / img_ratio)
            else:
                # 图片更高，按高度缩放
                new_height = resolution[1]
                new_width = int(resolution[1] * img_ratio)

            # 创建背景图片
            background = Image.new('RGB', resolution, (0, 0, 0))

            # 缩放图片
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 计算居中位置
            x_offset = (resolution[0] - new_width) // 2
            y_offset = (resolution[1] - new_height) // 2

            # 粘贴到背景
            background.paste(img_resized, (x_offset, y_offset))

            # 保存临时文件
            temp_img_path = TMP_DIR / f"_temp_frame_{i}.jpg"
            background.save(temp_img_path)
            temp_files.append(temp_img_path)

            # 创建视频片段
            clip = ImageClip(str(temp_img_path), duration=duration_per_image)
            clip = clip.set_fps(fps)

            # 添加淡入淡出效果
            if transition_duration > 0 and i > 1:
                clip = clip.crossfadein(transition_duration)
            if transition_duration > 0 and i < len(image_paths):
                clip = clip.crossfadeout(transition_duration)

            clips.append(clip)

            print(f"    原始尺寸: {original_size[0]}x{original_size[1]}, 处理后: {new_width}x{new_height}")

        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            continue

    if not clips:
        print("❌ 没有有效的图片，视频生成失败")
        return None

    # 4. 合并所有图片片段
    print("🎞️ 合并视频片段...")
    try:
        video_clip = concatenate_videoclips(clips, method="compose")
    except Exception as e:
        print(f"  使用 compose 方法失败: {e}")
        video_clip = concatenate_videoclips(clips)

    # 5. 添加音频
    print("🔊 添加音频轨道...")
    video_clip = video_clip.set_audio(audio_clip)

    # 如果视频长度超过音频，截断视频
    if video_clip.duration > audio_duration:
        print(f"  视频时长({video_clip.duration:.2f}秒)超过音频，截断视频")
        video_clip = video_clip.subclip(0, audio_duration)

    # 6. 添加文字标题（可选）
    if add_title:
        print("📝 添加文字标题...")
        try:
            # 尝试使用系统字体
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",  # Windows 黑体
                "C:/Windows/Fonts/msyh.ttc",  # Windows 微软雅黑
                "/System/Library/Fonts/PingFang.ttc",  # macOS
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux
            ]

            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = font_path
                    break

            if font:
                title_clip = TextClip(
                    "RAG 系统测试视频",
                    fontsize=50,
                    color='white',
                    font=font,
                    stroke_color='black',
                    stroke_width=2
                )
                title_clip = title_clip.set_position(('center', 50)).set_duration(3)
                title_clip = title_clip.crossfadein(0.5).crossfadeout(0.5)
                video_clip = CompositeVideoClip([video_clip, title_clip])
            else:
                print("  未找到中文字体，跳过标题")
        except Exception as e:
            print(f"  文字标题添加失败: {e}")

    # 7. 输出视频
    print("💾 输出视频文件...")
    try:
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        video_clip.write_videofile(
            str(output_path),
            fps=fps,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=str(TMP_DIR / "_temp_audio.m4a"),
            remove_temp=True,
            verbose=False,
            logger=None
        )

        # 获取文件大小
        file_size = output_path.stat().st_size / (1024 * 1024)

        print("-" * 50)
        print(f"✅ 视频生成成功!")
        print(f"  文件路径: {output_path}")
        print(f"  文件大小: {file_size:.2f} MB")
        print(f"  视频时长: {video_clip.duration:.2f}秒")
        print(f"  分辨率: {resolution[0]}x{resolution[1]}")

        return str(output_path)

    except Exception as e:
        print(f"❌ 视频输出失败: {e}")
        return None

    finally:
        # 关闭所有剪辑释放内存
        try:
            video_clip.close()
            audio_clip.close()
            for clip in clips:
                clip.close()
        except:
            pass

        # 清理临时图片文件
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()


def generate_video_with_opencv(
        image_paths: list,
        audio_path: Path,
        output_path: Path,
        duration_per_image: float = 3.0,
        resolution: tuple = (1280, 720),
        fps: int = 24
):
    """
    使用 OpenCV 生成视频（备用方案，不依赖 moviepy）

    Args:
        image_paths: 图片路径列表
        audio_path: 音频文件路径
        output_path: 输出视频路径
        duration_per_image: 每张图片显示时长（秒）
        resolution: 视频分辨率
        fps: 帧率
    """
    import cv2
    import numpy as np
    from PIL import Image
    import subprocess

    print(f"\n🎬 使用 OpenCV 生成测试视频...")
    print(f"  图片数量: {len(image_paths)}")
    print(f"  音频文件: {audio_path.name}")
    print(f"  输出路径: {output_path}")
    print("-" * 50)

    # 1. 计算总帧数
    frames_per_image = int(duration_per_image * fps)
    total_frames = frames_per_image * len(image_paths)

    # 2. 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        resolution
    )

    # 3. 处理每张图片
    print("🖼️ 处理图片并写入视频帧...")

    for i, img_path in enumerate(image_paths, 1):
        print(f"  [{i}/{len(image_paths)}] {img_path.name}")

        try:
            # 加载图片
            img = Image.open(img_path)

            # 转换为 RGB（如果不是）
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 计算缩放比例（保持宽高比，居中显示）
            img_ratio = img.width / img.height
            target_ratio = resolution[0] / resolution[1]

            if img_ratio > target_ratio:
                new_width = resolution[0]
                new_height = int(resolution[0] / img_ratio)
            else:
                new_height = resolution[1]
                new_width = int(resolution[1] * img_ratio)

            # 缩放图片
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 创建黑色背景
            background = Image.new('RGB', resolution, (0, 0, 0))

            # 计算居中位置
            x_offset = (resolution[0] - new_width) // 2
            y_offset = (resolution[1] - new_height) // 2

            # 粘贴图片到背景
            background.paste(img_resized, (x_offset, y_offset))

            # 转换为 numpy 数组和 BGR 格式
            frame = np.array(background)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 写入帧
            for _ in range(frames_per_image):
                video_writer.write(frame)

        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            continue

    # 4. 释放视频写入器
    video_writer.release()
    print("✅ 视频帧写入完成（无音频）")

    # 5. 添加音频（使用 ffmpeg）
    if audio_path.exists():
        print("🔊 添加音频轨道...")
        temp_video = output_path.with_suffix('.temp.mp4')
        output_path.rename(temp_video)

        # 使用 ffmpeg 合并音视频
        cmd = [
            'ffmpeg', '-y',
            '-i', str(temp_video),
            '-i', str(audio_path),
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            str(output_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            temp_video.unlink()
            print("✅ 音频添加成功")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ 音频添加失败: {e}")
            temp_video.rename(output_path)
        except FileNotFoundError:
            print("⚠️ ffmpeg 未安装，视频没有音频轨道")
            temp_video.rename(output_path)

    # 6. 获取文件信息
    file_size = output_path.stat().st_size / (1024 * 1024)

    print("-" * 50)
    print(f"✅ 视频生成成功!")
    print(f"  文件路径: {output_path}")
    print(f"  文件大小: {file_size:.2f} MB")
    print(f"  总帧数: {total_frames}")
    print(f"  分辨率: {resolution[0]}x{resolution[1]}")

    return str(output_path)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="生成 RAG 测试视频")
    parser.add_argument("--images", nargs="+", help="图片文件路径列表")
    parser.add_argument("--audio", help="音频文件路径")
    parser.add_argument("--output", help="输出视频路径")
    parser.add_argument("--duration", type=float, default=3.0, help="每张图片显示时长（秒）")
    parser.add_argument("--transition", type=float, default=0.5, help="转场时长（秒）")
    parser.add_argument("--resolution", default="1280x720", help="视频分辨率 (宽x高)")
    parser.add_argument("--fps", type=int, default=24, help="帧率")
    parser.add_argument("--no-title", action="store_true", help="不添加标题")
    parser.add_argument("--use-opencv", action="store_true", help="使用 OpenCV 模式（不依赖 moviepy）")

    args = parser.parse_args()

    # 设置文件路径
    if args.images:
        image_paths = [Path(p) for p in args.images]
    else:
        image_paths = DEFAULT_IMAGES

    audio_path = Path(args.audio) if args.audio else DEFAULT_AUDIO
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT

    # 解析分辨率
    resolution = tuple(map(int, args.resolution.split('x')))

    # 检查依赖
    print("=" * 50)
    print("RAG 测试视频生成器")
    print("=" * 50)

    # 检查输入文件
    if not check_files_exist(image_paths, audio_path):
        print("\n提示: 请将以下文件放入 tmp 目录:")
        print("  - 架构图.jpeg")
        print("  - 流程图.png")
        print("  - 汇报.mp3")
        return

    # 选择生成方法
    if args.use_opencv:
        generate_video_with_opencv(
            image_paths=image_paths,
            audio_path=audio_path,
            output_path=output_path,
            duration_per_image=args.duration,
            resolution=resolution,
            fps=args.fps
        )
    else:
        # 检查 moviepy 是否可用
        try:
            import moviepy
            generate_video_with_moviepy(
                image_paths=image_paths,
                audio_path=audio_path,
                output_path=output_path,
                duration_per_image=args.duration,
                transition_duration=args.transition,
                resolution=resolution,
                fps=args.fps,
                add_title=not args.no_title
            )
        except ImportError:
            print("\n⚠️ moviepy 未安装，切换到 OpenCV 模式...")
            print("提示: 安装 moviepy 可获得更好的效果: pip install moviepy")
            generate_video_with_opencv(
                image_paths=image_paths,
                audio_path=audio_path,
                output_path=output_path,
                duration_per_image=args.duration,
                resolution=resolution,
                fps=args.fps
            )


if __name__ == "__main__":
    main()
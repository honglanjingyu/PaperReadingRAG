# demo/demo_remote_parse.py

"""
测试MinerU远程PDF解析功能
直接使用项目中的 RemotePDFParser 模块
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
load_dotenv(project_root / ".env")

import logging
from app.service.core.deepdoc.parser.remote_pdf_parser import RemotePDFParser, is_remote_parse_enabled

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 测试文件路径
test_pdf = project_root / "uploads_tmp" / "所有者用户.pdf"


if __name__ == "__main__":
    if not is_remote_parse_enabled():
        logger.error("远程解析未启用，请检查 .env 配置:")
        logger.error("  ENABLE_REMOTE_PARSE=true")
        logger.error("  PARSE_API_TOKEN=your_token")
        sys.exit(1)

    if not test_pdf.exists():
        logger.error(f"测试文件不存在: {test_pdf}")
        sys.exit(1)

    parser = RemotePDFParser()

    logger.info("=" * 60)
    logger.info("开始测试 MinerU 远程PDF解析")
    logger.info("=" * 60)
    logger.info(f"测试文件: {test_pdf.name}")
    logger.info(f"文件大小: {test_pdf.stat().st_size / 1024:.2f} KB")
    logger.info(f"模型版本: {os.getenv('MINERU_MODEL_VERSION', 'vlm')}")
    logger.info(f"启用表格识别: {os.getenv('MINERU_ENABLE_TABLE', 'true')}")
    logger.info("=" * 60)

    try:
        sections, tables = parser.parse_pdf(str(test_pdf))

        logger.info(f"\n解析结果:")
        logger.info(f"  段落数量: {len(sections)}")
        logger.info(f"  表格数量: {len(tables)}")

        if tables:
            logger.info(f"\n表格详情:")
            for i, table in enumerate(tables, 1):
                if table and len(table) > 0:
                    rows = len(table)
                    cols = len(table[0]) if table else 0
                    logger.info(f"  表格 {i}: {rows} 行 x {cols} 列")
        else:
            logger.warning(f"\n⚠️ 未检测到表格！")

        # 打印前几个段落
        if sections:
            logger.info(f"\n段落预览 (前3个):")
            for i, (text, style) in enumerate(sections[:3], 1):
                preview = text[:100].replace('\n', ' ')
                logger.info(f"  [{i}] ({style}): {preview}...")

    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n" + "=" * 60)
    logger.info("测试完成")
    logger.info("=" * 60)
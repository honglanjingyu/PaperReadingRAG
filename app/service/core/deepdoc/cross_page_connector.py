# app/service/core/deepdoc/cross_page_connector.py
"""跨页内容连接模块 - 简化版，远程解析已处理"""

from typing import List, Dict

from .models import PageContent


class CrossPageConnector:
    """跨页内容连接器 - 简化版"""

    def __init__(self, max_print: int = 5):
        self._max_print = max_print
        self._stats = {'paragraphs_merged': 0, 'tables_merged': 0}

    def connect(self, pages_content: List[PageContent], verbose: bool = False, is_remote: bool = False) -> List[
        PageContent]:
        """连接跨页内容 - 远程解析模式直接返回"""
        # 远程解析已经处理了跨页内容，直接返回
        if is_remote:
            if verbose:
                print("  远程解析模式：跳过跨页连接（已由MinerU API处理）")
            return pages_content

        if verbose:
            print("  非远程模式：跳过跨页连接")
        return pages_content

    def get_stats(self) -> Dict[str, int]:
        """获取连接统计信息"""
        return self._stats.copy()


__all__ = ['CrossPageConnector']
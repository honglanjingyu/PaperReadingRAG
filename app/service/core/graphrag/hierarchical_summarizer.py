# app/service/core/graphrag/hierarchical_summary.py
"""
层次摘要生成器 - 为每个社区生成摘要
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CommunitySummary:
    """社区摘要"""
    community_id: int
    title: str
    summary: str
    key_insights: List[str]
    entities: List[str]
    level: int


class HierarchicalSummarizer:
    """
    层次摘要生成器
    为每个社区生成摘要，支持多层次摘要
    """

    def __init__(self, llm_service=None, max_context_length: int = 4000):
        """
        初始化摘要生成器

        Args:
            llm_service: LLM 服务实例
            max_context_length: 最大上下文长度
        """
        self.llm_service = llm_service
        self.max_context_length = max_context_length
        self.summaries: Dict[int, CommunitySummary] = {}

        logger.info(f"HierarchicalSummarizer 初始化, llm_service={llm_service is not None}")

    def generate_summaries(
            self,
            communities: Dict,
            entity_details: Dict[str, Dict],
            original_text: str = ""
    ) -> Dict[int, CommunitySummary]:
        """
        为所有社区生成摘要

        Args:
            communities: 社区字典
            entity_details: 实体详情
            original_text: 原始文本（用于提取上下文）

        Returns:
            摘要字典
        """
        if not communities:
            return {}

        self.summaries.clear()

        for comm_id, comm in communities.items():
            summary = self._generate_community_summary(
                comm_id, comm, entity_details, original_text
            )
            if summary:
                self.summaries[comm_id] = summary

        logger.info(f"生成摘要完成: {len(self.summaries)} 个社区")
        return self.summaries

    def _generate_community_summary(
            self,
            comm_id: int,
            community,
            entity_details: Dict[str, Dict],
            original_text: str
    ) -> Optional[CommunitySummary]:
        """为单个社区生成摘要"""
        if not community.entities:
            return None

        # 构建实体上下文
        entity_context = self._build_entity_context(community.entities, entity_details)

        # 使用 LLM 生成摘要
        if self.llm_service:
            summary_text = self._generate_with_llm(community, entity_context, original_text)
        else:
            summary_text = self._generate_basic_summary(community, entity_context)

        if not summary_text:
            return None

        # 提取关键洞察
        key_insights = self._extract_key_insights(summary_text)

        # 生成标题
        title = self._generate_title(community, key_insights)

        return CommunitySummary(
            community_id=comm_id,
            title=title,
            summary=summary_text,
            key_insights=key_insights,
            entities=community.entities[:15],  # 限制数量
            level=getattr(community, 'level', 0)
        )

    def _build_entity_context(
            self,
            entities: List[str],
            entity_details: Dict[str, Dict]
    ) -> str:
        """构建实体上下文"""
        contexts = []

        for entity_name in entities[:20]:  # 限制数量
            if entity_name in entity_details:
                detail = entity_details[entity_name]
                contexts.append(
                    f"- {entity_name} ({detail.get('type', '概念')}): "
                    f"出现 {detail.get('frequency', 1)} 次"
                )
            else:
                contexts.append(f"- {entity_name}")

        return "\n".join(contexts)

    def _generate_with_llm(
            self,
            community,
            entity_context: str,
            original_text: str
    ) -> Optional[str]:
        """使用 LLM 生成摘要"""
        if not self.llm_service:
            return None

        # 构建 prompt
        prompt = f"""请为以下实体集合生成一个简洁的摘要。

## 实体列表
{entity_context}

## 原始文档片段
{original_text[:self.max_context_length]}

## 摘要要求
1. 总结这些实体共同构成了什么主题
2. 指出实体之间的核心关系
3. 提炼关键信息（如果实体涉及公司财报或行业分析，请提炼关键数据）

请输出一个 100-200 字的摘要："""

        try:
            response = self.llm_service.generate([{"role": "user", "content": prompt}])
            return response.strip() if response else None
        except Exception as e:
            logger.error(f"LLM 摘要生成失败: {e}")
            return self._generate_basic_summary(community, entity_context)

    def _generate_basic_summary(
            self,
            community,
            entity_context: str
    ) -> str:
        """生成基础摘要（不使用 LLM）"""
        entity_count = len(community.entities)
        keywords = getattr(community, 'keywords', [])

        if keywords:
            keywords_str = "、".join(keywords[:5])
            return f"该社区包含 {entity_count} 个相关实体，主要涉及 {keywords_str} 等主题。"
        else:
            return f"该社区包含 {entity_count} 个实体，实体间关联度为 {community.density:.2f}。"

    def _extract_key_insights(self, summary: str) -> List[str]:
        """从摘要中提取关键洞察"""
        if not summary:
            return []

        # 简单分割（基于句子）
        sentences = summary.replace('。', '。\n').split('\n')
        insights = [s.strip() for s in sentences if len(s.strip()) > 20][:5]

        return insights

    def _generate_title(self, community, key_insights: List[str]) -> str:
        """生成社区标题"""
        keywords = getattr(community, 'keywords', [])

        if keywords:
            return f"{keywords[0]}{'相关' if len(keywords) > 1 else ''}{'与' + keywords[1] if len(keywords) > 1 else ''}主题"
        elif key_insights:
            # 从第一个洞察中提取关键词
            words = key_insights[0][:30]
            return f"{words}..."
        else:
            return f"实体社区 ({community.size} 个实体)"

    def get_global_summary(self, all_summaries: Dict[int, CommunitySummary]) -> str:
        """生成全局摘要（基于所有社区摘要）"""
        if not all_summaries:
            return ""

        summaries_text = []
        for summary in all_summaries.values():
            summaries_text.append(f"## {summary.title}\n{summary.summary}")

        combined = "\n\n".join(summaries_text)

        if self.llm_service and len(all_summaries) > 1:
            prompt = f"""请基于以下社区摘要，生成一个整体的知识图谱摘要。

## 各社区摘要
{combined[:self.max_context_length]}

请输出一个 150-300 字的整体摘要，涵盖核心主题和关键发现："""

            try:
                response = self.llm_service.generate([{"role": "user", "content": prompt}])
                if response:
                    return response.strip()
            except Exception as e:
                logger.error(f"全局摘要生成失败: {e}")

        return combined[:500] + ("..." if len(combined) > 500 else "")
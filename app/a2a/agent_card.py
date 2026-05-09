# paperreadingrag/app/a2a/agent_card.py
"""
A2A Agent Card - 符合 A2A 协议的 Agent 发现信息
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class AgentSkill(BaseModel):
    """Agent 技能描述"""
    name: str = Field(..., description="技能名称")
    description: str = Field(..., description="技能描述")
    tags: List[str] = Field(default_factory=list, description="技能标签")
    examples: List[str] = Field(default_factory=list, description="示例问题")


class AgentCapability(BaseModel):
    """Agent 能力描述"""
    streaming: bool = Field(default=True, description="是否支持流式输出")
    memory: bool = Field(default=True, description="是否支持对话记忆")
    rerank: bool = Field(default=True, description="是否支持重排序")
    query_rewrite: bool = Field(default=True, description="是否支持查询改写")


class AgentCard(BaseModel):
    """A2A Agent 信息卡片 - 符合 A2A 协议"""

    # 必填字段
    name: str = Field(..., description="Agent 名称")
    description: str = Field(..., description="Agent 描述")
    version: str = Field(default="1.0.0", description="版本号")

    # 标识信息
    id: str = Field(..., description="Agent 唯一标识")
    url: str = Field(..., description="Agent 服务地址")

    # 能力信息
    skills: List[AgentSkill] = Field(default_factory=list, description="技能列表")
    capabilities: AgentCapability = Field(default_factory=AgentCapability, description="能力配置")

    # 接口信息
    endpoints: Dict[str, str] = Field(default_factory=dict, description="API 端点")

    # 元信息
    provider: str = Field(default="PaperReadingRAG", description="提供者")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """序列化为字典"""
        data = super().model_dump(**kwargs)
        # 确保 URL 格式正确
        if not data["url"].endswith("/"):
            data["url"] += "/"
        return data


def get_agent_card(base_url: str = None) -> AgentCard:
    """
    获取 Agent 卡片信息

    Args:
        base_url: Agent 服务的基础 URL

    Returns:
        AgentCard: Agent 信息卡片
    """
    import os

    if base_url is None:
        base_url = os.getenv("A2A_BASE_URL", "http://localhost:8001")

    # 确保 URL 格式正确
    if not base_url.startswith("http"):
        base_url = f"http://{base_url}"
    base_url = base_url.rstrip("/")

    return AgentCard(
        id="paperreadingrag",
        name="PaperReadingRAG",
        description="专业的文档问答 RAG Agent，支持 PDF、DOCX、TXT 等格式的文档解析、检索和智能问答",
        version="1.0.0",
        url=base_url,

        skills=[
            AgentSkill(
                name="rag_search",
                description="从知识库中检索相关文档片段，支持混合检索和重排序",
                tags=["检索", "RAG", "文档"],
                examples=[
                    "请帮我查找关于 RAG 技术的内容",
                    "文档中提到了什么关键数据？",
                    "总结一下这个文档的主要内容"
                ]
            ),
            AgentSkill(
                name="document_qa",
                description="基于文档内容的智能问答，支持多轮对话",
                tags=["问答", "对话", "文档"],
                examples=[
                    "这个文档的主要观点是什么？",
                    "文档中的技术方案有哪些优缺点？",
                    "可以详细解释一下这个概念吗？"
                ]
            ),
            AgentSkill(
                name="knowledge_retrieval",
                description="纯检索接口，只返回相关文档片段，不生成答案",
                tags=["检索", "文档"],
                examples=[
                    "查找关于 XX 的内容",
                    "找到包含特定关键词的段落"
                ]
            )
        ],

        capabilities=AgentCapability(
            streaming=True,
            memory=True,
            rerank=True,
            query_rewrite=True
        ),

        endpoints={
            "health": f"{base_url}/a2a/health",
            "rag_search": f"{base_url}/a2a/rag/search",
            "rag_ask": f"{base_url}/a2a/rag/ask",
            "docs": f"{base_url}/docs"
        },

        provider="PaperReadingRAG System"
    )
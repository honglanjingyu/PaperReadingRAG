# paperreadingrag/app/a2a/well_known.py
"""
A2A Agent 发现端点 - 实现 /.well-known/agent.json
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any

from .agent_card import get_agent_card, AgentCard

# 创建路由器
router = APIRouter(tags=["A2A Discovery"])


@router.get("/.well-known/agent.json")
async def get_agent_json() -> Dict[str, Any]:
    """
    A2A 协议标准发现端点

    返回 Agent 的能力卡片，供其他 Agent 发现和调用。

    符合 A2A 协议规范:
    - 路径: /.well-known/agent.json
    - 返回: AgentCard JSON
    """
    card = get_agent_card()
    return card.model_dump()


@router.get("/.well-known/agent-card")
async def get_agent_card_alt() -> Dict[str, Any]:
    """
    备选发现端点（兼容某些客户端）
    """
    card = get_agent_card()
    return card.model_dump()


@router.get("/a2a/capabilities")
async def get_capabilities() -> Dict[str, Any]:
    """
    获取 Agent 能力摘要（简化版）
    """
    card = get_agent_card()
    return {
        "agent_id": card.id,
        "agent_name": card.name,
        "version": card.version,
        "skills": [
            {"name": s.name, "description": s.description, "tags": s.tags}
            for s in card.skills
        ],
        "capabilities": card.capabilities.model_dump(),
        "endpoints": card.endpoints
    }


@router.get("/a2a/info")
async def get_agent_info() -> Dict[str, Any]:
    """
    获取 Agent 基本信息（用于调试）
    """
    card = get_agent_card()
    return {
        "name": card.name,
        "description": card.description,
        "version": card.version,
        "available_skills": [s.name for s in card.skills],
        "supports_streaming": card.capabilities.streaming,
        "supports_memory": card.capabilities.memory,
        "endpoints": card.endpoints
    }
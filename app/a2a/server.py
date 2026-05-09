# paperreadingrag/app/a2a/server.py
"""
A2A 服务端 - 包含 agent.json 发现端点
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.service.core.rag import (
    enhanced_search_with_hybrid_and_rerank,
    generate_answer,
)
from app.service.core.memory import get_memory_manager

logger = logging.getLogger(__name__)


# ============================================================
# 请求/响应模型
# ============================================================

class A2ARAGRequest(BaseModel):
    """A2A RAG 检索请求"""
    query: str = Field(..., description="查询问题")
    session_id: Optional[str] = Field(None, description="会话ID")
    top_k: int = Field(5, description="返回文档数量")
    recall_k: int = Field(10, description="召回数量")
    need_answer: bool = Field(True, description="是否生成答案")
    include_documents: bool = Field(True, description="是否包含检索文档")


class A2ARAGResponse(BaseModel):
    """A2A RAG 检索响应"""
    success: bool = Field(..., description="是否成功")
    answer: Optional[str] = Field(None, description="生成的答案")
    documents: Optional[list] = Field(None, description="检索到的文档")
    session_id: Optional[str] = Field(None, description="会话ID")
    error: Optional[str] = Field(None, description="错误信息")


class A2AHealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    agent_name: str
    capabilities: list


# ============================================================
# A2A 服务类
# ============================================================

class A2AService:
    """A2A 服务 - 为其他 Agent 提供 RAG 能力"""

    def __init__(self):
        self.agent_name = "PaperReadingRAG"
        self.capabilities = [
            "rag_search",
            "document_qa",
            "knowledge_retrieval"
        ]
        logger.info(f"A2A 服务初始化: {self.agent_name}")

    async def process_rag_request(self, request: A2ARAGRequest) -> A2ARAGResponse:
        """处理 RAG 请求（非流式）"""
        logger.info(f"A2A: 收到 RAG 请求: query='{request.query[:50]}...'")

        try:
            # 获取或创建会话
            memory_manager = get_memory_manager()
            session_id = memory_manager.get_or_create_session(request.session_id)

            # 执行增强检索
            retrieval_result = enhanced_search_with_hybrid_and_rerank(
                question=request.query,
                index_name=os.getenv("VECTOR_INDEX_NAME", "rag_documents"),
                recall_k=request.recall_k,
                top_k=request.top_k,
                keyword_weight=float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.4")),
                vector_weight=float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6")),
                enable_rerank=os.getenv("ENABLE_RERANK", "true").lower() == "true",
                enable_query_rewrite=os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true",
                similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.3")),
                rerank_type=os.getenv("RERANK_TYPE", "remote"),
                verbose=False
            )

            if not retrieval_result:
                logger.error("检索返回为空")
                return A2ARAGResponse(
                    success=False,
                    error="检索失败：返回为空",
                    session_id=session_id
                )

            if not retrieval_result.get("success"):
                error_msg = retrieval_result.get("error", "检索失败")
                logger.error(f"检索失败: {error_msg}")
                return A2ARAGResponse(
                    success=False,
                    error=error_msg,
                    session_id=session_id
                )

            results = retrieval_result.get("results", [])
            rewritten_query = retrieval_result.get("rewritten_query", request.query)

            logger.info(f"检索完成: 找到 {len(results)} 个相关文档")

            # 生成答案（如果需要）
            answer = None
            if request.need_answer and results:
                try:
                    generation_result = generate_answer(
                        question=rewritten_query,
                        results=results,
                        history=None,
                        template_name="detailed",
                        verbose=False,
                        preview_answer=False
                    )

                    if generation_result and generation_result.get("success"):
                        answer = generation_result.get("answer")

                        # 保存到对话记忆
                        if answer:
                            memory_manager.add_message(session_id, "user", request.query)
                            memory_manager.add_message(session_id, "assistant", answer)
                            logger.info(f"答案生成成功，长度: {len(answer)}")
                    else:
                        error = generation_result.get("error") if generation_result else "生成失败"
                        logger.warning(f"答案生成失败: {error}")
                except Exception as e:
                    logger.error(f"生成答案异常: {e}")

            # 准备返回的文档（简化版）
            documents = None
            if request.include_documents:
                documents = [
                    {
                        "content": doc.get("content", "")[:500],
                        "score": doc.get("score", 0),
                        "document_name": doc.get("document_name", "")
                    }
                    for doc in results[:request.top_k]
                ]

            return A2ARAGResponse(
                success=True,
                answer=answer,
                documents=documents,
                session_id=session_id
            )

        except Exception as e:
            logger.error(f"A2A 处理请求失败: {e}", exc_info=True)
            return A2ARAGResponse(
                success=False,
                error=str(e),
                session_id=request.session_id
            )


# ============================================================
# Agent 发现端点
# ============================================================

def get_agent_card(base_url: str = None) -> dict:
    """
    获取 Agent 卡片信息

    Returns:
        dict: Agent 信息卡片
    """
    if base_url is None:
        base_url = os.getenv("A2A_BASE_URL", "http://localhost:8001")

    if not base_url.startswith("http"):
        base_url = f"http://{base_url}"
    base_url = base_url.rstrip("/")

    return {
        "id": "paperreadingrag",
        "name": "PaperReadingRAG",
        "description": "专业的文档问答 RAG Agent，支持 PDF、DOCX、TXT 等格式的文档解析、检索和智能问答",
        "version": "1.0.0",
        "url": base_url,
        "skills": [
            {
                "name": "rag_search",
                "description": "从知识库中检索相关文档片段，支持混合检索和重排序",
                "tags": ["检索", "RAG", "文档"],
                "examples": [
                    "请帮我查找关于 RAG 技术的内容",
                    "文档中提到了什么关键数据？"
                ]
            },
            {
                "name": "document_qa",
                "description": "基于文档内容的智能问答，支持多轮对话",
                "tags": ["问答", "对话", "文档"],
                "examples": [
                    "这个文档的主要观点是什么？",
                    "可以详细解释一下这个概念吗？"
                ]
            },
            {
                "name": "knowledge_retrieval",
                "description": "纯检索接口，只返回相关文档片段，不生成答案",
                "tags": ["检索", "文档"],
                "examples": [
                    "查找关于 XX 的内容"
                ]
            }
        ],
        "capabilities": {
            "streaming": True,
            "memory": True,
            "rerank": True,
            "query_rewrite": True
        },
        "endpoints": {
            "health": f"{base_url}/a2a/health",
            "rag_search": f"{base_url}/a2a/rag/search",
            "rag_ask": f"{base_url}/a2a/rag/ask",
            "docs": f"{base_url}/docs"
        },
        "provider": "PaperReadingRAG System"
    }


def create_well_known_router() -> APIRouter:
    """创建 .well-known 发现端点路由器"""
    router = APIRouter(tags=["A2A Discovery"])

    @router.get("/.well-known/agent.json")
    async def get_agent_json():
        """A2A 协议标准发现端点"""
        return get_agent_card()

    @router.get("/.well-known/agent-card")
    async def get_agent_card_alt():
        """备选发现端点"""
        return get_agent_card()

    @router.get("/a2a/capabilities")
    async def get_capabilities():
        """获取 Agent 能力摘要"""
        card = get_agent_card()
        return {
            "agent_id": card["id"],
            "agent_name": card["name"],
            "version": card["version"],
            "skills": [
                {"name": s["name"], "description": s["description"], "tags": s["tags"]}
                for s in card["skills"]
            ],
            "capabilities": card["capabilities"],
            "endpoints": card["endpoints"]
        }

    @router.get("/a2a/info")
    async def get_agent_info():
        """获取 Agent 基本信息"""
        card = get_agent_card()
        return {
            "name": card["name"],
            "description": card["description"],
            "version": card["version"],
            "available_skills": [s["name"] for s in card["skills"]],
            "supports_streaming": card["capabilities"]["streaming"],
            "supports_memory": card["capabilities"]["memory"],
            "endpoints": card["endpoints"]
        }

    return router


# ============================================================
# 创建 A2A API 路由
# ============================================================

def create_a2a_router(a2a_service: A2AService) -> APIRouter:
    """创建 A2A 路由"""
    router = APIRouter(prefix="/a2a", tags=["A2A Agent Communication"])

    @router.get("/health", response_model=A2AHealthResponse)
    async def a2a_health():
        """A2A 健康检查"""
        return A2AHealthResponse(
            status="healthy",
            agent_name=a2a_service.agent_name,
            capabilities=a2a_service.capabilities
        )

    @router.post("/rag/search", response_model=A2ARAGResponse)
    async def a2a_rag_search(request: A2ARAGRequest):
        """A2A RAG 检索接口"""
        return await a2a_service.process_rag_request(request)

    @router.post("/rag/ask")
    async def a2a_rag_ask(request: A2ARAGRequest):
        """A2A RAG 问答接口（简化版）"""
        result = await a2a_service.process_rag_request(request)
        # 确保 result 不为 None
        if result is None:
            return {
                "success": False,
                "answer": None,
                "session_id": request.session_id,
                "error": "服务内部错误"
            }
        return {
            "success": result.success,
            "answer": result.answer,
            "session_id": result.session_id,
            "error": result.error
        }

    return router


# ============================================================
# 集成到主应用
# ============================================================

def integrate_a2a_to_app(app: FastAPI):
    """将 A2A 路由集成到现有 FastAPI 应用"""
    a2a_service = A2AService()
    router = create_a2a_router(a2a_service)
    app.include_router(router)

    # 添加 .well-known 发现端点
    well_known_router = create_well_known_router()
    app.include_router(well_known_router)

    logger.info("A2A 路由已集成到 PaperReadingRAG")
    logger.info("  - A2A API: /a2a/*")
    logger.info("  - Agent Discovery: /.well-known/agent.json")
    return a2a_service


# 独立运行 A2A 服务
async def run_a2a_server(host: str = "0.0.0.0", port: int = 8001):
    """独立运行 A2A 服务"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    app = FastAPI(
        title="PaperReadingRAG A2A Server",
        description="A2A 协议服务，为其他 Agent 提供 RAG 检索能力",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    a2a_service = A2AService()
    router = create_a2a_router(a2a_service)
    app.include_router(router)

    well_known_router = create_well_known_router()
    app.include_router(well_known_router)

    logger.info(f"A2A 服务启动: http://{host}:{port}")
    logger.info(f"  - Agent Discovery: http://{host}:{port}/.well-known/agent.json")
    logger.info(f"  - Health Check: http://{host}:{port}/a2a/health")

    import uvicorn
    await uvicorn.serve(
        uvicorn.Config(app, host=host, port=port, log_level="info")
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_a2a_server())
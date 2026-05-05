# app/service/core/prompt/templates.py
"""
Prompt模板定义
"""

# 系统指令模板
SYSTEM_INSTRUCTION = """你是一个专业的文档问答助手。请根据提供的文档内容，准确、完整地回答用户的问题。

注意事项：
1. 只根据提供的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，请明确告知用户
3. 回答要简洁、清晰、有条理
4. 必要时可以引用文档中的原文"""

# 简单模板
SIMPLE_TEMPLATE = """请根据以下文档内容回答问题。

{context}

问题：{question}

请给出答案："""

# 详细模板（带引用标记）
DETAILED_TEMPLATE = """你是一个专业的文档分析助手。

## 参考文档内容
{context}

## 用户问题
{question}

## 回答要求
1. 基于上述文档内容回答
2. 如果引用了文档内容，请标注出处
3. 如果文档中没有相关信息，请明确说明
4. 回答要准确、完整

## 回答
"""

# 对话模板（支持历史记录）
CONVERSATION_TEMPLATE = """你是一个专业的文档问答助手。

## 参考文档内容
{context}

## 对话历史
{history}

## 当前问题
{question}

请基于参考文档和对话历史回答问题："""

# 简洁模板（用于快速测试）
CONCISE_TEMPLATE = """文档内容：
{context}

问题：{question}

答案："""

PROMPT_TEMPLATES = {
    "simple": SIMPLE_TEMPLATE,
    "detailed": DETAILED_TEMPLATE,
    "conversation": CONVERSATION_TEMPLATE,
    "concise": CONCISE_TEMPLATE,
}


def get_template(template_name: str = "simple") -> str:
    """获取指定名称的模板"""
    return PROMPT_TEMPLATES.get(template_name, SIMPLE_TEMPLATE)


def get_system_instruction() -> str:
    """获取系统指令"""
    return SYSTEM_INSTRUCTION
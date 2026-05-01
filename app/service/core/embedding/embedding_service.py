"""
向量化服务（向后兼容层）
"""

from .embedding_manager import get_embedding_manager, EmbeddingManager

# 导出主要功能
def generate_embedding(text):
    """生成单个文本的向量"""
    return get_embedding_manager().generate_embedding(text)


def generate_embeddings(texts):
    """批量生成文本向量"""
    return get_embedding_manager().generate_embeddings(texts)


def get_vector_field_name():
    """获取向量字段名"""
    return get_embedding_manager().get_vector_field_name()


def switch_to_remote(**kwargs):
    """切换到远程模型"""
    get_embedding_manager().switch_to_remote(**kwargs)


def switch_to_local(**kwargs):
    """切换到本地模型"""
    get_embedding_manager().switch_to_local(**kwargs)


def get_model_info():
    """获取模型信息"""
    return get_embedding_manager().get_model_info()


# 保持向后兼容的类
class EmbeddingService:
    """向后兼容的包装类"""

    def __init__(self):
        self.manager = get_embedding_manager()

    def generate_embedding(self, text):
        return self.manager.generate_embedding(text)

    def generate_embeddings(self, texts):
        return self.manager.generate_embeddings(texts)

    def get_vector_field_name(self):
        return self.manager.get_vector_field_name()

    @property
    def dimension(self):
        return self.manager.dimension
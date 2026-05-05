# app/service/synonymlist/loader.py
"""
同义词表加载器 - 从 YAML 文件读取同义词
"""

import os
import logging
from typing import Dict, List, Optional
from pathlib import Path

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logging.warning("PyYAML 未安装，请运行: pip install pyyaml")

logger = logging.getLogger(__name__)


class SynonymLoader:
    """同义词加载器 - 从 YAML 文件读取同义词"""

    def __init__(self, synonym_dir: Optional[str] = None):
        """
        初始化同义词加载器

        Args:
            synonym_dir: 同义词表目录路径，默认为 app/service/synonymlist
        """
        if synonym_dir is None:
            # 获取当前文件所在目录
            current_dir = Path(__file__).parent
            synonym_dir = str(current_dir)

        self.synonym_dir = Path(synonym_dir)
        self.synonyms: Dict[str, List[str]] = {}
        self._loaded = False

        if not YAML_AVAILABLE:
            logger.error("PyYAML 未安装，无法加载同义词表")

    def load_all(self) -> Dict[str, List[str]]:
        """
        加载所有同义词文件

        Returns:
            同义词字典
        """
        if not YAML_AVAILABLE:
            return {}

        if self._loaded:
            return self.synonyms

        self.synonyms = {}

        if not self.synonym_dir.exists():
            logger.warning(f"同义词目录不存在: {self.synonym_dir}")
            return {}

        # 加载所有 yaml/yml 文件
        yaml_files = list(self.synonym_dir.glob("*.yaml")) + list(self.synonym_dir.glob("*.yml"))

        for yaml_file in yaml_files:
            if yaml_file.name == "__init__.py":
                continue
            self._load_file(yaml_file)

        self._loaded = True
        logger.info(f"加载同义词完成: {len(self.synonyms)} 个词条, 来自 {len(yaml_files)} 个文件")

        return self.synonyms

    def _load_file(self, file_path: Path) -> None:
        """
        加载单个 YAML 文件

        Args:
            file_path: YAML 文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if not data:
                logger.warning(f"文件为空: {file_path}")
                return

            for key, values in data.items():
                if isinstance(values, list):
                    if key in self.synonyms:
                        # 合并同义词
                        existing = set(self.synonyms[key])
                        existing.update(values)
                        self.synonyms[key] = list(existing)
                    else:
                        self.synonyms[key] = values
                else:
                    logger.warning(f"跳过非列表格式: {file_path}:{key}")

            logger.debug(f"加载文件: {file_path.name}, {len(data)} 个词条")

        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {e}")

    def get_synonyms(self, word: str) -> List[str]:
        """
        获取指定词的同义词

        Args:
            word: 原词

        Returns:
            同义词列表
        """
        if not self._loaded:
            self.load_all()

        return self.synonyms.get(word, [])

    def get_all_synonyms(self) -> Dict[str, List[str]]:
        """获取所有同义词"""
        if not self._loaded:
            self.load_all()
        return self.synonyms

    def add_synonym(self, word: str, synonyms: List[str]) -> None:
        """
        动态添加同义词

        Args:
            word: 原词
            synonyms: 同义词列表
        """
        if not self._loaded:
            self.load_all()

        if word in self.synonyms:
            existing = set(self.synonyms[word])
            existing.update(synonyms)
            self.synonyms[word] = list(existing)
        else:
            self.synonyms[word] = synonyms

        logger.info(f"添加同义词: {word} -> {synonyms}")

    def save_to_file(self, file_path: str = None) -> bool:
        """
        保存同义词到文件

        Args:
            file_path: 保存路径，默认保存到 custom.yml

        Returns:
            是否保存成功
        """
        if not YAML_AVAILABLE:
            logger.error("PyYAML 未安装，无法保存")
            return False

        if file_path is None:
            file_path = self.synonym_dir / "custom.yml"
        else:
            file_path = Path(file_path)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.synonyms, f, allow_unicode=True, default_flow_style=False)
            logger.info(f"保存同义词到: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存失败: {e}")
            return False


# 全局单例
_synonym_loader = None


def get_synonym_loader() -> SynonymLoader:
    """获取同义词加载器实例（单例）"""
    global _synonym_loader
    if _synonym_loader is None:
        _synonym_loader = SynonymLoader()
    return _synonym_loader


def get_synonyms() -> Dict[str, List[str]]:
    """获取所有同义词"""
    return get_synonym_loader().get_all_synonyms()


__all__ = ['SynonymLoader', 'get_synonym_loader', 'get_synonyms']
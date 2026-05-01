# app/service/core/deepdoc/models.py
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class LayoutType(Enum):
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TEXT_WITH_TABLES = "text_with_tables"
    TEXT_WITH_IMAGES = "text_with_images"
    MIXED = "mixed"

@dataclass
class TextBlock:
    page_num: int
    content: str
    x0: float = 0
    y0: float = 0
    x1: float = 0
    y1: float = 0
    column: int = 0

@dataclass
class TableBlock:
    page_num: int
    data: List[List[str]]
    x0: float = 0
    y0: float = 0
    x1: float = 0
    y1: float = 0
    is_continued: bool = False
    continued_from_page: Optional[int] = None

@dataclass
class PageContent:
    page_num: int
    text_blocks: List[TextBlock] = field(default_factory=list)
    tables: List[TableBlock] = field(default_factory=list)
    layout_type: LayoutType = LayoutType.SINGLE_COLUMN
    columns: int = 1

@dataclass
class ParsedDocument:
    file_path: str
    file_name: str
    file_type: str
    pages: List[PageContent]
    total_pages: int
    cleaned_text: str = ""
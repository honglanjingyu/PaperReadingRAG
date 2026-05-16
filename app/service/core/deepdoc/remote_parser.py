# app/service/core/deepdoc/remote_parser.py
"""
远程文档解析器 - 使用 MinerU 云端 API 解析文档
支持: PDF, DOCX, TXT, Excel, PPT, HTML, Markdown 等
"""

import os
import re
import time
from io import BytesIO
from typing import List, Tuple, Optional, Dict, Any
import tempfile
import logging
from datetime import datetime
from pathlib import Path
import requests
import zipfile

logger = logging.getLogger(__name__)


class RemoteDocumentParser:
    """远程文档解析器 - 使用 MinerU API"""

    # 支持的文件类型
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.txt': 'txt',
        '.md': 'md',
        '.markdown': 'md',
        '.xlsx': 'xlsx',
        '.xls': 'xls',
        '.ppt': 'ppt',
        '.pptx': 'ppt',
        '.html': 'html',
        '.htm': 'html',
    }

    def __init__(self, api_token: str = None):
        """初始化远程文档解析器"""
        self.api_token = api_token or os.getenv("PARSE_API_TOKEN")
        self.model_version = os.getenv("MINERU_MODEL_VERSION", "vlm")
        self.enable_table = os.getenv("MINERU_ENABLE_TABLE", "true").lower() == "true"
        self.enable_formula = os.getenv("MINERU_ENABLE_FORMULA", "true").lower() == "true"
        self.is_ocr = os.getenv("MINERU_IS_OCR", "false").lower() == "true"
        self.language = os.getenv("MINERU_LANGUAGE", "ch")

        self.base_url = "https://mineru.net"
        self._last_parse_result: Dict[str, Any] = {}

        if not self.api_token:
            logger.warning("PARSE_API_TOKEN 未配置，远程文档解析不可用")

    def is_available(self) -> bool:
        """检查远程解析器是否可用"""
        return bool(self.api_token)

    def _upload_file(self, file_path: str) -> Optional[str]:
        """上传文件到 MinerU，返回 batch_id"""
        if not self.is_available():
            return None

        try:
            url = f"{self.base_url}/api/v4/file-urls/batch"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_token}"
            }

            file_name = os.path.basename(file_path)
            data = {
                "files": [{"name": file_name}],
                "model_version": self.model_version
            }

            if self.enable_table is not None:
                data["enable_table"] = self.enable_table
            if self.enable_formula is not None:
                data["enable_formula"] = self.enable_formula
            if self.is_ocr is not None:
                data["is_ocr"] = self.is_ocr
            if self.language:
                data["language"] = self.language

            logger.info(f"申请上传URL: {file_name}")
            response = requests.post(url, headers=headers, json=data, timeout=30)

            if response.status_code != 200:
                logger.error(f"申请上传URL失败: {response.status_code}")
                return None

            result = response.json()
            if result.get("code") != 0:
                logger.error(f"申请上传URL失败: {result.get('msg')}")
                return None

            batch_id = result["data"]["batch_id"]
            file_url = result["data"]["file_urls"][0]

            logger.info(f"获取到上传URL, batch_id={batch_id}")

            with open(file_path, 'rb') as f:
                upload_response = requests.put(file_url, data=f, timeout=60)

            if upload_response.status_code not in (200, 201):
                logger.error(f"文件上传失败: {upload_response.status_code}")
                return None

            logger.info("文件上传成功")
            return batch_id

        except Exception as e:
            logger.error(f"上传文件失败: {e}")
            return None

    def _wait_for_result(self, batch_id: str, timeout: int = 300) -> Optional[str]:
        """等待解析完成并返回结果 ZIP URL"""
        if not self.is_available():
            return None

        url = f"{self.base_url}/api/v4/extract-results/batch/{batch_id}"
        headers = {"Authorization": f"Bearer {self.api_token}"}

        start_time = time.time()
        interval = 3

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, headers=headers, timeout=30)

                if response.status_code != 200:
                    time.sleep(interval)
                    continue

                result = response.json()
                if result.get("code") != 0:
                    time.sleep(interval)
                    continue

                extract_results = result["data"].get("extract_result", [])
                if not extract_results:
                    time.sleep(interval)
                    continue

                file_result = extract_results[0]
                state = file_result.get("state")

                if state == "done":
                    zip_url = file_result.get("full_zip_url")
                    logger.info("解析完成")
                    return zip_url
                elif state == "failed":
                    err_msg = file_result.get("err_msg", "未知错误")
                    logger.error(f"解析失败: {err_msg}")
                    return None
                else:
                    elapsed = int(time.time() - start_time)
                    logger.info(f"解析中... 状态={state}, 已等待{elapsed}秒")
                    time.sleep(interval)

            except Exception as e:
                logger.warning(f"查询结果异常: {e}")
                time.sleep(interval)

        logger.error(f"等待超时 ({timeout}秒)")
        return None

    def _download_markdown(self, zip_url: str) -> Optional[str]:
        """下载 ZIP 包并提取 full.md 内容"""
        try:
            response = requests.get(zip_url, timeout=60)
            if response.status_code != 200:
                logger.error(f"下载ZIP失败: {response.status_code}")
                return None

            zip_data = BytesIO(response.content)

            with zipfile.ZipFile(zip_data, 'r') as zip_ref:
                for file_name in zip_ref.namelist():
                    if file_name.endswith('full.md') or file_name.endswith('.md'):
                        with zip_ref.open(file_name) as md_file:
                            content = md_file.read().decode('utf-8')
                            logger.info(f"找到 Markdown 文件: {file_name}, 长度: {len(content)}")
                            return content

            logger.error(f"ZIP包中未找到 Markdown 文件")
            return None

        except Exception as e:
            logger.error(f"下载解析结果失败: {e}")
            return None

    def parse_document(
            self,
            file_path_or_binary,
            from_page: int = 0,
            to_page: int = 100000
    ) -> Tuple[List[Tuple[str, str]], List[List[List[str]]], int]:
        """
        解析文档（支持所有格式）

        Returns:
            (sections, tables, total_pages): 段落列表、表格列表、总页数
        """
        if not self.is_available():
            logger.error("远程文档解析器不可用")
            return [], [], 0

        temp_file = None
        original_file_name = None
        total_pages = 0

        try:
            if isinstance(file_path_or_binary, (bytes, BytesIO)):
                if isinstance(file_path_or_binary, BytesIO):
                    binary_data = file_path_or_binary.getvalue()
                else:
                    binary_data = file_path_or_binary

                original_file_name = "uploaded_document"
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                temp_file.write(binary_data)
                temp_file.close()
                file_path = temp_file.name
            else:
                file_path = file_path_or_binary
                original_file_name = os.path.basename(file_path)

            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return [], [], 0

            file_size = os.path.getsize(file_path) / 1024
            logger.info(f"正在调用 MinerU API 解析文档: {original_file_name} ({file_size:.2f} KB)")

            batch_id = self._upload_file(file_path)
            if not batch_id:
                return [], [], 0

            zip_url = self._wait_for_result(batch_id)
            if not zip_url:
                return [], [], 0

            markdown_content = self._download_markdown(zip_url)
            if not markdown_content:
                return [], [], 0

            logger.info(f"API 调用成功，内容长度: {len(markdown_content)} 字符")

            sections, tables = self._parse_markdown(markdown_content)

            # ========== 估算页数（基于文本长度） ==========
            # 假设每页约 3000 字符（中文文档的平均值）
            total_pages = max(1, len(markdown_content) // 3000 + 1)

            # 对于 PDF 文件，尝试从文件名或内容中获取更精确的页数
            file_ext = os.path.splitext(original_file_name)[1].lower()
            if file_ext == '.pdf':
                # 对于 PDF，可以尝试从返回的内容中提取页数标记
                # MinerU 返回的 Markdown 中可能包含 "Page 1" 等标记
                page_markers = re.findall(r'Page\s+(\d+)', markdown_content, re.IGNORECASE)
                if page_markers:
                    max_page = max(int(p) for p in page_markers)
                    total_pages = max(total_pages, max_page)

            self._last_parse_result = {
                "file_name": original_file_name,
                "markdown_content": markdown_content,
                "sections": sections,
                "tables": tables,
                "total_pages": total_pages
            }

            logger.info(f"解析完成: {len(sections)} 段落, {len(tables)} 表格, 估算页数: {total_pages}")
            return sections, tables, total_pages

        except Exception as e:
            logger.error(f"远程文档解析失败: {e}")
            return [], [], 0
        finally:
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

    def parse_pdf(self, file_path_or_binary, from_page: int = 0, to_page: int = 100000):
        """解析 PDF（兼容旧方法名）"""
        sections, tables, pages = self.parse_document(file_path_or_binary, from_page, to_page)
        return sections, tables

    def _parse_markdown(self, markdown_content: str) -> Tuple[List[Tuple[str, str]], List[List[List[str]]]]:
        """将 MinerU 返回的 Markdown 解析为段落和表格"""
        if not markdown_content:
            return [], []

        lines = markdown_content.split('\n')
        total_lines = len(lines)

        tables = []
        table_ranges = []

        i = 0
        while i < total_lines:
            line = lines[i].strip()

            # 检测 Markdown 表格
            if line.startswith('|') and line.endswith('|'):
                start = i
                table_lines = []
                while i < total_lines:
                    current = lines[i].strip()
                    if current.startswith('|') and current.endswith('|'):
                        table_lines.append(current)
                        i += 1
                    else:
                        break

                if len(table_lines) >= 2:
                    table_data = self._parse_md_table(table_lines)
                    if table_data:
                        tables.append(table_data)
                        table_ranges.append((start, i, len(tables) - 1))
                continue

            i += 1

        # 解析段落
        sections = []
        current_paragraph = []

        i = 0
        while i < total_lines:
            in_table = any(start <= i < end for start, end, _ in table_ranges)

            if in_table:
                i += 1
                continue

            line = lines[i]
            stripped = line.strip()

            if not stripped:
                if current_paragraph:
                    text = ' '.join(current_paragraph).strip()
                    if text:
                        sections.append((text, "paragraph"))
                    current_paragraph = []
                i += 1
                continue

            if stripped.startswith('#'):
                if current_paragraph:
                    text = ' '.join(current_paragraph).strip()
                    if text:
                        sections.append((text, "paragraph"))
                    current_paragraph = []

                level = 0
                for ch in stripped:
                    if ch == '#':
                        level += 1
                    else:
                        break
                level = min(level, 6)
                title = stripped[level:].strip()
                title = self._clean_text(title)
                if title:
                    sections.append((title, f"heading_{level}"))
                i += 1
                continue

            cleaned = self._clean_text(stripped)
            if cleaned:
                current_paragraph.append(cleaned)
            i += 1

        if current_paragraph:
            text = ' '.join(current_paragraph).strip()
            if text:
                sections.append((text, "paragraph"))

        return sections, tables

    def _parse_md_table(self, table_lines: List[str]) -> List[List[str]]:
        """解析 Markdown 格式的表格"""
        if not table_lines or len(table_lines) < 2:
            return []

        result = []
        for line_idx, line in enumerate(table_lines):
            cells = line.split('|')
            cells = cells[1:-1]
            cells = [c.strip() for c in cells]

            if line_idx == 1 and all(self._is_separator(c) for c in cells):
                continue

            if not any(c for c in cells):
                continue

            cleaned_row = [self._clean_cell(cell) for cell in cells]
            if cleaned_row:
                result.append(cleaned_row)

        if result:
            max_cols = max(len(row) for row in result)
            for row in result:
                while len(row) < max_cols:
                    row.append("")

        return result

    def _is_separator(self, cell: str) -> bool:
        """判断是否为表格分隔行单元格"""
        if not cell:
            return False
        cleaned = cell.replace(' ', '').replace(':', '')
        return all(c == '-' for c in cleaned)

    def _clean_cell(self, text: str) -> str:
        """清理单元格内容"""
        if not text:
            return ""

        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\$[^$]+\$', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        if len(text) > 2000:
            text = text[:2000] + "..."

        return text

    def _clean_text(self, text: str) -> str:
        """清理普通文本"""
        if not text:
            return ""

        if text.startswith('#'):
            text = re.sub(r'^#+\s*', '', text)

        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def get_last_parse_result(self) -> Dict[str, Any]:
        """获取最后一次解析的结果"""
        return self._last_parse_result.copy()

    def save_chunked_report(
        self,
        file_name: str,
        chunks: List,
        sections: List = None,
        tables: List = None,
        markdown_content: str = None
    ) -> Optional[str]:
        """保存带分块结果的解析报告"""
        if sections is None:
            sections = self._last_parse_result.get("sections", [])
        if tables is None:
            tables = self._last_parse_result.get("tables", [])
        if markdown_content is None:
            markdown_content = self._last_parse_result.get("markdown_content", "")

        if not file_name:
            return None

        # 创建报告目录
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent.parent
        report_dir = project_root / "docparselist"
        report_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(file_name).stem
        safe_base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
        report_name = f"{date_str}_{safe_base_name}.md"
        report_path = report_dir / report_name

        report_lines = [
            f"# 文档解析报告",
            f"",
            f"**原始文件**: `{file_name}`",
            f"**解析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**解析引擎**: MinerU API",
            f"",
            f"---",
            f"",
            f"## 解析统计",
            f"",
            f"| 项目 | 数量 |",
            f"|------|------|",
            f"| 文字块数量 | {len(sections)} |",
            f"| 表格数量 | {len(tables)} |",
        ]
        if chunks:
            report_lines.append(f"| 分块数量 | {len(chunks)} |")
        report_lines.append(f"| Markdown 长度 | {len(markdown_content)} 字符 |")

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            logger.info(f"解析报告已保存: {report_path}")
            return str(report_path)
        except Exception as e:
            logger.error(f"保存解析报告失败: {e}")
            return None


# ============================================================
# 便捷函数
# ============================================================

def is_remote_parse_enabled() -> bool:
    """检查远程解析是否启用"""
    enabled = os.getenv("ENABLE_REMOTE_PARSE", "false").lower() == "true"
    has_token = bool(os.getenv("PARSE_API_TOKEN"))
    return enabled and has_token


def parse_document_remote(
    file_path_or_binary,
    from_page: int = 0,
    to_page: int = 100000
) -> Tuple[List[Tuple[str, str]], List[List[List[str]]]]:
    """使用远程 API 解析文档"""
    parser = RemoteDocumentParser()
    return parser.parse_document(file_path_or_binary, from_page, to_page)


def save_chunked_report(
    file_name: str,
    chunks: List,
    sections: List = None,
    tables: List = None,
    markdown_content: str = None
) -> Optional[str]:
    """保存带分块结果的解析报告"""
    parser = RemoteDocumentParser()
    return parser.save_chunked_report(file_name, chunks, sections, tables, markdown_content)


__all__ = [
    'RemoteDocumentParser',
    'parse_document_remote',
    'is_remote_parse_enabled',
    'save_chunked_report',
]
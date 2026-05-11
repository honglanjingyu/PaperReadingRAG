# app/service/core/deepdoc/parser/remote_pdf_parser.py
"""
远程文档解析器 - 使用 MinerU 云端 API 解析各种文档格式
支持: PDF, DOCX, TXT, Excel, PPT, HTML, Markdown 等
"""

import os
import re
import time
from io import BytesIO
from typing import List, Tuple, Optional, Dict, Any, Union
import tempfile
import logging
from datetime import datetime
from pathlib import Path
import requests
import zipfile
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class RemotePDFParser:
    """远程文档解析器 - 使用 MinerU API 解析各种文档"""

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

        self._client = None
        self._init_client()

        # 设置解析报告输出目录
        self._report_dir = self._get_report_dir()

        # 存储最后一次解析的结果
        self._last_parse_result = {
            "file_name": None,
            "markdown_content": None,
            "sections": None,
            "tables": None
        }

    def _get_report_dir(self) -> Path:
        """获取解析报告输出目录"""
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent.parent.parent
        report_dir = project_root / "docparselist"
        report_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"解析报告目录: {report_dir}")
        return report_dir

    def _init_client(self):
        """初始化客户端"""
        if not self.api_token:
            logger.warning("PARSE_API_TOKEN未配置，远程文档解析不可用")
            return

        self.base_url = "https://mineru.net"
        logger.info("远程文档解析器初始化成功（使用 REST API）")

    def is_available(self) -> bool:
        """检查远程解析器是否可用"""
        return self.api_token is not None and len(self.api_token) > 0

    def _upload_file(self, file_path: str) -> Optional[str]:
        """上传文件到 MinerU，返回 batch_id"""
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

            # 添加可选参数
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

            # 上传文件
            with open(file_path, 'rb') as f:
                upload_response = requests.put(file_url, data=f, timeout=60)

            if upload_response.status_code not in (200, 201):
                logger.error(f"文件上传失败: {upload_response.status_code}")
                return None

            logger.info(f"文件上传成功")
            return batch_id

        except Exception as e:
            logger.error(f"上传文件失败: {e}")
            return None

    def _wait_for_result(self, batch_id: str, timeout: int = 300) -> Optional[str]:
        """等待解析完成并返回结果ZIP URL"""
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
                    full_zip_url = file_result.get("full_zip_url")
                    logger.info(f"解析完成")
                    return full_zip_url
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
        """下载ZIP包并提取 full.md 内容"""
        try:
            response = requests.get(zip_url, timeout=60)
            if response.status_code != 200:
                logger.error(f"下载ZIP失败: {response.status_code}")
                return None

            zip_data = BytesIO(response.content)

            markdown_content = None
            with zipfile.ZipFile(zip_data, 'r') as zip_ref:
                logger.info(f"ZIP文件列表: {zip_ref.namelist()}")

                for file_name in zip_ref.namelist():
                    if file_name.endswith('full.md') or file_name.endswith('.md'):
                        with zip_ref.open(file_name) as md_file:
                            content = md_file.read().decode('utf-8')
                            logger.info(f"找到 Markdown 文件: {file_name}, 内容长度: {len(content)}")
                            return content

            logger.error(f"ZIP包中未找到 Markdown 文件，文件列表: {zip_ref.namelist()}")
            return None

        except Exception as e:
            logger.error(f"下载解析结果失败: {e}")
            return None

    def parse_document(
            self,
            file_path_or_binary: Union[str, bytes, BytesIO],
            from_page: int = 0,
            to_page: int = 100000,
            callback=None
    ) -> Tuple[List[Tuple[str, str]], List[List[List[str]]]]:
        """
        解析文档（支持所有格式）

        Args:
            file_path_or_binary: 文件路径或二进制数据
            from_page: 起始页（对 PDF 有效）
            to_page: 结束页（对 PDF 有效）
            callback: 回调函数

        Returns:
            (sections, tables): 段落列表和表格列表
        """
        if not self.is_available():
            logger.error("远程文档解析器不可用")
            return [], []

        temp_file = None
        original_file_name = None

        try:
            # 处理文件路径或二进制数据
            if isinstance(file_path_or_binary, (bytes, BytesIO)):
                if isinstance(file_path_or_binary, BytesIO):
                    binary_data = file_path_or_binary.getvalue()
                else:
                    binary_data = file_path_or_binary

                # 尝试检测文件类型
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
                return [], []

            file_size = os.path.getsize(file_path) / 1024
            logger.info(f"  正在调用 MinerU API 解析文档...")
            logger.info(f"  文件名: {original_file_name}")
            logger.info(f"  文件大小: {file_size:.2f} KB")

            # 步骤1：上传文件
            batch_id = self._upload_file(file_path)
            if not batch_id:
                logger.error("上传文件失败")
                return [], []

            # 步骤2：等待解析结果
            zip_url = self._wait_for_result(batch_id)
            if not zip_url:
                logger.error("等待解析结果失败")
                return [], []

            # 步骤3：下载并解析结果
            markdown_content = self._download_markdown(zip_url)
            if not markdown_content:
                logger.error("下载解析结果失败")
                return [], []

            logger.info(f"  ✓ API 调用成功，返回内容长度: {len(markdown_content)} 字符")
            logger.info(f"  Markdown 预览: {markdown_content[:200]}...")

            # 解析 Markdown 为段落和表格
            sections, tables = self.parse_markdown(markdown_content)

            # 将表格转换为文本段落
            table_sections = []
            for i, table in enumerate(tables):
                if table and len(table) > 0:
                    table_text = self._table_to_markdown(table)
                    if table_text:
                        table_sections.append((table_text, f"table_{i}"))

            all_sections = sections + table_sections

            logger.info(f"  解析完成: {len(sections)} 段落, {len(tables)} 表格")

            # 存储解析结果
            self._last_parse_result = {
                "file_name": original_file_name,
                "markdown_content": markdown_content,
                "sections": sections,
                "tables": tables
            }

            # 保存基础解析报告
            self._save_parse_report(
                file_name=original_file_name,
                markdown_content=markdown_content,
                sections=sections,
                tables=tables,
                chunks=None
            )

            return all_sections, tables

        except Exception as e:
            logger.error(f"远程文档解析失败: {e}")
            import traceback
            traceback.print_exc()
            return [], []
        finally:
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

    # 兼容旧方法名
    def parse_pdf(self, file_path_or_binary, from_page: int = 0, to_page: int = 100000, callback=None) -> Tuple[
        List[Tuple[str, str]], List[List[List[str]]]]:
        """解析 PDF（兼容旧方法名）"""
        return self.parse_document(file_path_or_binary, from_page, to_page, callback)

    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """将表格转换为 Markdown 格式的文本"""
        if not table or len(table) == 0:
            return ""

        lines = []
        # 表头
        header = "| " + " | ".join(str(cell) if cell else "" for cell in table[0]) + " |"
        lines.append(header)
        # 分隔线
        separator = "| " + " | ".join(["---"] * len(table[0])) + " |"
        lines.append(separator)
        # 数据行
        for row in table[1:]:
            line = "| " + " | ".join(str(cell) if cell else "" for cell in row) + " |"
            lines.append(line)

        return "\n".join(lines)

    def parse_markdown(self, markdown_content: str) -> Tuple[List[Tuple[str, str]], List[List[List[str]]]]:
        """
        将 MinerU 返回的 Markdown 解析为段落和表格

        Args:
            markdown_content: Markdown 内容

        Returns:
            (sections, tables): 段落列表和表格列表
        """
        if not markdown_content:
            return [], []

        lines = markdown_content.split('\n')
        total_lines = len(lines)

        logger.info(f"  开始解析 Markdown，共 {total_lines} 行")

        # 提取所有表格及其占用的行范围
        tables = []
        table_ranges = []

        i = 0
        while i < total_lines:
            line = lines[i].strip()

            # 检测 Markdown 表格（以 | 开头和结尾）
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
                    if table_data and len(table_data) > 0:
                        tables.append(table_data)
                        table_ranges.append((start, i, len(tables) - 1))
                        logger.info(f"    表格 {len(tables)}: {len(table_data)} 行 x {len(table_data[0])} 列")
                continue

            # 检测 HTML 表格
            elif '<table' in line.lower() or '<table>' in line.lower():
                start = i
                html_lines = []
                found_table_end = False
                table_tag_count = 0

                while i < total_lines:
                    current_line = lines[i]
                    html_lines.append(current_line)

                    if '<table' in current_line.lower() or '<thead' in current_line.lower() or '<tbody' in current_line.lower():
                        table_tag_count += 1
                    if '</table>' in current_line.lower():
                        table_tag_count -= 1
                        if table_tag_count == 0:
                            found_table_end = True
                            i += 1
                            break
                    i += 1

                    if len(html_lines) > 100:
                        break

                if found_table_end or len(html_lines) > 1:
                    html_content = '\n'.join(html_lines)
                    table_data = self._parse_html_table(html_content)
                    if table_data and len(table_data) > 0:
                        tables.append(table_data)
                        table_ranges.append((start, i, len(tables) - 1))
                        logger.info(f"    表格 {len(tables)}: {len(table_data)} 行 x {len(table_data[0])} 列")
                continue

            else:
                i += 1

        # 解析段落（跳过表格行）
        sections = []
        current_paragraph = []

        i = 0
        while i < total_lines:
            # 检查是否在表格范围内
            in_table = False
            for start, end, _ in table_ranges:
                if start <= i < end:
                    in_table = True
                    break

            if in_table:
                i += 1
                continue

            line = lines[i]
            stripped = line.strip()

            # 空行：结束当前段落
            if not stripped:
                if current_paragraph:
                    text = ' '.join(current_paragraph).strip()
                    if text:
                        sections.append((text, "paragraph"))
                    current_paragraph = []
                i += 1
                continue

            # 标题行（以 # 开头）
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

            # 普通文本行
            cleaned = self._clean_text(stripped)
            if cleaned:
                current_paragraph.append(cleaned)
            i += 1

        # 处理最后一段
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

            # 跳过分隔行
            if line_idx == 1 and all(self._is_separator(c) for c in cells):
                continue

            if not any(c for c in cells):
                continue

            cleaned_row = []
            for cell in cells:
                cleaned = self._clean_cell(cell)
                cleaned_row.append(cleaned)

            if cleaned_row:
                result.append(cleaned_row)

        # 确保每行列数一致
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

    def _parse_html_table(self, html_content: str) -> List[List[str]]:
        """解析 HTML 格式的表格"""
        if not html_content:
            return []

        result = []
        tr_pattern = r'<tr[^>]*>(.*?)</tr>'
        rows = re.findall(tr_pattern, html_content, re.DOTALL | re.IGNORECASE)

        for row_html in rows:
            cell_pattern = r'<t[dh][^>]*>(.*?)</t[dh]>'
            cells = re.findall(cell_pattern, row_html, re.DOTALL | re.IGNORECASE)

            if not cells:
                continue

            cleaned_row = []
            for cell_html in cells:
                text = re.sub(r'<[^>]+>', '', cell_html)
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()
                cleaned = self._clean_cell(text)
                cleaned_row.append(cleaned)

            if any(c for c in cleaned_row):
                result.append(cleaned_row)

        # 确保每行列数一致
        if result and len(result) > 1:
            max_cols = max(len(row) for row in result)
            for row in result:
                while len(row) < max_cols:
                    row.append("")

        return result

    def _clean_cell(self, text: str) -> str:
        """清理单元格内容"""
        if not text:
            return ""

        # 粗体
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        # 斜体
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        # 行内代码
        text = re.sub(r'`([^`]+)`', r'\1', text)
        # 链接
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # HTML 标签
        text = re.sub(r'<[^>]+>', '', text)
        # LaTeX 公式
        text = re.sub(r'\$[^$]+\$', '', text)
        # 标准化空白
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

    def _save_parse_report(
            self,
            file_name: str,
            markdown_content: str,
            sections: List[Tuple[str, str]],
            tables: List[List[List[str]]],
            chunks: List = None
    ) -> Optional[str]:
        """保存解析报告到文件"""
        if not file_name:
            return None

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(file_name).stem
        safe_base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
        report_name = f"{date_str}_{safe_base_name}.md"
        report_path = self._report_dir / report_name

        report_lines = []
        report_lines.append(f"# 文档解析报告")
        report_lines.append(f"")
        report_lines.append(f"**原始文件**: `{file_name}`")
        report_lines.append(f"**解析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**解析引擎**: MinerU API")
        report_lines.append(f"")
        report_lines.append(f"---")
        report_lines.append(f"")

        report_lines.append(f"## 📊 解析统计")
        report_lines.append(f"")
        report_lines.append(f"| 项目 | 数量 |")
        report_lines.append(f"|------|------|")
        report_lines.append(f"| 文字块数量 | {len(sections)} |")
        report_lines.append(f"| 表格数量 | {len(tables)} |")
        if chunks:
            report_lines.append(f"| 分块数量 | {len(chunks)} |")
        report_lines.append(f"| Markdown 长度 | {len(markdown_content)} 字符 |")
        report_lines.append(f"")
        report_lines.append(f"---")
        report_lines.append(f"")

        if sections:
            report_lines.append(f"## 📝 文字块详情 ({len(sections)} 个)")
            report_lines.append(f"")
            for idx, (text, style) in enumerate(sections[:50], 1):
                report_lines.append(f"### 块 {idx} (样式: {style})")
                report_lines.append(f"")
                display_text = text[:500] + "\n... (内容过长，已截断)" if len(text) > 500 else text
                report_lines.append(f"```")
                report_lines.append(display_text)
                report_lines.append(f"```")
                report_lines.append(f"")

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            logger.info(f"解析报告已保存: {report_path}")
            return str(report_path)
        except Exception as e:
            logger.error(f"保存解析报告失败: {e}")
            return None

    def save_chunked_report(
            self,
            file_name: str,
            chunks: List,
            sections: List[Tuple[str, str]] = None,
            tables: List[List[List[str]]] = None,
            markdown_content: str = None
    ) -> Optional[str]:
        """保存带分块结果的解析报告"""
        if sections is None:
            sections = self._last_parse_result.get("sections", [])
        if tables is None:
            tables = self._last_parse_result.get("tables", [])
        if markdown_content is None:
            markdown_content = self._last_parse_result.get("markdown_content", "")

        return self._save_parse_report(
            file_name=file_name,
            markdown_content=markdown_content or "",
            sections=sections or [],
            tables=tables or [],
            chunks=chunks
        )

    def get_last_parse_result(self) -> Dict[str, Any]:
        """获取最后一次解析的结果"""
        return self._last_parse_result.copy()


def is_remote_parse_enabled() -> bool:
    """检查远程解析是否启用"""
    enabled = os.getenv("ENABLE_REMOTE_PARSE", "false").lower() == "true"
    has_token = bool(os.getenv("PARSE_API_TOKEN"))
    return enabled and has_token


def parse_document_remote(
        file_path_or_binary,
        from_page: int = 0,
        to_page: int = 100000,
        api_token: str = None
) -> Tuple[List[Tuple[str, str]], List[List[List[str]]]]:
    """使用远程API解析文档"""
    parser = RemotePDFParser(api_token)
    return parser.parse_document(file_path_or_binary, from_page, to_page)


def save_chunked_report(
        file_name: str,
        chunks: List,
        sections: List[Tuple[str, str]] = None,
        tables: List[List[List[str]]] = None,
        markdown_content: str = None,
        api_token: str = None
) -> Optional[str]:
    """保存带分块结果的解析报告"""
    parser = RemotePDFParser(api_token)
    return parser.save_chunked_report(
        file_name=file_name,
        chunks=chunks,
        sections=sections,
        tables=tables,
        markdown_content=markdown_content
    )

def parse_pdf_remote(
        file_path_or_binary,
        from_page: int = 0,
        to_page: int = 100000,
        api_token: str = None
) -> Tuple[List[Tuple[str, str]], List[List[List[str]]]]:
    """使用远程API解析PDF（兼容旧接口）"""
    parser = RemotePDFParser(api_token)
    return parser.parse_document(file_path_or_binary, from_page, to_page)

__all__ = [
    'RemotePDFParser',
    'parse_document_remote',
    'parse_pdf_remote',  # 兼容旧名
    'is_remote_parse_enabled',
    'save_chunked_report'
]
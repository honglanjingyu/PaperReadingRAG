# app/service/core/deepdoc/parser/remote_pdf_parser.py
"""
远程PDF解析器 - 使用MinerU云端API解析PDF
将 MinerU 返回的 Markdown 正确解析为段落和表格
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
from io import BytesIO

logger = logging.getLogger(__name__)


class RemotePDFParser:
    """远程PDF解析器 - 使用MinerU API解析PDF"""

    def __init__(self, api_token: str = None):
        """初始化远程PDF解析器"""
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

        # 存储最后一次解析的结果（用于生成报告）
        self._last_parse_result = {
            "file_name": None,
            "markdown_content": None,
            "sections": None,
            "tables": None
        }

    def _get_report_dir(self) -> Path:
        """获取解析报告输出目录（run_api.py 同级的 docparselist 目录）"""
        # 获取项目根目录
        current_file = Path(__file__).resolve()
        # app/service/core/deepdoc/parser/ -> 项目根目录
        project_root = current_file.parent.parent.parent.parent.parent.parent
        report_dir = project_root / "docparselist"
        report_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"解析报告目录: {report_dir}")
        return report_dir

    def save_parse_report(
            self,
            file_name: str,
            markdown_content: str,
            sections: List[Tuple[str, str]],
            tables: List[List[List[str]]],
            chunks: List = None
    ) -> Optional[str]:
        """
        保存解析报告到文件

        Args:
            file_name: 原始文件名
            markdown_content: MinerU 返回的 Markdown 内容
            sections: 解析出的段落列表
            tables: 解析出的表格列表
            chunks: 分块后的结果（可选）

        Returns:
            保存的文件路径，失败返回 None
        """
        if not file_name:
            return None

        # 生成报告文件名: 日期+解析文件名.md
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(file_name).stem
        # 清理文件名中的非法字符
        safe_base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
        report_name = f"{date_str}_{safe_base_name}.md"
        report_path = self._report_dir / report_name

        # 构建报告内容
        report_lines = []

        # 标题
        report_lines.append(f"# PDF解析报告")
        report_lines.append(f"")
        report_lines.append(f"**原始文件**: `{file_name}`")
        report_lines.append(f"**解析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**解析引擎**: MinerU API")
        report_lines.append(f"")
        report_lines.append(f"---")
        report_lines.append(f"")

        # 统计信息
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

        # 文字块详情
        report_lines.append(f"## 📝 文字块详情 ({len(sections)} 个)")
        report_lines.append(f"")
        for idx, (text, style) in enumerate(sections, 1):
            report_lines.append(f"### 块 {idx} (样式: {style})")
            report_lines.append(f"")
            report_lines.append(f"```")
            # 限制长度，避免报告过大
            display_text = text[:1000] + "\n... (内容过长，已截断)" if len(text) > 1000 else text
            report_lines.append(display_text)
            report_lines.append(f"```")
            report_lines.append(f"")

        # 表格详情
        if tables:
            report_lines.append(f"## 📋 表格详情 ({len(tables)} 个)")
            report_lines.append(f"")
            for idx, table in enumerate(tables, 1):
                report_lines.append(f"### 表格 {idx}")
                report_lines.append(f"")
                if table and len(table) > 0:
                    # 转换为 Markdown 表格格式
                    if len(table) > 0:
                        # 表头
                        header = [str(cell) if cell else "" for cell in table[0]]
                        report_lines.append("| " + " | ".join(header) + " |")
                        report_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
                        # 数据行（最多显示 20 行）
                        for row in table[1:21]:
                            row_cells = [str(cell) if cell else "" for cell in row]
                            report_lines.append("| " + " | ".join(row_cells) + " |")
                        if len(table) > 21:
                            report_lines.append(f"| ... | (还有 {len(table) - 21} 行) |")
                    report_lines.append(f"")
                report_lines.append(f"")

        # 分块结果
        if chunks:
            report_lines.append(f"## 🔗 分块结果 ({len(chunks)} 个)")
            report_lines.append(f"")
            for idx, chunk in enumerate(chunks, 1):
                report_lines.append(f"### 分块 {idx}")
                report_lines.append(f"")
                # 显示 chunk 内容
                if hasattr(chunk, 'content'):
                    content = chunk.content
                elif isinstance(chunk, dict):
                    content = chunk.get('content', chunk.get('content_with_weight', str(chunk)))
                else:
                    content = str(chunk)

                display_content = content[:500] + "\n... (内容过长，已截断)" if len(content) > 500 else content
                report_lines.append(f"```")
                report_lines.append(display_content)
                report_lines.append(f"```")
                report_lines.append(f"")

                # 显示 token 数量（如果有）
                if hasattr(chunk, 'token_count') and chunk.token_count:
                    report_lines.append(f"*Token 数量: {chunk.token_count}*")
                    report_lines.append(f"")

        # 原始 Markdown 内容（可选，放在最后）
        report_lines.append(f"---")
        report_lines.append(f"")
        report_lines.append(f"## 📄 原始 Markdown 内容")
        report_lines.append(f"")
        report_lines.append(f"<details>")
        report_lines.append(f"<summary>点击展开</summary>")
        report_lines.append(f"")
        report_lines.append(f"```markdown")
        # 限制原始内容大小
        if len(markdown_content) > 50000:
            report_lines.append(markdown_content[:50000])
            report_lines.append(f"\n... (内容过长，已截断，共 {len(markdown_content)} 字符)")
        else:
            report_lines.append(markdown_content)
        report_lines.append(f"```")
        report_lines.append(f"")
        report_lines.append(f"</details>")

        # 写入文件
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
        """
        保存带分块结果的解析报告

        Args:
            file_name: 原始文件名
            chunks: 分块结果
            sections: 段落列表（可选）
            tables: 表格列表（可选）
            markdown_content: 原始 Markdown 内容（可选）

        Returns:
            保存的文件路径
        """
        # 使用存储的最后解析结果作为补充
        if sections is None:
            sections = self._last_parse_result.get("sections", [])
        if tables is None:
            tables = self._last_parse_result.get("tables", [])
        if markdown_content is None:
            markdown_content = self._last_parse_result.get("markdown_content", "")

        return self.save_parse_report(
            file_name=file_name,
            markdown_content=markdown_content or "",
            sections=sections or [],
            tables=tables or [],
            chunks=chunks
        )

    def _init_client(self):
        """初始化MinerU客户端（不再使用 mineru 库）"""
        if not self.api_token:
            logger.warning("PARSE_API_TOKEN未配置，远程PDF解析不可用")
            return

        # 不再初始化 mineru 客户端，改为使用 requests
        self.base_url = "https://mineru.net"
        logger.info("远程PDF解析器初始化成功（使用 REST API）")

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
                # 打印ZIP中的所有文件，用于调试
                logger.info(f"ZIP文件列表: {zip_ref.namelist()}")

                for file_name in zip_ref.namelist():
                    # 匹配 full.md 或任何 .md 文件
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

    def parse_pdf(
            self,
            file_path_or_binary,
            from_page: int = 0,
            to_page: int = 100000,
            callback=None
    ) -> Tuple[List[Tuple[str, str]], List[List[List[str]]]]:
        """
        解析PDF文件（使用 MinerU REST API）
        """
        if not self.is_available():
            logger.error("远程PDF解析器不可用")
            return [], []

        temp_file = None
        original_file_name = None

        try:
            # 处理文件路径
            if isinstance(file_path_or_binary, (bytes, BytesIO)):
                if isinstance(file_path_or_binary, BytesIO):
                    binary_data = file_path_or_binary.getvalue()
                else:
                    binary_data = file_path_or_binary

                original_file_name = "uploaded_file.pdf"
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

            logger.info(f"  正在调用 MinerU API...")
            logger.info(f"  文件大小: {os.path.getsize(file_path) / 1024:.2f} KB")
            logger.info(f"  配置参数: model_version={self.model_version}, enable_table={self.enable_table}")

            # 步骤1：申请上传URL
            batch_id = self._upload_file(file_path)
            if not batch_id:
                logger.error("申请上传URL失败")
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

            # 调试日志：打印前500字符
            logger.info(f"  Markdown 预览: {markdown_content[:200]}...")

            # 解析 Markdown 为段落和表格
            sections, tables = self.parse_markdown(markdown_content)
            table_sections = []
            for i, table in enumerate(tables):
                if table and len(table) > 0:
                    # 将表格转换为 Markdown 格式文本
                    table_text = self._table_to_markdown(table)
                    if table_text:
                        table_sections.append((table_text, f"table_{i}"))

            all_sections = sections + table_sections

            logger.info(f"  解析完成: {len(sections)}段落, {len(tables)}表格")

            # 调试日志：打印表格数量
            logger.info(f"  表格详情: {tables}")

            # 存储解析结果
            self._last_parse_result = {
                "file_name": original_file_name,
                "markdown_content": markdown_content,  # 确保这里保存了完整内容
                "sections": sections,
                "tables": tables
            }

            # 保存基础解析报告（确保 markdown_content 不为空）
            self.save_parse_report(
                file_name=original_file_name,
                markdown_content=markdown_content,  # 确保传递完整内容
                sections=sections,
                tables=tables,
                chunks=None
            )

            return all_sections, tables

        except Exception as e:
            logger.error(f"远程PDF解析失败: {e}")
            import traceback
            traceback.print_exc()
            return [], []
        finally:
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """将表格转换为 Markdown 格式的文本"""
        if not table or len(table) == 0:
            return ""

        lines = []
        # 表头
        header = "| " + " | ".join(str(cell) for cell in table[0]) + " |"
        lines.append(header)
        # 分隔线
        separator = "| " + " | ".join(["---"] * len(table[0])) + " |"
        lines.append(separator)
        # 数据行
        for row in table[1:]:
            line = "| " + " | ".join(str(cell) for cell in row) + " |"
            lines.append(line)

        return "\n".join(lines)

    def parse_markdown(self, markdown_content: str) -> Tuple[List[Tuple[str, str]], List[List[List[str]]]]:
        """
        将 MinerU 返回的 Markdown 解析为段落和表格
        """
        if not markdown_content:
            return [], []

        lines = markdown_content.split('\n')
        total_lines = len(lines)

        logger.info(f"  开始解析 Markdown，共 {total_lines} 行")

        # 第一步：提取所有表格及其占用的行范围
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

            # ========== 修复：检测 HTML 表格（不区分大小写，检测整个行内容） ==========
            elif '<table' in line.lower() or '</table>' in line.lower():
                start = i
                html_lines = []
                found_table_end = False
                table_tag_count = 0

                # 收集从 <table 到 <table> 的所有行
                while i < total_lines:
                    current_line = lines[i]
                    html_lines.append(current_line)

                    # 统计表格标签
                    if '<table' in current_line.lower() or '<thead' in current_line.lower() or '<tbody' in current_line.lower():
                        table_tag_count += 1
                    if '</table>' in current_line.lower():
                        table_tag_count -= 1
                        if table_tag_count == 0:
                            found_table_end = True
                            i += 1
                            break
                    i += 1

                    # 防止无限循环，最多收集 100 行
                    if len(html_lines) > 100:
                        break

                # 如果找到了表格结束标签，或者至少收集到了内容
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

        # 第二步：解析段落（跳过表格行）
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
        """
        解析 Markdown 格式的表格

        Args:
            table_lines: Markdown 表格行，如：
                "| 列1 | 列2 | 列3 |"
                "| --- | --- | --- |"
                "| 值1 | 值2 | 值3 |"

        Returns:
            二维列表
        """
        if not table_lines or len(table_lines) < 2:
            return []

        result = []

        for line_idx, line in enumerate(table_lines):
            # 分割单元格
            cells = line.split('|')
            # 去掉首尾空元素（因为行以 | 开头和结尾）
            cells = cells[1:-1]
            # 清理每个单元格
            cells = [c.strip() for c in cells]

            # 跳过分隔行（包含 --- 或 :--- 的行）
            if line_idx == 1 and all(self._is_separator(c) for c in cells):
                continue

            # 过滤掉全空的行
            if not any(c for c in cells):
                continue

            # 清理单元格内容
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
        # 移除空白和冒号
        cleaned = cell.replace(' ', '').replace(':', '')
        return all(c == '-' for c in cleaned)

    def _parse_html_table(self, html_content: str) -> List[List[str]]:
        """
        解析 HTML 格式的表格
        """
        if not html_content:
            return []

        result = []

        # 方法1：使用正则提取所有行（保持原有逻辑）
        tr_pattern = r'<tr[^>]*>(.*?)</tr>'
        rows = re.findall(tr_pattern, html_content, re.DOTALL | re.IGNORECASE)

        for row_html in rows:
            # 提取所有单元格（td 和 th）
            cell_pattern = r'<t[dh][^>]*>(.*?)</t[dh]>'
            cells = re.findall(cell_pattern, row_html, re.DOTALL | re.IGNORECASE)

            if not cells:
                continue

            cleaned_row = []
            for cell_html in cells:
                # 移除内部 HTML 标签
                text = re.sub(r'<[^>]+>', '', cell_html)
                # 清理空白
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()
                cleaned = self._clean_cell(text)
                cleaned_row.append(cleaned)

            if any(c for c in cleaned_row):
                result.append(cleaned_row)

        # 方法2：如果方法1没有解析出结果，尝试按单元格直接提取
        if not result:
            # 提取所有单元格
            all_cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', html_content, re.DOTALL | re.IGNORECASE)
            if all_cells:
                # 尝试确定列数（通过查找第一行）
                # 简单处理：如果找到的单元格数量 > 0，按每行假设的列数分组
                # 这里先返回原始单元格列表
                logger.warning(f"未找到完整的表格行结构，直接返回 {len(all_cells)} 个单元格")
                # 将所有单元格作为单行返回
                cleaned_row = []
                for cell in all_cells:
                    text = re.sub(r'<[^>]+>', '', cell)
                    text = re.sub(r'\s+', ' ', text).strip()
                    cleaned_row.append(self._clean_cell(text))
                if cleaned_row:
                    result.append(cleaned_row)

        # 确保每行列数一致
        if result and len(result) > 1:
            max_cols = max(len(row) for row in result)
            for row in result:
                while len(row) < max_cols:
                    row.append("")

        return result

    def _clean_cell(self, text: str) -> str:
        """
        清理单元格内容，移除 Markdown 格式标记

        Args:
            text: 原始单元格文本

        Returns:
            清理后的文本
        """
        if not text:
            return ""

        # 粗体 **text** 或 __text__
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)

        # 斜体 *text* 或 _text_
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)

        # 行内代码 `code`
        text = re.sub(r'`([^`]+)`', r'\1', text)

        # 链接 [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # 移除 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)

        # 移除 LaTeX 公式 $...$
        text = re.sub(r'\$[^$]+\$', '', text)

        # 移除特殊符号
        text = text.replace('\\', '').replace('*', '').replace('_', '')

        # 标准化空白
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # 限制长度
        if len(text) > 2000:
            text = text[:2000] + "..."

        return text

    def _clean_text(self, text: str) -> str:
        """
        清理普通文本，移除 Markdown 格式标记

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        if not text:
            return ""

        # 移除标题标记
        if text.startswith('#'):
            text = re.sub(r'^#+\s*', '', text)

        # 移除粗体
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)

        # 移除斜体
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)

        # 移除行内代码
        text = re.sub(r'`([^`]+)`', r'\1', text)

        # 移除链接
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # 移除 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)

        # 标准化空白
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def get_last_parse_result(self) -> Dict[str, Any]:
        """获取最后一次解析的结果"""
        return self._last_parse_result.copy()


def is_remote_parse_enabled() -> bool:
    """检查远程解析是否启用"""
    enabled = os.getenv("ENABLE_REMOTE_PARSE", "false").lower() == "true"
    has_token = bool(os.getenv("PARSE_API_TOKEN"))
    return enabled and has_token


def parse_pdf_remote(
        file_path_or_binary,
        from_page: int = 0,
        to_page: int = 100000,
        api_token: str = None
) -> Tuple[List[Tuple[str, str]], List[List[List[str]]]]:
    """使用远程API解析PDF"""
    parser = RemotePDFParser(api_token)
    return parser.parse_pdf(file_path_or_binary, from_page, to_page)


def save_chunked_report(
        file_name: str,
        chunks: List,
        sections: List[Tuple[str, str]] = None,
        tables: List[List[List[str]]] = None,
        markdown_content: str = None,
        api_token: str = None
) -> Optional[str]:
    """
    保存带分块结果的解析报告（便捷函数）

    Args:
        file_name: 原始文件名
        chunks: 分块结果
        sections: 段落列表（可选）
        tables: 表格列表（可选）
        markdown_content: 原始 Markdown 内容（可选）
        api_token: API Token（可选）

    Returns:
        保存的文件路径
    """
    parser = RemotePDFParser(api_token)
    return parser.save_chunked_report(
        file_name=file_name,
        chunks=chunks,
        sections=sections,
        tables=tables,
        markdown_content=markdown_content
    )


__all__ = [
    'RemotePDFParser',
    'parse_pdf_remote',
    'is_remote_parse_enabled',
    'save_chunked_report'
]
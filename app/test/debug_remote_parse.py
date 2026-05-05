import os
from dotenv import load_dotenv

load_dotenv()


def debug_raw_markdown():
    print("=" * 60)
    print("查看 MinerU 原始返回内容")
    print("=" * 60)

    from mineru import MinerU

    # 设置环境变量
    os.environ["MINERU_TOKEN"] = os.getenv("PARSE_API_TOKEN")

    # 初始化客户端
    client = MinerU()

    # 解析 PDF
    pdf_file = "../../【兴证电子】世运电路2023中报点评2.pdf"

    print(f"\n解析: {pdf_file}")
    result = client.extract(pdf_file)

    # 查看 result 对象的属性
    print(f"\nresult 类型: {type(result)}")
    print(f"result 属性: {[attr for attr in dir(result) if not attr.startswith('_')]}")

    # 获取 markdown 内容
    if hasattr(result, 'markdown'):
        content = result.markdown
        print(f"\nMarkdown 内容长度: {len(content)} 字符")

        # 保存完整内容到文件
        with open("debug_raw_markdown.md", "w", encoding="utf-8") as f:
            f.write(content)
        print("完整内容已保存到: debug_raw_markdown.md")

        # 打印前 2000 字符
        print("\n" + "=" * 60)
        print("前 2000 字符:")
        print("=" * 60)
        print(content[:2000])

        # 检查是否包含表格标记
        print("\n" + "=" * 60)
        print("检查表格标记:")
        print("=" * 60)
        print(f"  包含 '|': {'|' in content}")
        print(f"  包含 '<td>': {'</td>' in content}")

        # 修复：不能直接在 f-string 中使用反斜杠
        has_tabular = '\\begin{tabular}' in content
        print(f"  包含 '\\begin{{tabular}}': {has_tabular}")

        # 统计行数
        lines = content.split('\n')
        print(f"\n总行数: {len(lines)}")

        # 打印所有非空行（调试）
        print("\n所有非空行内容:")
        for i, line in enumerate(lines):
            if line.strip():
                # 限制每行显示长度
                display_line = line[:150] + "..." if len(line) > 150 else line
                print(f"  {i + 1}: {display_line}")

    else:
        print("result 对象没有 markdown 属性")


if __name__ == "__main__":
    debug_raw_markdown()
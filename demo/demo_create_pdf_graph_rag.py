# scripts/create_graphrag_test_pdfs.py
"""创建三个两页的 GraphRAG 测试PDF文档"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import hashlib


def register_chinese_font():
    """注册中文字体 - WSL Ubuntu版本"""
    font_paths = [
        "/mnt/c/Windows/Fonts/simhei.ttf",
        "/mnt/c/Windows/Fonts/simsun.ttc",
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                print(f"成功加载字体: {font_path}")
                return 'ChineseFont'
            except Exception as e:
                print(f"加载字体失败 {font_path}: {e}")
                continue

    print("警告：未找到中文字体，PDF中文可能显示异常")
    print("建议运行: sudo apt install fonts-wqy-microhei fonts-noto-cjk")
    return 'Helvetica'


def create_pdf1_tech_company(output_path):
    """PDF1: 科技公司财报 - 测试实体: 公司名、人物、财务数据、产品线"""

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm
    )

    font_name = register_chinese_font()
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title', parent=styles['Heading1'], fontName=font_name,
        fontSize=20, textColor=colors.HexColor('#1a1a2e'), alignment=TA_CENTER, spaceAfter=15
    )
    chapter_style = ParagraphStyle(
        'Chapter', parent=styles['Heading2'], fontName=font_name,
        fontSize=14, textColor=colors.HexColor('#667eea'), spaceBefore=15, spaceAfter=8
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'], fontName=font_name,
        fontSize=10, leading=16, alignment=TA_JUSTIFY, spaceAfter=6
    )
    entity_style = ParagraphStyle(
        'Entity', parent=styles['Normal'], fontName=font_name,
        fontSize=9, textColor=colors.HexColor('#28a745'), alignment=TA_LEFT, leftIndent=10
    )

    story = []

    # ========== 第1页 ==========
    story.append(Paragraph("云创科技 2024年第一季度财报", title_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("一、公司概况", chapter_style))
    story.append(Paragraph(
        "云创科技股份有限公司（股票代码：688888）成立于2015年，总部位于深圳市南山区科技园。"
        "公司董事长兼CEO为张伟先生，CTO为李华博士，CFO为王芳女士。公司员工总数超过3000人，"
        "其中研发人员占比45%。2024年第一季度，公司实现营业收入12.5亿元，同比增长35.2%；"
        "净利润2.3亿元，同比增长42.8%。",
        body_style
    ))

    story.append(Paragraph("二、主营业务收入构成", chapter_style))

    revenue_data = [
        ['业务板块', '收入(亿元)', '同比增长', '毛利率'],
        ['云计算服务', '5.8', '45%', '52%'],
        ['人工智能平台', '3.2', '68%', '65%'],
        ['企业软件', '2.1', '12%', '78%'],
        ['硬件设备', '1.4', '-5%', '22%'],
    ]

    revenue_table = Table(revenue_data, colWidths=[80, 70, 60, 60])
    revenue_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
    ]))
    story.append(revenue_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("三、主要客户", chapter_style))
    story.append(Paragraph(
        "前五大客户分别为：中国移动通信集团有限公司（收入占比18%）、"
        "平安银行股份有限公司（12%）、华为技术有限公司（10%）、"
        "腾讯科技（深圳）有限公司（8%）、比亚迪股份有限公司（6%）。",
        body_style
    ))

    # ========== 第2页 ==========
    story.append(PageBreak())

    story.append(Paragraph("四、研发投入与新产品", chapter_style))
    story.append(Paragraph(
        "2024年Q1研发投入2.8亿元，占营业收入22.4%。主要研发项目包括：\n"
        "• 星火大模型：参数规模达到1000亿，预计2024年Q3发布\n"
        "• 智能客服系统：已服务超过500家企业客户\n"
        "• 数据中台产品：签约客户包括招商银行、顺丰速运\n"
        "• AI芯片设计项目：与寒武纪联合研发，预计2025年量产",
        body_style
    ))

    story.append(Paragraph("五、关键实体信息提取（GraphRAG测试点）", chapter_style))
    story.append(Paragraph(
        "【实体列表】\n"
        "PERSON: 张伟(CEO), 李华(CTO), 王芳(CFO)\n"
        "ORGANIZATION: 云创科技, 华为, 腾讯, 比亚迪, 中国移动, 平安银行, 招商银行, 顺丰速运, 寒武纪\n"
        "PRODUCT: 星火大模型, 智能客服系统, 数据中台, AI芯片\n"
        "LOCATION: 深圳市南山区科技园\n"
        "DATE: 2015年(成立), 2024年Q3(发布), 2025年(量产)\n"
        "NUMBER: 12.5亿元(营收), 2.3亿元(净利润), 22.4%(研发占比)",
        entity_style
    ))

    story.append(Spacer(1, 15))
    story.append(Paragraph("六、关系图谱（GraphRAG测试）", chapter_style))
    story.append(Paragraph(
        "【关系列表】\n"
        "张伟(CEO) — WORKS_FOR → 云创科技\n"
        "李华(CTO) — WORKS_FOR → 云创科技\n"
        "云创科技 — COOPERATES_WITH → 华为, 腾讯, 比亚迪, 寒武纪\n"
        "云创科技 — SELLS_TO → 中国移动, 平安银行, 招商银行, 顺丰速运\n"
        "云创科技 — LOCATED_IN → 深圳市南山区科技园\n"
        "云创科技 — PRODUCES → 星火大模型, 智能客服系统, 数据中台, AI芯片\n"
        "星火大模型 — RELATED_TO → 人工智能平台\n"
        "AI芯片 — PARTNER_WITH → 寒武纪",
        entity_style
    ))

    doc.build(story)
    print(f"✓ 已创建: {output_path}")


def create_pdf2_industry_report(output_path):
    """PDF2: 人工智能行业报告 - 测试实体: 行业术语、技术概念、市场分析"""

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm
    )

    font_name = register_chinese_font()
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title', parent=styles['Heading1'], fontName=font_name,
        fontSize=20, textColor=colors.HexColor('#1a1a2e'), alignment=TA_CENTER, spaceAfter=15
    )
    chapter_style = ParagraphStyle(
        'Chapter', parent=styles['Heading2'], fontName=font_name,
        fontSize=14, textColor=colors.HexColor('#2196f3'), spaceBefore=15, spaceAfter=8
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'], fontName=font_name,
        fontSize=10, leading=16, alignment=TA_JUSTIFY, spaceAfter=6
    )
    entity_style = ParagraphStyle(
        'Entity', parent=styles['Normal'], fontName=font_name,
        fontSize=9, textColor=colors.HexColor('#2196f3'), alignment=TA_LEFT, leftIndent=10
    )

    story = []

    # ========== 第1页 ==========
    story.append(Paragraph("2024年人工智能行业发展趋势报告", title_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("一、行业概述", chapter_style))
    story.append(Paragraph(
        "2024年全球人工智能市场规模预计达到1800亿美元，同比增长28%。"
        "中国市场表现尤为突出，规模达到4500亿元人民币，同比增长32%。"
        "大语言模型（LLM）成为行业热点，以GPT-4、Claude、Gemini、通义千问、文心一言为代表的大模型产品"
        "正在重塑各行各业的智能化进程。RAG（检索增强生成）技术作为大模型落地的关键技术，"
        "能够有效解决大模型幻觉问题和知识更新问题。",
        body_style
    ))

    story.append(Paragraph("二、技术趋势对比", chapter_style))

    tech_data = [
        ['技术方向', '2023年成熟度', '2024年成熟度', '主要厂商', '应用场景'],
        ['大语言模型(LLM)', '发展期', '成熟期', 'OpenAI, Google, Anthropic, 阿里, 百度', '对话, 写作, 编程'],
        ['RAG技术', '探索期', '发展期', 'LangChain, LlamaIndex, 智谱, 百川', '企业知识库, 问答系统'],
        ['多模态模型', '早期', '发展期', 'OpenAI, Google, Meta', '图像生成, 视频理解'],
        ['Agent智能体', '概念期', '探索期', 'AutoGPT, BabyAGI', '自动化任务, 决策辅助'],
    ]

    tech_table = Table(tech_data, colWidths=[70, 60, 60, 100, 80])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2196f3')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
    ]))
    story.append(tech_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("三、头部企业竞争格局", chapter_style))
    story.append(Paragraph(
        "国内大模型厂商竞争激烈，主要包括：\n"
        "• 百度（文心一言）：背靠百度搜索生态，用户规模最大\n"
        "• 阿里巴巴（通义千问）：依托阿里云基础设施，企业客户最多\n"
        "• 腾讯（混元大模型）：整合微信生态，C端优势明显\n"
        "• 字节跳动（豆包）：短视频场景深度融合\n"
        "• 智谱AI（ChatGLM）：开源生态完善，开发者社区活跃\n"
        "• 百川智能（百川大模型）：王小川创办，融资速度最快",
        body_style
    ))

    # ========== 第2页 ==========
    story.append(PageBreak())

    story.append(Paragraph("四、投资与融资动态", chapter_style))
    story.append(Paragraph(
        "2024年上半年AI领域主要融资事件：\n"
        "1. 月之暗面（Moonshot AI）：完成10亿美元融资，估值25亿美元，投资方包括红杉资本、高瓴创投\n"
        "2. 百川智能：完成3亿美元A轮融资，估值12亿美元，阿里、腾讯参投\n"
        "3. 智谱AI：完成4亿美元B轮融资，估值20亿美元\n"
        "4. 零一万物：完成2亿美元天使轮融资，李开复创办\n"
        "5. 深度求索（DeepSeek）：完成1亿美元融资，幻方量化孵化",
        body_style
    ))

    story.append(Paragraph("五、关键实体信息提取（GraphRAG测试点）", chapter_style))
    story.append(Paragraph(
        "【实体列表】\n"
        "CONCEPT: 大语言模型(LLM), RAG技术, 多模态模型, Agent智能体\n"
        "ORGANIZATION: OpenAI, Google, Anthropic, 百度, 阿里巴巴, 腾讯, 字节跳动, 智谱AI, 百川智能, 月之暗面, 零一万物, 深度求索\n"
        "PRODUCT: GPT-4, Claude, Gemini, 通义千问, 文心一言, 混元大模型, 豆包, ChatGLM, 百川大模型\n"
        "PERSON: 王小川(百川智能创始人), 李开复(零一万物创始人)\n"
        "INVESTOR: 红杉资本, 高瓴创投, 阿里, 腾讯, 幻方量化\n"
        "DATE: 2024年\n"
        "NUMBER: 1800亿美元(全球市场), 4500亿元(中国市场), 32%(增长率)",
        entity_style
    ))

    story.append(Spacer(1, 15))
    story.append(Paragraph("六、关系图谱（GraphRAG测试）", chapter_style))
    story.append(Paragraph(
        "【关系列表】\n"
        "LLM — RELATED_TO → RAG技术, 多模态模型, Agent智能体\n"
        "百度 — PRODUCES → 文心一言\n"
        "阿里巴巴 — PRODUCES → 通义千问\n"
        "腾讯 — PRODUCES → 混元大模型\n"
        "字节跳动 — PRODUCES → 豆包\n"
        "智谱AI — PRODUCES → ChatGLM\n"
        "百川智能 — FOUNDED_BY → 王小川\n"
        "零一万物 — FOUNDED_BY → 李开复\n"
        "红杉资本 — INVESTED_IN → 月之暗面\n"
        "高瓴创投 — INVESTED_IN → 月之暗面\n"
        "阿里 — INVESTED_IN → 百川智能\n"
        "腾讯 — INVESTED_IN → 百川智能\n"
        "幻方量化 — INCUBATED → 深度求索",
        entity_style
    ))

    doc.build(story)
    print(f"✓ 已创建: {output_path}")


def create_pdf3_research_collaboration(output_path):
    """PDF3: 产学研合作报告 - 测试实体: 高校、实验室、合作项目、发表论文"""

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm
    )

    font_name = register_chinese_font()
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title', parent=styles['Heading1'], fontName=font_name,
        fontSize=20, textColor=colors.HexColor('#1a1a2e'), alignment=TA_CENTER, spaceAfter=15
    )
    chapter_style = ParagraphStyle(
        'Chapter', parent=styles['Heading2'], fontName=font_name,
        fontSize=14, textColor=colors.HexColor('#ff9800'), spaceBefore=15, spaceAfter=8
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'], fontName=font_name,
        fontSize=10, leading=16, alignment=TA_JUSTIFY, spaceAfter=6
    )
    entity_style = ParagraphStyle(
        'Entity', parent=styles['Normal'], fontName=font_name,
        fontSize=9, textColor=colors.HexColor('#ff9800'), alignment=TA_LEFT, leftIndent=10
    )

    story = []

    # ========== 第1页 ==========
    story.append(Paragraph("产学研合作与科研成果年度报告", title_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("一、合作高校与实验室", chapter_style))
    story.append(Paragraph(
        "2024年度，公司与以下高校建立了战略合作关系：\n"
        "• 清华大学人工智能研究院：联合成立「清华-云创智能计算联合实验室」\n"
        "• 北京大学信息科学技术学院：共建「数据科学与智能系统研究中心」\n"
        "• 上海交通大学计算机系：合作开发分布式训练框架\n"
        "• 浙江大学计算机学院：联合培养AI博士项目\n"
        "• 中国科学技术大学：共建认知智能联合实验室\n"
        "• 哈尔滨工业大学：自然语言处理方向深度合作",
        body_style
    ))

    story.append(Paragraph("二、重点合作项目", chapter_style))

    project_data = [
        ['项目名称', '合作单位', '负责人', '经费(万元)', '周期'],
        ['大模型高效训练算法', '清华大学', '张教授', '500', '2024-2026'],
        ['多模态知识图谱构建', '北京大学', '李教授', '400', '2024-2025'],
        ['低资源语言NLP技术', '哈工大', '刘教授', '350', '2024-2025'],
        ['联邦学习隐私保护', '上海交大', '王教授', '300', '2024-2026'],
    ]

    project_table = Table(project_data, colWidths=[100, 70, 60, 60, 60])
    project_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff9800')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
    ]))
    story.append(project_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("三、联合发表论文", chapter_style))
    story.append(Paragraph(
        "2024年联合发表的代表性论文：\n"
        "1. 'Efficient Fine-tuning of Large Language Models with Adapter Fusion' - 张伟等, ACL 2024\n"
        "2. 'Knowledge Graph Enhanced Retrieval for Question Answering' - 李明等, EMNLP 2024\n"
        "3. 'Cross-modal Representation Learning for Vision-Language Tasks' - 王华等, CVPR 2024\n"
        "4. 'Privacy-preserving Federated Learning for Vertical Data Partitioning' - 刘强等, NeurIPS 2024",
        body_style
    ))

    # ========== 第2页 ==========
    story.append(PageBreak())

    story.append(Paragraph("四、人才交流与培养", chapter_style))
    story.append(Paragraph(
        "双方人才交流计划：\n"
        "• 联合培养博士生：目前在读12人，2024年计划新增6人\n"
        "• 企业导师制度：公司派出10名高级工程师担任企业导师\n"
        "• 学术讲座系列：2024年举办20场前沿技术讲座\n"
        "• 博士后工作站：目前在站博士后8人，出站后留用5人",
        body_style
    ))

    story.append(Paragraph("五、科研成果转化", chapter_style))
    story.append(Paragraph(
        "已落地的科研成果转化项目：\n"
        "1. 智能客服语义理解模块（源自清华联合实验室）\n"
        "2. 知识图谱自动构建工具（源自北大项目）\n"
        "3. 低资源语言机器翻译系统（源自哈工大合作）\n"
        "4. 多模态内容审核平台（源自上海交大项目）",
        body_style
    ))

    story.append(Paragraph("六、关键实体信息提取（GraphRAG测试点）", chapter_style))
    story.append(Paragraph(
        "【实体列表】\n"
        "ORGANIZATION: 清华大学人工智能研究院, 北京大学信息科学技术学院, 上海交通大学计算机系, 浙江大学计算机学院, 中国科学技术大学, 哈尔滨工业大学\n"
        "LAB: 清华-云创智能计算联合实验室, 数据科学与智能系统研究中心, 认知智能联合实验室\n"
        "PERSON: 张教授, 李教授, 刘教授, 王教授, 张伟, 李明, 王华, 刘强\n"
        "PROJECT: 大模型高效训练算法, 多模态知识图谱构建, 低资源语言NLP技术, 联邦学习隐私保护\n"
        "PAPER: ACL 2024, EMNLP 2024, CVPR 2024, NeurIPS 2024\n"
        "DATE: 2024年",
        entity_style
    ))

    story.append(Spacer(1, 15))
    story.append(Paragraph("七、关系图谱（GraphRAG测试）", chapter_style))
    story.append(Paragraph(
        "【关系列表】\n"
        "清华大学 — COLLABORATES_WITH → 云创科技\n"
        "北京大学 — COLLABORATES_WITH → 云创科技\n"
        "张教授 — WORKS_FOR → 清华大学\n"
        "张教授 — LEADS → 大模型高效训练算法\n"
        "李教授 — WORKS_FOR → 北京大学\n"
        "李教授 — LEADS → 多模态知识图谱构建\n"
        "清华-云创联合实验室 — PART_OF → 清华大学, 云创科技\n"
        "ACL 2024 — PUBLISHED_BY → 张伟\n"
        "EMNLP 2024 — PUBLISHED_BY → 李明\n"
        "智能客服模块 — DERIVED_FROM → 清华-云创联合实验室",
        entity_style
    ))

    doc.build(story)
    print(f"✓ 已创建: {output_path}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("创建 GraphRAG 测试 PDF 文档")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "uploads_tmp"
    output_dir.mkdir(exist_ok=True)

    print(f"\n输出目录: {output_dir}")
    print("\n正在创建三个两页的PDF文档...\n")

    create_pdf1_tech_company(output_dir / "科技公司财报.pdf")
    create_pdf2_industry_report(output_dir / "人工智能行业报告.pdf")
    create_pdf3_research_collaboration(output_dir / "产学研合作报告.pdf")

    print("\n" + "=" * 60)
    print("所有PDF文档创建完成！共3个文件，每个2页")
    print("=" * 60)

    print("\n📄 GraphRAG 测试点说明：")
    print("  1. 科技公司财报.pdf")
    print("     - 实体类型：公司、人物、产品、财务数据")
    print("     - 关系类型：WORKS_FOR, PRODUCES, SELLS_TO, LOCATED_IN")
    print("")
    print("  2. 人工智能行业报告.pdf")
    print("     - 实体类型：概念、公司、产品、人物、投资机构")
    print("     - 关系类型：PRODUCES, FOUNDED_BY, INVESTED_IN, RELATED_TO")
    print("")
    print("  3. 产学研合作报告.pdf")
    print("     - 实体类型：高校、实验室、论文、项目、人物")
    print("     - 关系类型：COLLABORATES_WITH, LEADS, PART_OF, PUBLISHED_BY")
    print("=" * 60)


if __name__ == "__main__":
    main()
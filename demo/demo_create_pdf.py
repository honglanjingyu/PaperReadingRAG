# scripts/create_test_pdfs.py
"""创建三个不同等级的测试PDF文档"""

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
        # WSL Ubuntu 可访问的 Windows 字体路径
        "/mnt/c/Windows/Fonts/simhei.ttf",  # 黑体
        "/mnt/c/Windows/Fonts/simsun.ttc",  # 宋体
        "/mnt/c/Windows/Fonts/msyh.ttc",  # 微软雅黑
        # Ubuntu 系统自带字体
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # 文泉驿微米黑
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Google Noto字体
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

    # 如果所有字体都加载失败，使用内置字体并给出警告
    print("警告：未找到中文字体，PDF中文可能显示异常")
    print("建议运行: sudo apt install fonts-wqy-microhei fonts-noto-cjk")
    return 'Helvetica'


def create_normal_user_pdf(output_path):
    """创建普通用户级别的PDF - 公司简介、公告等公开信息"""

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

    # 自定义样式
    title_style = ParagraphStyle(
        'Title', parent=styles['Heading1'], fontName=font_name,
        fontSize=22, textColor=colors.HexColor('#1a1a2e'), alignment=TA_CENTER, spaceAfter=20
    )
    chapter_style = ParagraphStyle(
        'Chapter', parent=styles['Heading2'], fontName=font_name,
        fontSize=16, textColor=colors.HexColor('#16213e'), spaceBefore=15, spaceAfter=10
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'], fontName=font_name,
        fontSize=11, leading=18, alignment=TA_JUSTIFY, spaceAfter=8
    )
    level_badge = ParagraphStyle(
        'LevelBadge', parent=styles['Normal'], fontName=font_name,
        fontSize=10, textColor=colors.HexColor('#28a745'), alignment=TA_CENTER
    )

    story = []

    # ========== 第1页 ==========

    story.append(Paragraph("🔓 文档等级：普通用户 (可公开访问)", level_badge))
    story.append(Spacer(1, 10))
    story.append(Paragraph("XX科技有限公司 - 公司简介", title_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("一、公司概况", chapter_style))
    story.append(Paragraph(
        "XX科技有限公司成立于2010年，总部位于中国北京，是一家专注于人工智能和大数据技术的高新技术企业。"
        "公司现有员工500余人，其中研发人员占比超过60%。公司致力于为各行业提供智能化解决方案，"
        "目前已服务超过2000家企业客户。公司于2021年获得国家级高新技术企业认证，"
        "2022年被评为北京市专精特新小巨人企业。",
        body_style
    ))

    story.append(Paragraph("二、主营业务", chapter_style))
    story.append(Paragraph(
        "1. 人工智能平台：提供一站式AI开发平台，支持机器学习、深度学习模型的训练和部署。\n"
        "2. 大数据分析：为企业提供数据采集、清洗、分析、可视化等全流程服务。\n"
        "3. 智能客服系统：基于自然语言处理的智能问答系统，7x24小时在线服务。\n"
        "4. 企业级RAG解决方案：帮助企业构建基于私有知识的智能问答系统。",
        body_style
    ))

    story.append(Paragraph("三、企业文化", chapter_style))
    story.append(Paragraph(
        "公司秉承'客户第一、团队合作、诚信正直、创新进取'的核心价值观。"
        "我们相信技术能够改变世界，致力于用AI技术为客户创造更大价值。",
        body_style
    ))

    story.append(Spacer(1, 20))
    story.append(Paragraph("四、最新公告", chapter_style))
    story.append(Paragraph(
        "• 2024年6月1日：公司成功完成B轮融资，融资金额1亿美元\n"
        "• 2024年5月15日：公司荣获'2024年度最佳AI企业'称号\n"
        "• 2024年5月1日：公司官网全新改版上线\n"
        "• 2024年4月20日：公司员工突破500人",
        body_style
    ))

    # ========== 第2页 ==========
    story.append(PageBreak())

    story.append(Paragraph("五、公司组织架构", chapter_style))
    story.append(Paragraph(
        "公司设有以下主要部门：\n"
        "• 技术研发部：负责产品研发和技术创新\n"
        "• 产品部：负责产品规划和设计\n"
        "• 市场销售部：负责市场开拓和客户维护\n"
        "• 客户成功部：负责客户服务和售后支持\n"
        "• 人力资源部：负责招聘和员工关系\n"
        "• 财务部：负责财务管理和资金运作",
        body_style
    ))

    story.append(Paragraph("六、合作伙伴", chapter_style))
    story.append(Paragraph(
        "公司与以下机构建立了战略合作关系：\n"
        "• 清华大学人工智能研究院\n"
        "• 北京大学大数据研究中心\n"
        "• 阿里巴巴集团\n"
        "• 腾讯云计算（北京）有限责任公司\n"
        "• 华为技术有限公司",
        body_style
    ))

    story.append(Paragraph("七、联系我们", chapter_style))
    story.append(Paragraph(
        "地址：北京市朝阳区XX科技园A座18层\n"
        "电话：010-12345678\n"
        "邮箱：contact@xxtech.com\n"
        "网址：www.xxtech.com",
        body_style
    ))

    story.append(Spacer(1, 20))
    story.append(Paragraph("--- 本文件为公开信息，欢迎查阅 ---", body_style))

    doc.build(story)
    print(f"✓ 已创建: {output_path}")


def create_admin_user_pdf(output_path):
    """创建管理员级别的PDF - 包含用户账户名和密码哈希等敏感信息"""

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
        fontSize=22, textColor=colors.HexColor('#1a1a2e'), alignment=TA_CENTER, spaceAfter=20
    )
    chapter_style = ParagraphStyle(
        'Chapter', parent=styles['Heading2'], fontName=font_name,
        fontSize=16, textColor=colors.HexColor('#16213e'), spaceBefore=15, spaceAfter=10
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'], fontName=font_name,
        fontSize=11, leading=18, alignment=TA_LEFT, spaceAfter=8
    )
    sensitive_style = ParagraphStyle(
        'Sensitive', parent=styles['Normal'], fontName=font_name,
        fontSize=10, leading=16, textColor=colors.HexColor('#dc3545'), backColor=colors.HexColor('#fff3cd'),
        leftIndent=10, rightIndent=10, spaceAfter=6
    )
    level_badge = ParagraphStyle(
        'LevelBadge', parent=styles['Normal'], fontName=font_name,
        fontSize=10, textColor=colors.HexColor('#ff9800'), alignment=TA_CENTER
    )

    # 创建表格样式
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
    ])

    story = []

    # ========== 第1页 ==========

    story.append(Paragraph("🔐 文档等级：管理员 (内部敏感信息)", level_badge))
    story.append(Spacer(1, 10))
    story.append(Paragraph("系统用户账户信息 - 权限管理", title_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<font color='#dc3545'>⚠️ 警告：本文件包含系统敏感信息，仅限管理员查看，严禁外传！</font>",
        body_style
    ))
    story.append(Spacer(1, 15))

    story.append(Paragraph("一、系统管理员账户列表", chapter_style))

    # 用户表数据
    user_data = [
        ['用户ID', '用户名', '角色', '密码哈希值', '最后登录IP', '状态'],
        ['1001', 'admin', '超级管理员', hashlib.sha256('Admin@2024!'.encode()).hexdigest()[:32], '192.168.1.100',
         '活跃'],
        ['1002', 'zhang_san', '系统管理员', hashlib.sha256('ZhangSan@123'.encode()).hexdigest()[:32], '192.168.1.101',
         '活跃'],
        ['1003', 'li_si', '安全管理员', hashlib.sha256('LiSi$2024'.encode()).hexdigest()[:32], '192.168.1.102', '活跃'],
        ['1004', 'wang_wu', '审计管理员', hashlib.sha256('WangWu#456'.encode()).hexdigest()[:32], '192.168.1.103',
         '锁定'],
        ['1005', 'zhao_liu', '备份管理员', hashlib.sha256('ZhaoLiu_789'.encode()).hexdigest()[:32], '192.168.1.104',
         '活跃'],
        ['1006', 'sun_qi', '网络管理员', hashlib.sha256('SunQi@999'.encode()).hexdigest()[:32], '192.168.1.105',
         '不活跃'],
    ]

    user_table = Table(user_data, colWidths=[50, 80, 70, 180, 90, 50])
    user_table.setStyle(table_style)
    story.append(user_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("二、普通用户账户列表", chapter_style))

    normal_users = [
        ['用户ID', '用户名', '邮箱', '密码哈希值', '注册时间', '文档数'],
        ['2001', 'user001', 'user001@xxtech.com', hashlib.sha256('Pass123!'.encode()).hexdigest()[:32], '2024-01-15',
         '5'],
        ['2002', 'user002', 'user002@xxtech.com', hashlib.sha256('Hello@2024'.encode()).hexdigest()[:32], '2024-02-20',
         '12'],
        ['2003', 'user003', 'user003@xxtech.com', hashlib.sha256('Welcome#1'.encode()).hexdigest()[:32], '2024-03-10',
         '8'],
        ['2004', 'user004', 'user004@xxtech.com', hashlib.sha256('Test$567'.encode()).hexdigest()[:32], '2024-04-05',
         '3'],
        ['2005', 'user005', 'user005@xxtech.com', hashlib.sha256('Secure_88'.encode()).hexdigest()[:32], '2024-05-12',
         '20'],
        ['2006', 'user006', 'user006@xxtech.com', hashlib.sha256('MyPass@123'.encode()).hexdigest()[:32], '2024-06-01',
         '7'],
        ['2007', 'user007', 'user007@xxtech.com', hashlib.sha256('Qwerty!456'.encode()).hexdigest()[:32], '2024-06-10',
         '15'],
        ['2008', 'user008', 'user008@xxtech.com', hashlib.sha256('Changeme_1'.encode()).hexdigest()[:32], '2024-06-15',
         '9'],
    ]

    normal_table = Table(normal_users, colWidths=[50, 70, 120, 180, 70, 50])
    normal_table.setStyle(table_style)
    story.append(normal_table)

    # ========== 第2页 ==========
    story.append(PageBreak())

    story.append(Paragraph("三、API密钥和访问令牌", chapter_style))

    api_keys = [
        ['服务名称', 'API Key', '密钥哈希', '权限范围', '过期时间', '状态'],
        ['RAG服务', 'sk-rag-8f9a3b2c1d4e5f6a7b8c', hashlib.sha256('rag_key_2024'.encode()).hexdigest()[:32], '读写',
         '2025-12-31', '正常'],
        ['向量数据库', 'vec-7e6d5c4b3a2f1e9d8c7b', hashlib.sha256('vector_secret'.encode()).hexdigest()[:32], '读写',
         '2025-06-30', '正常'],
        ['LLM服务', 'llm-9a8b7c6d5e4f3a2b1c0d', hashlib.sha256('llm_api_key'.encode()).hexdigest()[:32], '只读',
         '2024-12-31', '正常'],
        ['监控系统', 'mon-1a2b3c4d5e6f7a8b9c0d', hashlib.sha256('monitor_token'.encode()).hexdigest()[:32], '读写',
         '2024-09-30', '即将过期'],
    ]

    api_table = Table(api_keys, colWidths=[60, 160, 150, 60, 70, 40])
    api_table.setStyle(table_style)
    story.append(api_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("四、数据库连接配置", chapter_style))
    story.append(Paragraph(
        "<font color='#dc3545'>⚠️ 以下为生产环境数据库配置，请勿泄露！</font>",
        body_style
    ))
    story.append(Spacer(1, 10))

    db_configs = [
        ['数据库', '主机地址', '端口', '数据库名', '用户名', '连接池大小'],
        ['PostgreSQL', '10.10.10.100', '5432', 'rag_db', 'rag_admin', '20'],
        ['Milvus', '10.10.10.101', '19530', 'rag_vector', 'milvus_user', '10'],
        ['Elasticsearch', '10.10.10.102', '9200', 'rag_bm25', 'es_admin', '15'],
        ['Redis', '10.10.10.103', '6379', 'db0', 'redis_auth', '50'],
    ]

    db_table = Table(db_configs, colWidths=[70, 100, 50, 70, 80, 60])
    db_table.setStyle(table_style)
    story.append(db_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("五、审计日志记录", chapter_style))
    story.append(Paragraph(
        "最近30天系统登录异常汇总：\n"
        "• 2024-06-15 03:22:15 - IP 203.0.113.45 尝试暴力破解用户 admin（已阻止）\n"
        "• 2024-06-14 22:10:33 - IP 198.51.100.78 使用弱密码尝试登录（已锁定账户）\n"
        "• 2024-06-13 08:45:22 - 用户 zhang_san 从新IP 192.168.2.50 登录（已记录）\n"
        "• 2024-06-12 15:30:18 - 用户 li_si 权限变更操作（已审计）\n"
        "• 2024-06-11 11:20:05 - 密码修改操作15次（正常）",
        body_style
    ))

    story.append(Spacer(1, 20))
    story.append(Paragraph("--- 本文件为内部敏感信息，请妥善保管 ---", body_style))

    doc.build(story)
    print(f"✓ 已创建: {output_path}")


def create_owner_user_pdf(output_path):
    """创建所有者级别的PDF - 公司下一步规划等绝密内容"""

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
        fontSize=22, textColor=colors.HexColor('#1a1a2e'), alignment=TA_CENTER, spaceAfter=20
    )
    chapter_style = ParagraphStyle(
        'Chapter', parent=styles['Heading2'], fontName=font_name,
        fontSize=16, textColor=colors.HexColor('#16213e'), spaceBefore=15, spaceAfter=10
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'], fontName=font_name,
        fontSize=11, leading=18, alignment=TA_JUSTIFY, spaceAfter=8
    )
    highlight_style = ParagraphStyle(
        'Highlight', parent=styles['Normal'], fontName=font_name,
        fontSize=11, leading=18, textColor=colors.HexColor('#dc3545'), alignment=TA_JUSTIFY,
        backColor=colors.HexColor('#fff3cd'), spaceAfter=8, leftIndent=10, rightIndent=10
    )
    level_badge = ParagraphStyle(
        'LevelBadge', parent=styles['Normal'], fontName=font_name,
        fontSize=10, textColor=colors.HexColor('#dc3545'), alignment=TA_CENTER
    )

    story = []

    # ========== 第1页 ==========

    story.append(Paragraph("🔒 文档等级：绝密 (仅所有者可访问)", level_badge))
    story.append(Spacer(1, 10))
    story.append(Paragraph("公司绝密战略规划 - 2024-2025年度", title_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<font color='#dc3545'>🚨 绝密文件 - 未经授权披露将追究法律责任！🚨</font>",
        highlight_style
    ))
    story.append(Spacer(1, 15))

    story.append(Paragraph("一、核心战略方向", chapter_style))
    story.append(Paragraph(
        "1. AI大模型战略：投资5亿元自主研发企业级大模型，预计2025年Q1发布1.0版本\n"
        "2. 国际化扩张：2024年Q4在新加坡设立东南亚总部，2025年进入欧美市场\n"
        "3. IPO计划：目标2025年Q3在科创板上市，预计估值200亿元\n"
        "4. 并购计划：正在洽谈收购两家AI初创公司，总预算15亿元\n"
        "5. 人才战略：计划招聘100名顶尖AI人才，设置1亿元人才激励基金",
        body_style
    ))

    story.append(Paragraph("二、2024年下半年关键里程碑", chapter_style))

    milestones = [
        ['时间节点', '里程碑事件', '负责人', '预算(万元)', '完成度'],
        ['2024-07-31', '大模型1.0内测版发布', '技术部-张总', '5000', '60%'],
        ['2024-08-15', '完成B+轮融资(2亿美元)', '财务部-李总', '0', '80%'],
        ['2024-09-30', '新加坡子公司注册完成', '海外事业部-王总', '3000', '40%'],
        ['2024-10-31', '月营收突破5000万元', '销售部-赵总', '0', '55%'],
        ['2024-11-30', '并购A公司交割完成', '战略部-孙总', '80000', '30%'],
        ['2024-12-31', '完成年度目标120%', 'CEO-周总', '0', '65%'],
    ]

    milestone_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc3545')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
    ])

    milestone_table = Table(milestones, colWidths=[70, 130, 70, 70, 50])
    milestone_table.setStyle(milestone_style)
    story.append(milestone_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("三、核心技术路线图", chapter_style))
    story.append(Paragraph(
        "1. 大模型技术栈：\n"
        "   - 基座模型：基于Llama 3和Qwen 2的二次开发\n"
        "   - 训练数据：2TB高质量中文语料（已采购）\n"
        "   - 算力资源：部署500台A100/H800服务器\n"
        "2. RAG技术升级：\n"
        "   - 多模态检索：支持图像、音频、视频检索\n"
        "   - 实时知识更新：从T+7提升至T+1\n"
        "   - 长上下文支持：从8K扩展到128K\n"
        "3. 安全合规：\n"
        "   - 通过等保3级认证\n"
        "   - 获得ISO 27001信息安全管理体系认证\n"
        "   - 建立数据安全岛（隐私计算平台）",
        body_style
    ))

    # ========== 第2页 ==========
    story.append(PageBreak())

    story.append(Paragraph("四、财务预测与融资计划", chapter_style))

    financials = [
        ['指标', '2024年(H1)', '2024年(H2预计)', '2025年(预计)', '2026年(预计)'],
        ['营业收入(万元)', '18000', '32000', '100000', '250000'],
        ['净利润(万元)', '2000', '5000', '20000', '60000'],
        ['研发投入(万元)', '8000', '12000', '35000', '80000'],
        ['员工人数(人)', '520', '650', '1000', '1500'],
        ['毛利率(%)', '65%', '68%', '70%', '72%'],
    ]

    financial_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff9800')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
    ])

    financial_table = Table(financials, colWidths=[90, 90, 100, 90, 90])
    financial_table.setStyle(financial_style)
    story.append(financial_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("五、重大合作与收购计划", chapter_style))
    story.append(Paragraph(
        "1. 战略投资：某知名互联网公司拟以50亿元估值投资10亿元（尽调中）\n"
        "2. 被收购对象：\n"
        "   - B公司（AI芯片设计）：估值8亿元，预计2024年Q4交割\n"
        "   - C公司（自动驾驶算法）：估值12亿元，预计2025年Q1交割\n"
        "3. 战略合作：\n"
        "   - 与某云计算巨头签署5年独家合作协议（预计年收入5亿元）\n"
        "   - 与某高校共建AI联合实验室（3年投入1.5亿元）",
        body_style
    ))

    story.append(Spacer(1, 15))
    story.append(Paragraph("六、风险与应对措施", chapter_style))
    story.append(Paragraph(
        "主要风险及应对方案：\n"
        "1. 技术风险：大模型研发进度延迟 → 备用方案：与第三方大模型公司合作\n"
        "2. 市场风险：竞争加剧导致市场份额下降 → 应对：加强差异化产品策略\n"
        "3. 人才风险：核心技术人员流失 → 应对：实施股权激励计划（已获批）\n"
        "4. 合规风险：数据安全法规趋严 → 应对：成立法务合规部（编制20人）\n"
        "5. 财务风险：现金流紧张 → 应对：已获得银行授信额度10亿元",
        body_style
    ))

    story.append(Spacer(1, 15))
    story.append(Paragraph("七、董事会决议摘要", chapter_style))
    story.append(Paragraph(
        "2024年6月10日董事会决议：\n"
        "• 批准公司IPO上市计划，由中信证券担任保荐机构\n"
        "• 批准5亿元AI大模型研发专项资金\n"
        "• 批准员工股权激励计划（覆盖30%核心员工）\n"
        "• 批准设立新加坡海外总部，首期投入3000万美元\n"
        "• 授权CEO签署并购协议（金额上限15亿元）",
        body_style
    ))

    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "<font color='#dc3545'>--- 绝密文件 - 未经授权不得复制、传播或讨论 ---</font>",
        body_style
    ))

    doc.build(story)
    print(f"✓ 已创建: {output_path}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("创建测试PDF文档")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path(__file__).parent.parent / "uploads_tmp"
    output_dir.mkdir(exist_ok=True)

    print(f"\n输出目录: {output_dir}")
    print("\n正在创建PDF文档...\n")

    # 创建三个PDF
    create_normal_user_pdf(output_dir / "普通用户.pdf")
    create_admin_user_pdf(output_dir / "管理员用户.pdf")
    create_owner_user_pdf(output_dir / "所有者用户.pdf")

    print("\n" + "=" * 60)
    print("所有PDF文档创建完成！")
    print("=" * 60)
    print("\n文档说明：")
    print("  1. 普通用户.pdf - 公司简介、公告等公开信息（normal级别）")
    print("  2. 管理员用户.pdf - 用户账户密码哈希、API密钥等敏感信息（admin级别）")
    print("  3. 所有者用户.pdf - 公司IPO计划、并购计划等绝密信息（owner级别）")
    print("\n测试建议：")
    print("  - 使用普通用户账号登录：只能看到'普通用户.pdf'")
    print("  - 使用管理员账号(admin开头)登录：可以看到前两个PDF")
    print("  - 使用所有者账号(root/system开头)登录：可以看到全部三个PDF")
    print("=" * 60)


if __name__ == "__main__":
    main()
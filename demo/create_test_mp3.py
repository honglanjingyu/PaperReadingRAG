from gtts import gTTS
import os

# 文本内容
text = """
终端需求疲软，收入短期承压。公司上半年收入同比下降 2.32%，
主要受终端需求疲软、下游渠道持续调整库存影响。
公司目前下游应用领域中，汽车电子份额占比最高，
同时公司也在大力拓展风力、光伏及储能相关产品的PCB业务。
Q2收入环比上升11.88%，主要因为稼动率有所提升。
"""

# 语言为中文
tts = gTTS(text=text, lang='zh-cn', slow=False)

# 保存为 mp3 文件
output_file = "output.mp3"
tts.save(output_file)

print(f"MP3 文件已生成：{output_file}")
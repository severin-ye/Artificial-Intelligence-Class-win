
from gtts import gTTS
from playsound import playsound
import os

# 创建新闻文本
news_text = "孙兴慜在比赛中表现出色，攻入两球，助攻一次，最终帮助球队以3比1获胜。"

# 将文本转换为语音
tts = gTTS(text=news_text, lang='zh-cn')
tts.save("news.mp3")

# 播放生成的语音文件
playsound("news.mp3") # 播放语音文件

# 删除生成的语音文件（可选）
os.remove("news.mp3")


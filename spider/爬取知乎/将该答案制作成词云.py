#!/usr/bin/python3
'''
    auth:gzy
    date:20190927
    version:V0.1.0
'''

import pandas as pd
import os
from wordcloud import WordCloud
import jieba.analyse
import numpy as np
from PIL import Image
import collections

answersfile = os.getcwd()+"/答案.text"
# data = pd.read_csv("ZHIHU2.csv")
# answers = data["answer"]
# for i in answers:
#     with open(answersfile,"a",encoding="utf-8") as f:
#         f.write(i+"\n")

with open("stopwords.text","r",encoding="utf-8") as p:
    stops = set(p.read().splitlines())

with open(answersfile,'r',encoding='utf-8') as q:
    text = q.read()

tags = jieba.cut(text)
cloudtext = ",".join(tags)

# #词频率统计
# word_counts = collections.Counter(cloudtext)
# word_counts_top10 = word_counts.most_common(100)
# print(word_counts_top10)

# 创建词云
mask = np.array(Image.open("bg1.jpeg"))
woc = WordCloud(
    background_color="white",
    max_font_size=80,
    min_font_size=4,
    font_path="../SourceHanSans-Normal.otf",
    stopwords=stops,
    width=800,
    height=600,
    mask=mask,

)
woc.generate(cloudtext)
# 保存图片
woc.to_file("结果1.png")
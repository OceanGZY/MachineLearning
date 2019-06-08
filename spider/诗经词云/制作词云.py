#!/usr/bin/python
#-*- coding:utf-8 -*-
'''
    auth:gzy
    version:0.1.0
    date:20190608
'''
import jieba
import jieba.analyse
from wordcloud import WordCloud,ImageColorGenerator
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

with open('诗经.text',encoding="utf-8") as f:
    text = f.read()
with open("stopwords.text",encoding="utf-8") as s:
    stops = {}.fromkeys(s.read().split("\n"))

tags = jieba.cut(text)
cloud_text = ",".join(tags)
# st = set(["诗经","先秦","殷商","父母","小子","汝之何","叔兮伯兮","岂不怀","不尔思","上帝","王事靡","不知","不知","谓之","孙子","如何","止于","曾孙","心之忧","文王","之子","扬之水","兄弟","至于","既见"])
st = set(["诗经","先秦","殷商"])
mask1 = np.array(Image.open("bg1.png"))
mask2 = np.array(Image.open("bg2.png"))
wc = WordCloud(
    background_color="white",
    max_font_size=400,
    font_path="../SourceHanSans-Normal.otf",
    min_font_size=4,
    width=1440,
    height=960,
    stopwords=st,
    relative_scaling=0.6,
    # mask=mask1,
    mask=mask2,
)
wc.generate(cloud_text)
# wc.to_file('诗经.png')
# wc.to_file('诗经_bg1.png')
wc.to_file('诗经_bg2.png')
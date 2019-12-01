#!/usr/bin/python3
'''
    author:gzy
    date:20191004
    version:0.1.0
'''
import pandas as pd
from wordcloud import WordCloud
import numpy as np
from PIL import Image
import os
import seaborn as sns
import matplotlib.pyplot as plt

class WordS(object):
    def __init__(self):
        self.data = pd.read_csv("nalanci.csv")

    def make_text_file(self):
        titles = self.data["title"]
        poems = self.data["poem"]



if __name__ == "__main__":
    word = WordS()


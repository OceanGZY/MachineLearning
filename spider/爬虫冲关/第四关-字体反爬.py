#!/usr/bin/python3
'''
    date:20190904
    author:GZY
'''
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from fontTools.ttLib import TTFont

font = TTFont("4Jdx7H.woff")
font.saveXML('result.xml')
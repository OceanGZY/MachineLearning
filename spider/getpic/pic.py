import requests
from bs4 import BeautifulSoup
import time
import os
homepath = os.getcwd()
base = 'https://www.zbjuran.com/'
for i in range(1,22):
    url = base +'mei/xinggan/201906/96063_' + str(i) +'.html'
    time.sleep(1)
    res = requests.get(url)
    html = res.content
    soup = BeautifulSoup(html,'lxml')
    imgdetails = soup.select(".picbox img")
    for imgdetail in imgdetails:
        imgname = imgdetail.get("alt").strip()
        imgurl = imgdetail.get("src")
        print(imgname,imgurl)
        rimg = requests.get(imgurl)
        cimg = rimg.content
        file = homepath+"/meizi/"+imgname +".jpg"
        with open(file,"wb+") as f:
            f.write(cimg)
            print("保存 %s",str(i),"张")
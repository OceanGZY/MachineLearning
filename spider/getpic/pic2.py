import requests
from bs4 import BeautifulSoup
import time
import os
homepath = os.getcwd()
base = 'https://www.zbjuran.com/'
for j in range(1,11):
    pageurl = base + 'mei/xinggan/' +'list_13_'+ str(j)+'.html'
    time.sleep(1)
    pres = requests.get(pageurl)
    phtml = pres.text
    psoup = BeautifulSoup(phtml,"lxml")
    # print(psoup)
    pimgpages = psoup.select(".name a")
    # print(pimgpages)
    for pimgpage in pimgpages:
        pimgname  = pimgpage.text.strip()
        pimgurl = pimgpage.get("href").strip(".html")
        filepath = homepath + '/' +  pimgname + '/'
        try:
            os.mkdir(filepath)
        except Exception as e:
            print('创建文件夹失败',e)
        for i in range(1,22):
            url  = base + pimgurl + '_'+str(i) + '.html'
            time.sleep(1)
            res = requests.get(url)
            html = res.content
            soup = BeautifulSoup(html, 'lxml')
            imgdetails = soup.select(".picbox img")
            for imgdetail in imgdetails:
                imgname = imgdetail.get("alt").strip()
                imgurl = imgdetail.get("src")
                print(imgname, imgurl)
                rimg = requests.get(imgurl)
                cimg = rimg.content
                file = filepath + imgname + ".jpg"

                with open(file, "wb+") as f:
                    f.write(cimg)
                    print("保存",str(i),"张")


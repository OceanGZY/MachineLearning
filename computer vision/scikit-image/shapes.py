# -*- coding:utf-8 -*-
'''
    auth:YSilhouette
    date:2018/11/14
    version :0.1.0
'''
import math
import numpy as np
import matplotlib.pyplot as plt

from skimage.draw import (line,polygon,circle,circle_perimeter,
                          ellipse,ellipse_perimeter,bezier_curve,
                          line_aa,circle_perimeter_aa)

fig,(ax1,ax2) =plt.subplots(ncols=2,nrows=1,figsize =(10,6))


img = np.zeros((100,100),dtype=np.double)

# anti-aliased line

rr,cc,val =line_aa(12,12,20,50)
img[rr,cc] = val

# anti-aliased circle
rr,cc,val = circle_perimeter_aa(60,40,30)
img[rr,cc] =val

ax2.imshow (img,cmap = plt.cm.gray,interpolation = 'nearest')
ax2.set_title ('Anti-aliased Circle')
ax2.axis('off')
# plt.show()


print(np.random.rand(3,3) -0.5)



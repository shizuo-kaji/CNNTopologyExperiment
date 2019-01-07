#!/usr/bin/env python
# create random images for learning Betti numbers
# there two types of images: binary images and their distance transform 

from PIL import Image
import numpy as np
import random
import sys
import skimage.morphology as sm
import skimage.io as io
from scipy.ndimage.morphology import distance_transform_edt


if __name__ == '__main__':
    n,m=130,130
    min_width = 8
    max_width = 20
    n_components = 20
    n_imgs = 1000
    tool = sm.disk(5)

    for j in range(n_imgs):
        img = np.zeros((n,m),dtype=np.uint8)
        for i in range(n_components):
            top = random.randint(min_width,n-2*min_width)
            left = random.randint(min_width,m-2*min_width)
            w = random.randint(min_width,max_width)
            h = random.randint(min_width,max_width)
            img[left:min(n-min_width,left+w),top:min(m-min_width,top+h)] = 1

        img = sm.binary_closing(img, tool)
        io.imsave('{:0>5}.png'.format(j),img*255)
        img = distance_transform_edt(img)-distance_transform_edt(~img)
        #np.savetxt(fname,img,delimiter=",",fmt="%1d")
#        print(img.max(),img.min())
        io.imsave('dt_{:0>5}.jpg'.format(j),np.uint8(127*(img+n)/max(n,m)))

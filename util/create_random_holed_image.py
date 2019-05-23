#!/usr/bin/env python
# create random images for learning Betti numbers
# there two types of images: binary images and their distance transform 

from PIL import Image
import numpy as np
import random
import sys
import os
import skimage.morphology as sm
import skimage.io as io
from scipy.ndimage.morphology import distance_transform_edt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create sinograms for artificial images')
    parser.add_argument('--size', '-s', type=int, default=130,
                        help='size of the image')
    parser.add_argument('--min_width', '-mi', type=int, default=8,
                        help='minimum width of component to be created')
    parser.add_argument('--max_width', '-ma', type=int, default=20,
                        help='maximum width of component to be created')
    parser.add_argument('--n_components', '-nc', type=int, default=20,
                        help='Number of components to be created')
    parser.add_argument('--num', '-n', type=int, default=2000,
                        help='Number of images to be created')
    parser.add_argument('--noise', '-z', type=int, default=0,
                        help='Strength of noise')
    parser.add_argument('--outdir', '-o', default='betti',
                        help='output directory')
    args = parser.parse_args()

    ###
    os.makedirs(args.outdir, exist_ok=True)
    tool = sm.disk(5)

    for j in range(args.num):
        img = np.zeros((args.size,args.size),dtype=np.uint8)
        for i in range(args.n_components):
            top = random.randint(args.min_width,args.size-2*args.min_width)
            left = random.randint(args.min_width,args.size-2*args.min_width)
            w = random.randint(args.min_width,args.max_width)
            h = random.randint(args.min_width,args.max_width)
            img[left:min(args.size-args.min_width,left+w),top:min(args.size-args.min_width,top+h)] = 1

        img = sm.binary_closing(img, tool)
        io.imsave(os.path.join(args.outdir,'{:0>5}.png'.format(j)),(img*255).astype(np.uint8))
#        img = distance_transform_edt(img)-distance_transform_edt(~img)
#        img = 127.5*(3*img+1.5*args.size)/args.size
#        print(img.max(),img.min())
#        io.imsave(os.path.join(args.outdir,'dt_{:0>5}.png'.format(j)),np.clip(np.uint8(img),0,255))

import os
import random
import glob

from chainer.dataset import dataset_mixin
import numpy as np
from PIL import Image
import pandas as pd

from chainercv.transforms import random_crop,center_crop,random_flip,resize
from chainercv.utils import read_image
import chainer.links as L


## dataset preparation
# the first column contains the filename
# the other columns are for target values
class Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, path, csv, cw=480,ch=320,random=0, regression=True, time_series=False, cols=[1]):
        self.path = path
        self.time_series = time_series
        self.skiprows = 1
        self.ids = []
        self.random = random
        self.cw = cw
        self.ch = ch
        self.cols=cols
        self.color=True
        self.amp = 30 # multiplier for small valued time series data
        self.regression = regression
        dtype = np.float32 if regression else np.int32
        dat = pd.read_csv(csv, header=None)
        self.csvdata = dat.iloc[:,cols].values.astype(dtype)
        if regression:
            self.mean = self.csvdata.mean(axis=0, keepdims=True)
            self.std  = np.std(self.csvdata, axis=0, keepdims=True)
            self.chs = len(cols)
            self.dat = (self.csvdata-self.mean)/self.std
        else: # for classification, only the first target column is used
            self.mean = np.zeros_like(self.csvdata[:,0])
            self.std = np.ones_like(self.csvdata[:,0])
            self.chs = np.max(self.csvdata[:,0]) + 1   # +1 for class id "0"
            self.dat = self.csvdata[:,0]
        print("{} subjects, {} out channels, {}".format(len(self.csvdata),self.chs,self.dat.dtype))
        for e in dat.iloc[:,0].values:
#            fn='{0:05d}.png'.format(int(e[0]))
            self.ids.append(os.path.join(path,e))

    def __len__(self):
        return len(self.ids)

    def get_img_path(self, i):
        return self.ids[i]

    def get_example(self, i):
        if self.time_series:
            offset = random.randrange(self.random) if self.random>0 else 0
            img = np.loadtxt(self.get_img_path(i), delimiter=',', skiprows=self.skiprows, dtype=np.float32)*self.amp
            img = img[offset:(offset+3*self.cw*self.ch),self.cols[0]].reshape(3,self.ch,self.cw)
        else:
            img = read_image(self.get_img_path(i),color=self.color)
            img = L.model.vision.resnet.prepare(img, size=None)
            img = center_crop(img,(self.ch+self.random, self.cw+self.random))
            img = random_crop(img, (self.ch,self.cw))
#            img = img * 2 / 255.0 - 1.0  # [-1, 1)
            if self.random>0:
                img = random_flip(img, x_random=True)
        return (img,self.dat[i])




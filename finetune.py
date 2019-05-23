#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training,datasets,iterators
from chainer.training import extensions
from chainer.dataset import dataset_mixin, convert, concat_examples
import numpy as np
import functools

#from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args

import os
import glob
from datetime import datetime as dt

from dataset_img import Dataset

# optimisers
optim = {
    'Momentum': functools.partial(chainer.optimizers.MomentumSGD, lr=0.01, momentum=0.9),
    'AdaDelta': functools.partial(chainer.optimizers.AdaDelta,rho=0.95, eps=1e-06),
    'AdaGrad': functools.partial(chainer.optimizers.AdaGrad,lr=0.001, eps=1e-08),
    'Adam': functools.partial(chainer.optimizers.Adam,alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08),
    'RMSprop': functools.partial(chainer.optimizers.RMSprop,lr=0.01, alpha=0.99, eps=1e-08),
    'NesterovAG': functools.partial(chainer.optimizers.NesterovAG,lr=0.01, momentum=0.9)
}

dtypes = {
    'fp16': np.float16,
    'fp32': np.float32
}

## NN definition
class Resnet(chainer.Chain):
    def __init__(self, args):
        self.dropout = args.dropout
        w = chainer.initializers.HeNormal()
        bias = chainer.initializers.Zero()
        out_ch = len(args.cols) if args.regress else args.nclass 
        super(Resnet, self).__init__(
            base = L.ResNet152Layers(),
#            pointwise = L.Convolution2D(None, len(args.cols), 1, 1, 0, initialW=w, initial_bias=bias),
            fc1 = L.Linear(None,1024),
            fc2 = L.Linear(1024, out_ch)
        )
    def __call__(self, x):
        h = self.base(x, layers=['res5'])['res5']
#        h = self.base(x, layers=['pool5'])['pool5']
        h = F.dropout(h,ratio=self.dropout/2)
#        h = F.max(self.pointwise(h),axis=(2,3))
        h = F.relu(self.fc1(h))
        h = F.dropout(h,ratio=self.dropout)
        h = self.fc2(h)
        return h


def main():
    # command line argument parsing
    parser = argparse.ArgumentParser(description='Multi-Perceptron classifier/regressor')
    parser.add_argument('train', help='Path to csv file')
    parser.add_argument('--root', '-R', default="betti", help='Path to image files')
    parser.add_argument('--val', help='Path to validation csv file',required=True)
    parser.add_argument('--regress', '-r', action='store_true', help='set for regression, otherwise classification')
    parser.add_argument('--time_series', '-ts', action='store_true', help='set for time series data')
    parser.add_argument('--batchsize', '-b', type=int, default=10,
                        help='Number of samples in each mini-batch')
    parser.add_argument('--nclass', '-nc', type=int, default=2,
                        help='Number of classes for classification')
    parser.add_argument('--cols', '-c', type=int, nargs="*", default=[1],
                        help='column indices in csv of target variables')
    parser.add_argument('--epoch', '-e', type=int, default=500,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--snapshot', '-s', type=int, default=100,
                        help='snapshot interval')
    parser.add_argument('--initmodel', '-i',
                        help='Initialize the model from given file')
    parser.add_argument('--random', '-rt', type=int, default=1,
                        help='random translation')
    parser.add_argument('--gpu', '-g', type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--loaderjob', '-j', type=int, default=3,
                        help='Number of parallel data loading processes')
    parser.add_argument('--outdir', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--optimizer', '-op',choices=optim.keys(),default='Adam',
                        help='optimizer')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume the training from snapshot')
    parser.add_argument('--predict', '-p', action='store_true', help='prediction with a specified model')
    parser.add_argument('--tuning_rate', '-tr', type=float, default=0.1,
                        help='learning rate for pretrained layers')
    parser.add_argument('--dropout', '-dr', type=float, default=0,
                        help='dropout ratio for the FC layers')
    parser.add_argument('--cw', '-cw', type=int, default=128,
                        help='crop image width')
    parser.add_argument('--ch', '-ch', type=int, default=128,
                        help='crop image height')
    parser.add_argument('--weight_decay', '-w', type=float, default=1e-6,
                        help='weight decay for regularization')
    parser.add_argument('--wd_norm', '-wn', choices=['none','l1','l2'], default='l2',
                        help='norm of weight decay for regularization')
    parser.add_argument('--dtype', '-dt', choices=dtypes.keys(), default='fp32',
                        help='floating point precision')
    args = parser.parse_args()

    args.outdir = os.path.join(args.outdir, dt.now().strftime('%m%d_%H%M'))
    # Enable autotuner of cuDNN
    chainer.config.autotune = True
    chainer.config.dtype = dtypes[args.dtype]
    chainer.print_runtime_info()
    ##
    if not args.gpu:
        if chainer.cuda.available:
            args.gpu = 0
        else:
            args.gpu = -1          
    print(args)

    if args.regress:
        accfun = F.mean_absolute_error
        lossfun = F.mean_squared_error
        out_ch = len(args.cols)
    else:
        accfun = F.accuracy
        lossfun = F.softmax_cross_entropy
        out_ch = args.nclass 

    # Set up a neural network to train
    model = L.Classifier(Resnet(args), lossfun=lossfun, accfun=accfun)
    
    # Set up an optimizer
    optimizer = optim[args.optimizer]()
    optimizer.setup(model)
    if args.weight_decay>0:
        if args.wd_norm =='l2':
            optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
        elif args.wd_norm =='l1':
            optimizer.add_hook(chainer.optimizer_hooks.Lasso(args.weight_decay))
    # slow update for pretrained layers
    if args.optimizer in ['Adam']:
        for func_name in model.predictor.base._children:
            for param in model.predictor.base[func_name].params():
                param.update_rule.hyperparam.alpha *= args.tuning_rate

    if args.initmodel:
        print('Load model from: ', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # select numpy or cupy
    xp = chainer.cuda.cupy if args.gpu >= 0 else np

    # read csv file
    train = Dataset(args.root,args.train, cw=args.cw,ch=args.ch,random=args.random,regression=args.regress,time_series=args.time_series,cols=args.cols)
    test = Dataset(args.root,args.val,cw=args.cw,ch=args.ch, regression=args.regress,time_series=args.time_series,cols=args.cols)

    train_iter = iterators.SerialIterator(train, args.batchsize, shuffle=True)
    test_iter = iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.outdir)

    frequency = args.epoch if args.snapshot == -1 else max(1, args.snapshot)
    log_interval = 1, 'epoch'
    val_interval = 20, 'epoch' # frequency/10, 'epoch'

#    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, 'model_epoch_{.updater.epoch}'), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu),trigger=val_interval)

    if args.optimizer in ['Momentum','AdaGrad','RMSprop']:
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.ExponentialShift('lr', 0.5), trigger=(args.epoch/5, 'epoch'))
    elif args.optimizer in ['Adam']:
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.ExponentialShift("alpha", 0.5, optimizer=optimizer), trigger=(args.epoch/5, 'epoch'))

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['main/loss','validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(['main/accuracy','validation/main/accuracy'],
                                  'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.PrintReport([
            'epoch', 'main/loss', 'main/accuracy','validation/main/loss','validation/main/accuracy',
          'elapsed_time', 'lr'
         ]),trigger=log_interval)

    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # ChainerUI
    #trainer.extend(CommandsExtension())
    save_args(args, args.outdir)
    trainer.extend(extensions.LogReport(trigger=log_interval))

    if not args.predict:
        trainer.run()

    ## prediction
    print("predicting: {} entries...".format(len(test)))
    test_iter = iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)
    converter=concat_examples
    idx=0
    with open(os.path.join(args.outdir,'result.txt'),'w') as output:
        for batch in test_iter:
            x, t = converter(batch, device=args.gpu)
            with chainer.using_config('train', False):
                with chainer.function.no_backprop_mode():
                    if args.regress:
                        y = model.predictor(x).data
                        if args.gpu>-1:
                            y = xp.asnumpy(y)
                            t = xp.asnumpy(t)
                        y = y * test.std + test.mean
                        t = t * test.std + test.mean
                    else:
                        y = F.softmax(model.predictor(x)).data
                        if args.gpu>-1:
                            y = xp.asnumpy(y)
                            t = xp.asnumpy(t)
            for i in range(y.shape[0]):
                output.write(os.path.basename(test.ids[idx]))
                if(len(t.shape)>1):
                    for j in range(t.shape[1]):
                        output.write(",{}".format(t[i,j]))
                        output.write(",{}".format(y[i,j]))
                else:
                    output.write(",{}".format(t[i]))
                    output.write(",{}".format(np.argmax(y[i,:])))
                    output.write(",{0[0]:1.5f},{0[1]:1.5f}".format(y[i,:]))
                output.write("\n")
                idx += 1

#    np.savetxt(os.path.join(args.outdir,"result.csv"), result , fmt='%s', delimiter=",") # fmt='%1.5f'

if __name__ == '__main__':
    main()

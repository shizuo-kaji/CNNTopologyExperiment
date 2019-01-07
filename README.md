Can a CNN learn topology?
=============
An experiment to investigate whether a CNN can learn to compute topological invariants
by Shizuo KAJI

Topological invariants encode global features of a shape/image.
Their definitions are usually highly sophisticated.
Thus, it is natural to believe that it is difficult to be learned by a CNN,
which consists of simple and local information manipulation layers.

## Licence
MIT Licence

## Requirements
- chainer >= 5.0.0: > pip install cupy,chainer,chainerui,chainercv
- R and the TDA package: (in R) > install.packages("TDA")

## Experiment
- Download all the files and extract betti_images.zip
- Start training for Betti number computation by

```python finetune.py betti_train.txt --val betti_val.txt  -c 1 2 -o ~/Downloads/result/ -r -R betti```

The result is, the CNN succeeds in learning homology!

![results](https://github.com/shizuo-kaji/CNNTopologyExperiment/blob/master/H0.jpeg?raw=true)
![results](https://github.com/shizuo-kaji/CNNTopologyExperiment/blob/master/H1.jpeg?raw=true)

The blue line shows the true value and the red line shows the corresponding prediction.

### Details
- We prepare random binary images with holes (random_image.py).
- We compute their (persistent) homology (Pbetti.R).
- We have two experiments; regression for (1) the ordinary 0th and 1st Betti numbers of the binary image (00???.png)
(2) the life-time weighted Betti numbers of the sub-level filtration of the distance transform (dt_00???.jpg) of the binary image. This is one of the typical ways to define a topological feature of a binary image. 
- Our CNN is a ResNet pre-trained for image classification, and we finetune it.
- We use binary images (not their distance transform) for learning.

## TODO
- 2nd homology for volumetric images


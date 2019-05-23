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
- chainer >= 5.3.0: > pip install cupy,chainer,chainerui,chainercv
- R and the TDA package: (in R) > install.packages("TDA")

## Experiment
- Download all the files and extract betti.zip
- Start training for Betti number computation by

```python finetune.py betti_train.csv --val betti_val.csv -c 1 2 -nc 11 -R betti -rt 1 -cw 128 -ch 128```
for 11-class (0 to 10) classification for ordinary Betti number.

(note: to ignore the boundary of the image, set cw + rt =< 130, which is the size of the image)

```python finetune.py betti_train.csv --val betti_val.csv -c 3 4 --regress -R betti -rt 1 -cw 128 -ch 128```
for regression for persistent (lifetime-weighted) Betti number

The csv files contain lines of the form
```
filename, 0th Betti, 1th Betti, 0th persistent Betti, 1th persistent Betti
```

The result is, the CNN succeeds in learning homology!

![results](https://github.com/shizuo-kaji/CNNTopologyExperiment/blob/master/H0.jpeg?raw=true)
![results](https://github.com/shizuo-kaji/CNNTopologyExperiment/blob/master/H1.jpeg?raw=true)

The blue line shows the true value and the red line shows the corresponding prediction.

### Details
- We prepare random binary images with holes (util/create_random_holed_image).
- We compute their (persistent) homology (util/Pbetti.R).
- We have two experiments; classification for (1) the ordinary 0th and 1st Betti numbers of the binary image (00???.png)
(2) the life-time weighted Betti numbers of the sub-level filtration of the distance transform (00???_dt.png) of the binary image. This is one of the popular ways to define a topological feature of a binary image. 
- Our CNN is a ResNet pre-trained for image classification, and we finetune it.
- We use binary images (not their distance transform) for learning.

## TODO
- 2nd homology for volumetric images

## Bonus: Time-series analysis
This is a totally different topic.
We found that the pre-trained ResNet for image classification worked well with a time series data!
The idea is simple: given a one-dimensional time series x[t], reshape it into an RGB image of dimension
(3,ch,cw) and feed it into the network.
We do not know why this works, but our speculation is:
- it captures not only short range auto-correlation but long range auto-correlation by "folding" the time series.
(this means, the performance depends highly on the image size (ch,cw).) 
- there are universal characteristic signal patterns which are valid both for image and time-series.

```python finetune.py train.csv --val test.csv -ts -cw 56 -ch 56 -rt 20 -R tsdata/ -c 1```


# Video_Colorization

### Table of Contents
1. [Introduction](#introduction)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Dataset](#dataset)
1. [Apply Pre-trained Models](#apply-pre-trained-models)
1. [Training and Testing](#training-and-testing)
1. [Evaluation](#evaluation)
1. [Image Processing Algorithms](#image-processing-algorithms)
1. [Acknowledge](#acknowledge)


### Introduction
Our method is designed to colorize videos based on [PWC-Net](https://github.com/NVlabs/PWC-Net), a newer method to calculate optical flow on pytorch1.0.0 and python3.6.

It is a project for my Summer Term Directed Studies. Thank for my supervisor [Dr L M Po](http://www.ee.cityu.edu.hk/~lmpo/) and my project director [Yuzhi ZHAO](https://github.com/zhaoyuzhi). They helped me a lot to finish my project.


### Requirements and dependencies
- Python3.6
- [Pytorch 1.0.0](https://pytorch.org/)
- [pytorch-pwc](https://github.com/sniklaus/pytorch-pwc) (Code for [PWC-Net](https://github.com/NVlabs/PWC-Net) on pytorch1.0 and python2.0)

Our code is tested on Ubuntu 16.04 with cuda 9.0 and cudnn 7.0.


### Installation
Download repository:

    git clone https://github.com/Sheroa/Video_Colorization.git

### Dataset
The dataset I used for training and testing is [DAVIS](https://davischallenge.org/index.html) and [videvo](https://www.videvo.net/).
    
### Apply pre-trained models
I have done many different experiments. And put some trained models on google-drive. 

You can just download it in the folder of this repository and unzip it to get the pre-trained model. They are in the folder named "pre_trained_model"

### Training and testing
Train a new model:

    python train.py

The default parameters are specified in train.py. There are four train functions. Two for single frame training, enhancing the colorization network. Two for neighboring frames training, enhancing the temporal continuity. Between the functions, two are with GAN.
1. firstly, train the network using the image dataset of [ILSVRC2012_train_256](http://image-net.org/challenges/LSVRC/2012/) without GAN. 
You can download the dataset on this [website](http://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2).

	The video dataset is much less than the dataset of the image. If just use the video dataset to train, the colorization result may be poor. So we used the image dataset to pre-train the network.

		python train.py --epoch 10 --pre_train True --singleFrame True --gan_mode False --baseroot folderNameOfDataSet
2. Secondly, train the network using the same dataset in step1 with GAN.

		python train.py --epoch 10 --pre_train False --singleFrame True --gan_mode False --baseroot folderName --load_name ModelName
3. Thirdly, train the network using video datasets without GAN.

		python train.py --epoch 500 --pre_train True --singleFrame False --gan_mode False --baseroot folderNameOfDataSet

4. Forth, train the network using video datasets with GAN.

		python train.py --epoch 500 --pre_train False --singleFrame False --gan_mode True --baseroot folderNameOfDataSet --load_name ModelName

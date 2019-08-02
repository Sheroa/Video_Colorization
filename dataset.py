import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import color
import random

class NormalRGBDataset(Dataset):
    def __init__(self, opt, imglist):                                   # root: list ; transform: torch transform
        self.baseroot = opt.baseroot
        self.imglist = imglist
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        imgpath = self.baseroot + '/' + self.imglist[index]            # path of one image
        colorimg = Image.open(imgpath)                                  # read one image
        colorimg = colorimg.convert('RGB')
        img = colorimg.crop((256, 0, 512, 256))
        target = colorimg.crop((0, 0, 256, 256))
        img = self.transform(img)
        target = self.transform(target)
        return img, target
    
    def __len__(self):
        return len(self.imglist)
        
class ColorizationDataset(Dataset):
    def __init__(self, opt, imglist):                                   # root: list ; transform: torch transform
        self.baseroot = opt.baseroot
        self.imglist = imglist
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        imgpath = self.baseroot + '/' + self.imglist[index]            # path of one image
        colorimg = Image.open(imgpath)                                  # read one image
        greyimg = colorimg.convert('L').convert('RGB')                  # convert to grey scale, and concat to 3 channels
        colorimg = colorimg.convert('RGB')                              # convert to color RGB
        img = self.transform(greyimg)
        target = self.transform(colorimg)
        return img, target
    
    def __len__(self):
        return len(self.imglist)

class MultiFramesDataset(Dataset):
    def __init__(self, opt, imglist, classlist):
        # if you want to use this code, please ensure:
        # 1. the structure of the training set should be given as: baseroot + classname (many classes) + imagename (many images)
        # 2. all the input images should be categorized, and each folder contains all the images of this class
        # 3. all the name of images should be given as: 00000.jpg, 00001.jpg, 00002.jpg, ... , 00NNN.jpg (if possible, not mandatary)
        # note that:
        # 1. self.baseroot: the overall base
        # 2. self.classlist: the second base, it could not be used in get_item
        # 3. self.imgroot: the relative root for each image, and classified by categories 二维列表
        # number of classes = len(self.classlist) = len(self.imgroot)
        # number of images in a specific class = len(self.imgroot[k])
        # this dataset is fair for each class, not for each image, because the number of images in different classes is different
        self.baseroot = opt.baseroot                                    # baseroot is the base of all images
        self.classlist = classlist                                      # classlist should contain the category name of the series of frames
        self.imgroot = [list() for i in range(len(classlist))]          # imgroot contains the relative path of all images
        self.task = opt.task                                            # specific task
        self.iter_frames = opt.iter_frames                              # in one iteration, the number of images are used
        self.totensor = transforms.ToTensor()
        self.transform_gray = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # calculate the whole number of each class
        for i, classname in enumerate(self.classlist):
            for j, imgname in enumerate(imglist):
                if imgname.split('/')[-2] == classname:
                    self.imgroot[i].append(imgname)
        # raise error
        for i in range(len(imglist)):
            if self.iter_frames > len(imglist[i]):
                raise Exception("Your given iter_frames is too big for this training set!")

    def get_lab(self, imgpath):                                         # for colorization task
        img = Image.open(imgpath)                                       # read one image
        # pre-processing, let all the images are in RGB color space
        img = img.resize((256, 256), Image.ANTIALIAS).convert('RGB')    # PIL Image RGB: R [0, 255], G [0, 255], B [0, 255], order [H, W, C]
        img = np.array(img)                                             # numpy RGB: R [0, 255], G [0, 255], B [0, 255], order [H, W, C]
        # convert RGB to Lab, finally get Tensor
        img = color.rgb2lab(img).astype(np.float32)                     # skimage Lab: L [0, 100], a [-128, 127], b [-128, 127], order [H, W, C]
        img = self.totensor(img)                                        # Tensor Lab: L [0, 100], a [-128, 127], b [-128, 127], order [C, H, W]
        # normaization
        l = img[[0], ...] / 50 - 1.0                                    # L, normalized to [-1, 1]
        ab = img[[1, 2], ...] / 110.0                                   # a and b, normalized to [-1, 1], approximately
        return l, ab

    def get_rgb(self, imgpath):
        img = Image.open(imgpath)                                       # read one image
        # pre-processing, let all the images are in RGB color space
        img = img.resize((192, 192), Image.ANTIALIAS).convert('RGB')    # PIL Image RGB: R [0, 255], G [0, 255], B [0, 255], order [H, W, C]
        l = img.convert('L').convert('RGB')                             # PIL Image L: L [0, 255], order [H, W]
        # normalization
        l = self.transform_gray(l)                                      # L, normalized to [-1, 1]
        rgb = self.transform_rgb(img)                                   # rgb, normalized to [-1, 1]
        return l, rgb

    def __getitem__(self, index):
        # choose a category of dataset, it is fair for each dataset to be chosen
        N = len(self.imgroot[index])
        # pre-define the starting frame index in 0 ~ N - opt.iter_frames
        T = random.randint(0, N - self.iter_frames)
        # sample from T to T + opt.iter_frames
        in_part = []
        out_part = []
        for i in range(T, T + self.iter_frames):
            imgpath = self.baseroot + '/' + self.imgroot[index][i]      # path of one image
            if self.task == 'colorization':
                l, ab = self.get_rgb(imgpath)
                in_part.append(l)
                out_part.append(ab)
        return in_part, out_part
    
    def __len__(self):
        return len(self.classlist)


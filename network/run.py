#!/usr/bin/env python

import torch
import torchvision
import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import cv2

from pwcnet import *


##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:3])) >= 41) # requires at least pytorch version 0.4.1

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performancePwcEstimate

##########################################################

arguments_strModel = 'default'
arguments_strFirst = './data/first.png'
arguments_strSecond = './images/second.png'
arguments_strOut = './out.flo'

real_folder = 'data/comic'
style_folder = 'data/bike-packing'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use
	if strOption == '--first' and strArgument != '': arguments_strFirst = strArgument # path to the first frame
	if strOption == '--second' and strArgument != '': arguments_strSecond = strArgument # path to the second frame
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

moduleNetwork = PwcNet().cuda().eval()

##########################################################

def flow_warp(x, flo):
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = torch.nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = torch.nn.functional.grid_sample(mask, vgrid)
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output*mask
# end

##########################################################
def flow_visible(im1, flow):
    flow = flow / 40 +0.5
    # print(flow)
    flow = flow.transpose(1, 2, 0)
    hsv = numpy.zeros(im1.shape, dtype=numpy.uint8)
    hsv[..., 1] = 0
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    ang = ang * 180 / numpy.pi / 2
    hsv[..., 0] = cv2.normalize(ang, None, 0, 180, cv2.NORM_MINMAX)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('./data/comic-result/flow_36250.jpg', bgr)
#end

############################################################

def preprocessedImg(img_path, grayM=False, resizeM=False):
    PILImg = PIL.Image.open(img_path)
    if resizeM:
        PILImg = PILImg.resize(resizeM, PIL.Image.ANTIALIAS)
    PILImgNp = numpy.array(PILImg).astype(numpy.float32)
    if grayM:
        grayArr = numpy.zeros((PILImgNp.shape[0], PILImgNp.shape[1]), dtype=numpy.float32)
        grayArr[:, :] = PILImgNp[:, :, 0]*0.299 + PILImgNp[:, :, 1] * 0.587 + PILImgNp[:, :, 2]*0.114
        PILImgNp[:, :, 0] = PILImgNp[:, :, 1] = PILImgNp[:, :, 2] = grayArr[:, :]
        PILImg = torchvision.transforms.functional.to_grayscale(PILImg, 3)
    return [PILImg, torch.FloatTensor(PILImgNp[:, :, ::-1].transpose(2, 0, 1) * (1.0 / 255.0))]

if __name__ == '__main__':
    #read the image
    f1 = '36250.jpg'
    f2 = '36251.jpg'
    pathFirst = os.path.join(real_folder, f1)
    pathSecond = os.path.join(real_folder, f2)
    # pathFirstS = os.path.join(style_folder, f1)
    # pathSecondS = os.path.join(style_folder, f2)
    PILImgFirst, tensorFirst = preprocessedImg(pathFirst, False)
    PILImgSecond, tensorSecond = preprocessedImg(pathSecond, False)
    #transform the image to gray mode and Keep three channels
    # PILImgFirstGray, tensorFirstGray = preprocessedImg(pathFirst, True)
    # PILImgSecondGray, tensorSecondGray = preprocessedImg(pathSecond, True)

    # PILImgFirstS, tensorFirstS = preprocessedImg(pathFirstS, False)
    # PILImgSecondS, tensorSecondS = preprocessedImg(pathSecondS, False)
    #change the type of image
    # tensorFirst = torch.FloatTensor(numpy.array(PILImgFirst)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
    # tensorSecond = torch.FloatTensor(numpy.array(PILImgSecond)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))

    tensorOutput = PwcEstimate(tensorFirst, tensorSecond)
    # tensorOutputS = estimate(tensorFirstS, tensorSecondS)
    # tensorOutputGray = estimate(tensorFirstGray, tensorSecondGray)
    # diff1 = numpy.sqrt(numpy.sum(numpy.square(numpy.array(tensorOutput) - numpy.array(tensorOutputGray))))/(tensorOutput.shape[1])/(tensorOutput.shape[2])
    # diff2 = numpy.sqrt(numpy.sum(numpy.square(numpy.array(tensorOutput) - numpy.array(tensorOutputS))))/(tensorOutput.shape[1])/(tensorOutput.shape[2])
    # print('the difference between real and gray image is %f, and the difference between real and generated image is %f, the second difference is %f of the first' %(diff1, diff2, diff2/diff1))
    flow_visible(numpy.array(PILImgFirst), numpy.array(tensorOutput))
    # objectOutput = open(arguments_strOut, 'wb')
    #
    # numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objectOutput)
    # numpy.array([ tensorOutput.size(2), tensorOutput.size(1) ], numpy.int32).tofile(objectOutput)
    # numpy.array(tensorOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objectOutput)
    #
    # objectOutput.close()
    print(tensorFirst.shape)
    tensorSecond4D = torch.unsqueeze(tensorSecond, 0).cuda()
    print(tensorSecond4D.shape)
    # print(tensorOutput)
    tensorOutput4D = (torch.unsqueeze(tensorOutput, 0)).cuda()
    print(tensorOutput4D.shape)
    # tensorOutput4D = - tensorOutput.reshape([1, 2, 436, 1024]).cuda()
    # print(tensorOutput4D)
    # tensorOutput4D = torch.FloatTensor(numpy.zeros((1, 2, 436, 1024))).cuda()
    # print(tensorOutput4D.shape)
    # tensorOutput4D[:, 0, :, :] = -0.01
    # tensorOutput4D[:, 1, :, :] = -0.1
    # warpResult = flow_warp(tensorFirst4D, tensorOutput4D)
    warpResult = Backward(tensorSecond4D, tensorOutput4D)
    warpResult = torch.squeeze(warpResult, 0).permute(1, 2, 0).cpu()
    print(warpResult)
    # warpResult = warpResult.reshape([3, 436, 1024]).permute(1, 2, 0).cpu()
    warpResult = warpResult * 255
    imageResult = numpy.uint8(warpResult.numpy())[:, :, ::-1]
    imageResult = PIL.Image.fromarray(imageResult).convert('RGB')
    imageResult.save("./data/comic-result/36250_warp.jpg")
    #
    # tensorOutput = tensorOutput / 40 + 0.5
    # tensorOutputx = tensorOutput[[0], :, :].mul(tensorOutput[[0], :, :])
    # tensorOutputy = tensorOutput[[1], :, :].mul(tensorOutput[[1], :, :])
    # tensorOne = torch.sqrt(tensorOutputx + tensorOutputy) / 1.5
    # tensorOutput = torch.cat((tensorOne, tensorOutput), 0)
    # tensorOutput = tensorOutput.permute(1, 2, 0) * 255
    # tensorOutput = tensorOutput.numpy().astype(numpy.uint8)
    # tensorOutput = PIL.Image.fromarray(tensorOutput)
    # tensorOutput.save('./images/00055_flow.jpg')


# end



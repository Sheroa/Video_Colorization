import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from skimage import color
import os
import utils

# ----------------------------------------
#                 Testing
# ----------------------------------------

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    numList = []
    for root, dirs, files in os.walk(path):
        files.sort()
        for filespath in files:
            ret.append(filespath)
    return ret
def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

def test(rgb, colornet):
    out_rgb = colornet(rgb)
    out_rgb = out_rgb.cpu().detach().numpy().reshape([3, 256, 256])
    out_rgb = out_rgb.transpose(1, 2, 0)
    out_rgb = (out_rgb * 0.5 + 0.5) * 255
    out_rgb = out_rgb.astype(np.uint8)
    return out_rgb

def getImage(root):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    img = Image.open(root).convert('RGB')
    #img = img.crop((256, 0, 512, 256))
    rgb = img.resize((256, 256), Image.ANTIALIAS)
    rgb = transform(rgb)
    rgb = rgb.reshape([1, 3, 256, 256]).cuda()
    return rgb

def comparison(root, colornet):
    # Read raw image
    img = Image.open(root).convert('RGB')
    real = img.crop((0, 0, 256, 256))
    real = real.resize((256, 256), Image.ANTIALIAS)
    real = np.array(real)
    # Forward propagation
    torchimg = getImage(root)
    out_rgb = test(torchimg, colornet)
    # Show
    out_rgb = np.concatenate((out_rgb, real), axis = 1)
    img_rgb = Image.fromarray(out_rgb)
    img_rgb.show()
    return img_rgb

def colorization(root, colornet):
    # Forward propagation
    torchimg = getImage(root)
    out_rgb = test(torchimg, colornet)
    # Show
    img_rgb = Image.fromarray(out_rgb)
    img_rgb.show()
    return img_rgb

def generation(baseroot, saveroot, imglist, colornet):
    for i in range(len(imglist)):
		# Read raw image
        readname = baseroot + imglist[i]
        print(readname)
        # Forward propagation
        torchimg = getImage(readname)
        out_rgb = test(torchimg, colornet)
        # Save
        img_rgb = Image.fromarray(out_rgb)
        savename = saveroot + imglist[i]
        img_rgb.save(savename)
    print('Done!')

def video_test(x_t, y_t_last, lstm_state, colornet):
    out_rgb, lstm_state = colornet(x_t, y_t_last, lstm_state)
    out_rgb = out_rgb.cpu().detach().numpy().reshape([3, 256, 256])
    out_rgb = out_rgb.transpose(1, 2, 0)
    out_rgb = (out_rgb * 0.5 + 0.5) * 255
    out_rgb = out_rgb.astype(np.uint8)
    return out_rgb, lstm_state

def video_generation(baseroot, saveroot, imglist, colornet):
    y_t_last = torch.zeros(1, 3, 256, 256).cuda()
    lstm_state = None
    for i in range(len(imglist)):
        #Read raw image
        readname = baseroot + imglist[i]
        x_t = getImage(readname)
        #Forword propagation
        y_t_last, lstm_state = colornet(x_t, y_t_last, lstm_state)
        lstm_state = utils.repackage_hidden(lstm_state)
        # Save
        y_t_last = y_t_last.detach()
        out_rgb = y_t_last.cpu()
        out_rgb = out_rgb.numpy().reshape([3, 256, 256]).transpose(1, 2, 0)
        out_rgb = (out_rgb * 0.5 + 0.5) * 255
        out_rgb = out_rgb.astype(np.uint8)
        img_rgb = Image.fromarray(out_rgb)
        numName, typeName = imglist[i].split('.')
        savename = saveroot + numName + '_800_old.'+ typeName
        img_rgb.save(savename)

if __name__ == "__main__":

    # Define the basic variables
    root = './'
    colornet = torch.load('./models/batchSize1_startChanne32/Pre_colorization_epoch800_bs1.pth')
    # Define generation variables
    txtname = './Varidation/names.txt'
    # imglist = utils.text_readlines(txtname)
    baseroot = './Varidation/kite-surf/'
    saveroot = './Varidation/result_kite-surf_epoch800_old/'
    imglist = get_jpgs('./Varidation/kite-surf')
    # Choose a task:
    choice = 'video_generation'
    save = True

    # comparison: Compare the colorization output and ground truth
    # colorization: Show the colorization as original size
    # generation: Generate colorization results given a folder
    if choice == 'comparison':
        img_rgb = comparison(root, colornet)
        if save:
            imgname = root.split('/')[-1]
            img_rgb.save('./' + imgname)
    if choice == 'colorization':
        img_rgb = colorization(root, colornet)
        if save:
            imgname = root.split('/')[-1]
            img_rgb.save('./' + imgname)
    if choice == 'generation':
        generation(baseroot, saveroot, imglist, colornet)
    if choice == 'video_generation':
        video_generation(baseroot, saveroot, imglist, colornet)
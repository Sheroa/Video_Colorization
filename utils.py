import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import os
import networks
import networks.basenet as basenet
import networks.pwcnet as pwcnet
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

# There are many functions:
# ----------------------------------------
# 1. create_generator:
# In: opt, init_type, init_gain
# Parameters: init type and gain, we highly recommend that Gaussian init with standard deviation of 0.02
# Out: colorizationnet
# ----------------------------------------
# 2. create_discriminator:
# In: opt, init_type, init_gain
# Parameters: init type and gain, we highly recommend that Gaussian init with standard deviation of 0.02
# Out: discriminator_coarse_color, discriminator_coarse_sal, discriminator_fine_color, discriminator_fine_sal
# ----------------------------------------
# 3. create_discriminator:
# In: opt, init_type, init_gain
# Parameters: init type and gain, we highly recommend that Gaussian init with standard deviation of 0.02
# Out: discriminator_coarse_color, discriminator_coarse_sal, discriminator_fine_color, discriminator_fine_sal
# ----------------------------------------
# 4. create_perceptualnet:
# In: None
# Parameters: None
# Out: perceptualnet
# ----------------------------------------
# 5. repackage_hidden:
# In: dictionary the contains Tensors
# Parameters: None
# Out: detached Tensors
# ----------------------------------------
# 6. text_readlines:
# In: a str nominating the a txt
# Parameters: None
# Out: list
# ----------------------------------------
# 7. text_save:
# In: content, a str nominating the a txt
# Parameters: None
# Out: txt
# ----------------------------------------
# 8. text_np_save:
# In: content, a str nominating the a txt
# Parameters: None
# Out: txt
# ----------------------------------------
# 9. get_files
# In: path
# Out: txt
# ----------------------------------------
# 10. get_jpgs
# In: path
# Out: txt
# ----------------------------------------
# 11. get_dirs
# In: path
# Out: txt
# ----------------------------------------
# 12. get_relative_dirs
# In: path
# Out: txt
# ----------------------------------------


class SubsetSeSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)


def create_dataloader(dataset, opt):
    #generate random index
    indices = np.random.permutation(len(dataset))
    indices = np.tile(indices, opt.batch_size)
    #generate data sampler and loader
    data_sampler = SubsetSeSampler(indices)
    data_loader = DataLoader(dataset=dataset, num_workers=opt.num_workers, batch_size = opt.batch_size, sampler=data_sampler, pin_memory=True)
    return data_loader

def create_generator(opt):
    if opt.pre_train:
        # Initialize the network
        generator = basenet.ConvLSTMGenerator_1in(opt)
        # Init the network
        networks.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Generator is created!')
    else:
        # Initialize the network
        generator = basenet.ConvLSTMGenerator_1in(opt)
        # Load a pre-trained network
        pretrained_net = torch.load(opt.load_name + '.pth')
        networks.load_dict(generator, pretrained_net)
        print('Generator is loaded!')
    return generator
    
def create_discriminator(opt):
    # Initialize the network
    discriminator = basenet.PatchDiscriminator70(opt)
    # Init the network
    networks.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Discriminators is created!')
    return discriminator

def create_pwcnet(opt):
    # Initialize the network
    flownet = pwcnet.PWCNet().eval()
    # Load a pre-trained network
    data = torch.load(opt.pwcnet_path)
    if 'state_dict' in data.keys():
        flownet.load_state_dict(data['state_dict'])
    else:
        flownet.load_state_dict(data)
    print('PWCNet is loaded!')
    # It does not gradient
    for param in flownet.parameters():
        param.requires_grad = False
    return flownet

def repackage_hidden(h):
    # Wraps hidden states in new Variables, to detach them from their history
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

def text_np_save(content, filename):
    np_content = np.array(content)
    np.savetxt(filename, np_content)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files: 
            ret.append(os.path.join(root,filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

def get_dirs(path):
    #read a folder, return a list of names of child folders
    ret = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            if root == path:
                ret.append(name)
    return ret

def get_relative_dirs(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            a = os.path.join(root,filespath)
            a = a.split('\\')[-2] + '\\' + a.split('\\')[-1]
            ret.append(a)
    return ret

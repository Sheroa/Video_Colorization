import time
import datetime
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import dataset
import utils
import sys

import networks.pwcnet as pwcnet


def Pre_train_single(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    criterion_MSE = torch.nn.MSELoss().cuda()

    # Initialize Generator
    generatorNet = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)

    # To device
    if opt.multi_gpu:
        generatorNet = nn.DataParallel(generatorNet)
        generatorNet = generatorNet.cuda()
        discriminator = nn.DataParallel(discriminator)
        discriminator = discriminator.cuda()
    else:
        discriminator = discriminator.cuda()
        generatorNet = generatorNet.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generatorNet.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator.module, 'Pre_%s_epoch%d_bs%d.pth' % (opt.task, epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator.module, 'Pre_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator, 'Pre_%s_epoch%d_bs%d.pth' % (opt.task, epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator, 'Pre_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the class list
    imglist = utils.text_readlines('ILSVRC2012_train_sal_name.txt')[:1272480]

    # Define the dataset
    trainset = dataset.ColorizationDataset(opt, imglist)
    print('The overall number of classes:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # Count start time
    prev_time = time.time()
    
    # For loop training
    # For loop training
    for epoch in range(opt.epochs):
        for iteration, (x_t, y_t) in enumerate(dataloader):
            
            # Train Generator
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            lstm_state = None

            x_t = x_t.cuda()
            y_t = y_t.cuda()
            
            valid = Tensor(np.ones((x_t.shape[0], 1, 30, 30)))
            fake = Tensor(np.zeros((x_t.shape[0], 1, 30, 30)))

            p_t_last = torch.zeros(opt.batch_size, opt.out_channels, opt.resize_h, opt.resize_w).cuda()

            # Train Discriminator
            # Generator output
            p_t, lstm_state = generatorNet(x_t, p_t_last, lstm_state)

            # Fake samples
            fake_scalar = discriminator(x_t, p_t.detach())
            loss_fake = criterion_MSE(fake_scalar, fake)
            # True samples
            true_scalar = discriminator(x_t, y_t)
            loss_true = criterion_MSE(true_scalar, valid)
            # Overall Loss and optimize
            loss_D = 0.5 * (loss_fake + loss_true)
    
            # Train Generator
            # GAN Loss
            fake_scalar = discriminator(x_t, p_t)
            loss_G = criterion_MSE(fake_scalar, valid)
            
            # Pixel-level loss
            loss_L1 = criterion_L1(p_t, y_t)


            # Overall Loss and optimize
            loss = loss_L1 + opt.lambda_gan * loss_G
            loss.backward()
            loss_D.backward()
            optimizer_G.step()
            optimizer_D.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + iteration
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.4f] [G Loss: %.4f] [D Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, iteration, len(dataloader), loss_L1.item(), loss_G.item(), loss_D.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generatorNet)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D)


def Train_GAN(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    criterion_MSE = torch.nn.MSELoss().cuda()

    # Initialize Generator
    generatorNet = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    flownet = utils.create_pwcnet(opt)

    # To device
    if opt.multi_gpu:
        generatorNet = nn.DataParallel(generatorNet)
        generatorNet = generatorNet.cuda()
        discriminator = nn.DataParallel(discriminator)
        discriminator = discriminator.cuda()
        flownet = nn.DataParallel(flownet)
        flownet = flownet.cuda()
    else:
        discriminator = discriminator.cuda()
        generatorNet = generatorNet.cuda()
        flownet = flownet.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generatorNet.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator.module, 'Pre_%s_epoch%d_bs%d_Gan%d_os%d_ol%d.pth' % (opt.task, epoch, opt.batch_size, opt.lambda_gan, opt.lambda_flow, opt.lambda_flow_long))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator.module, 'Pre_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator, 'Pre_%s_epoch%d_bs%d_GAN%d_os%d_ol%d.pth' % (opt.task, epoch, opt.batch_size, opt.lambda_gan, opt.lambda_flow, opt.lambda_flow_long))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator, 'Pre_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the class list
    imglist = utils.text_readlines('videocolor_linux.txt')
    classlist = utils.get_dirs(opt.baseroot)
    '''
    imgnumber = len(imglist) - (len(imglist) % opt.batch_size)
    imglist = imglist[:imgnumber]
    '''

    # Define the dataset
    trainset = dataset.MultiFramesDataset(opt, imglist, classlist)
    print('The overall number of classes:', len(trainset))

    # Define the dataloader
    dataloader = utils.create_dataloader(trainset, opt)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # Count start time
    prev_time = time.time()
    
    # For loop training
    for epoch in range(opt.epochs):
        for iteration, (in_part, out_part) in enumerate(dataloader):
            
            # Train Generator
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            lstm_state = None
            loss_flow = 0
            loss_flow_long = 0
            loss_L1 = 0
            loss_D = 0
            loss_G = 0

            x_0 = in_part[0].cuda()
            p_t_0 = in_part[0].cuda()
            # Adversarial ground truth
            valid = Tensor(np.ones((in_part[0].shape[0], 1, 30, 30)))
            fake = Tensor(np.zeros((in_part[0].shape[0], 1, 30, 30)))

            for iter_frame in range(opt.iter_frames):
                # Read data
                x_t = in_part[iter_frame].cuda()
                y_t = out_part[iter_frame].cuda()
                
                # Initialize the second input and compute flow loss
                if iter_frame == 0:
                    p_t_last = torch.zeros(opt.batch_size, opt.out_channels, opt.resize_h, opt.resize_w).cuda()
                elif iter_frame == 1:
                    x_t_last = in_part[iter_frame - 1].cuda()
                    p_t_last = p_t.detach()
                    p_t_0 = p_t.detach()
                    p_t_last.requires_grad = False
                    p_t_0.requires_grad = False
                    # o_t_last_2_t range is [-20, +20]
                    o_t_last_2_t = pwcnet.PWCEstimate(flownet, x_t, x_t_last)
                    x_t_warp = pwcnet.PWCNetBackward((x_t_last + 1) / 2, o_t_last_2_t)
                    # y_t_warp range is [0, 1]
                    p_t_warp = pwcnet.PWCNetBackward((p_t_last + 1) / 2, o_t_last_2_t)
                else:
                    x_t_last = in_part[iter_frame - 1].cuda()
                    p_t_last = p_t.detach()
                    p_t_last.requires_grad = False
                    # o_t_last_2_t o_t_first_2_t range is [-20, +20]
                    o_t_last_2_t = pwcnet.PWCEstimate(flownet, x_t, x_t_last)
                    o_t_first_2_t = pwcnet.PWCEstimate(flownet,x_t, x_0)
                    # y_t_warp, y_t_warp_long range is [0, 1]
                    x_t_warp = pwcnet.PWCNetBackward((x_t_last + 1) / 2, o_t_last_2_t)
                    p_t_warp = pwcnet.PWCNetBackward((p_t_last + 1) / 2, o_t_last_2_t)
                    x_t_warp_long = pwcnet.PWCNetBackward((x_0 + 1) / 2, o_t_first_2_t)
                    p_t_warp_long = pwcnet.PWCNetBackward((p_t_0 + 1) / 2, o_t_first_2_t)
                # Generator output
                p_t, lstm_state = generatorNet(x_t, p_t_last, lstm_state)
                lstm_state = utils.repackage_hidden(lstm_state)
                if iter_frame == 1:
                    mask_flow = torch.exp( -opt.mask_para * torch.sum((x_t + 1) / 2 - x_t_warp, dim=1).pow(2) ).unsqueeze(1)
                    loss_flow += criterion_L1(mask_flow * (p_t + 1) / 2, mask_flow * p_t_warp)
                elif iter_frame > 1:
                    mask_flow = torch.exp( -opt.mask_para * torch.sum((x_t + 1) / 2 - x_t_warp, dim=1).pow(2) ).unsqueeze(1)
                    loss_flow += criterion_L1(mask_flow * (p_t + 1) / 2, mask_flow * p_t_warp)
                    mask_flow_long = torch.exp( -opt.mask_para * torch.sum((x_t + 1) / 2 - x_t_warp_long, dim=1).pow(2) ).unsqueeze(1)
                    loss_flow_long += criterion_L1(mask_flow_long * (p_t + 1) / 2, mask_flow_long * p_t_warp_long)
                # Fake samples
                fake_scalar = discriminator(x_t, p_t.detach())
                loss_fake = criterion_MSE(fake_scalar, fake)
                # True samples
                true_scalar = discriminator(x_t, y_t)
                loss_true = criterion_MSE(true_scalar, valid)
                # Train Discriminator
                loss_D += 0.5 * (loss_fake + loss_true)
                # Train Generator
                # GAN Loss
                fake_scalar = discriminator(x_t, p_t)
                loss_G += criterion_MSE(fake_scalar, valid)
                
                # Pixel-level loss
                loss_L1 += criterion_L1(p_t, y_t)

            # Overall Loss and optimize
            loss = loss_L1 + opt.lambda_flow * loss_flow + opt.lambda_flow_long * loss_flow_long + opt.lambda_gan * loss_G
            loss.backward()
            loss_D.backward()
            optimizer_G.step()
            optimizer_D.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + iteration
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.4f] [Flow Loss Short: %.8f] [Flow Loss Long: %.8f] [G Loss: %.4f] [D Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, iteration, len(dataloader), loss_L1.item(), loss_flow.item(), loss_flow_long.item(), loss_G.item(), loss_D.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generatorNet)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D)
    

def Train_No_GAN(opt):    # w / o GAN
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()

    # Initialize Generator
    generatorNet = utils.create_generator(opt)
    flownet = utils.create_pwcnet(opt)

    # To device
    if opt.multi_gpu:
        generatorNet = nn.DataParallel(generatorNet)
        generatorNet = generatorNet.cuda()
        flownet = nn.DataParallel(flownet)
        flownet = flownet.cuda()
    else:
        generatorNet = generatorNet.cuda()
        flownet = flownet.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generatorNet.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator.module, 'Pre_%s_epoch%d_bs%d.pth' % (opt.task, epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator.module, 'Pre_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator, 'Pre_%s_epoch%d_bs%d.pth' % (opt.task, epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator, 'Pre_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the class list
    imglist = utils.text_readlines('videocolor_linux.txt')
    classlist = utils.get_dirs(opt.baseroot)
    '''
    imgnumber = len(imglist) - (len(imglist) % opt.batch_size)
    imglist = imglist[:imgnumber]
    '''

    # Define the dataset
    trainset = dataset.MultiFramesDataset(opt, imglist, classlist)
    print('The overall number of classes:', len(trainset))

    # Define the dataloader
    dataloader = utils.create_dataloader(trainset, opt)
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    # For loop training
    for epoch in range(opt.epochs):
        for iteration, (in_part, out_part) in enumerate(dataloader):
            
            # Train Generator
            optimizer_G.zero_grad()

            lstm_state = None
            loss_flow = 0
            loss_flow_long = 0
            loss_L1 = 0

            x_0 = in_part[0].cuda()
            p_t_0 = in_part[0].cuda()

            for iter_frame in range(opt.iter_frames):
                # Read data
                x_t = in_part[iter_frame].cuda()
                y_t = out_part[iter_frame].cuda()
                
                # Initialize the second input and compute flow loss
                if iter_frame == 0:
                    p_t_last = torch.zeros(opt.batch_size, opt.out_channels, opt.resize_h, opt.resize_w).cuda()
                elif iter_frame == 1:
                    x_t_last = in_part[iter_frame - 1].cuda()
                    p_t_last = p_t.detach()
                    p_t_0 = p_t.detach()
                    p_t_last.requires_grad = False
                    p_t_0.requires_grad = False
                    # o_t_last_2_t range is [-20, +20]
                    o_t_last_2_t = pwcnet.PWCEstimate(flownet, x_t, x_t_last)
                    x_t_warp = pwcnet.PWCNetBackward((x_t_last + 1) / 2, o_t_last_2_t)
                    # y_t_warp range is [0, 1]
                    p_t_warp = pwcnet.PWCNetBackward((p_t_last + 1) / 2, o_t_last_2_t)
                else:
                    x_t_last = in_part[iter_frame - 1].cuda()
                    p_t_last = p_t.detach()
                    p_t_last.requires_grad = False
                    # o_t_last_2_t o_t_first_2_t range is [-20, +20]
                    o_t_last_2_t = pwcnet.PWCEstimate(flownet, x_t, x_t_last)
                    o_t_first_2_t = pwcnet.PWCEstimate(flownet,x_t, x_0)
                    # y_t_warp, y_t_warp_long range is [0, 1]
                    x_t_warp = pwcnet.PWCNetBackward((x_t_last + 1) / 2, o_t_last_2_t)
                    p_t_warp = pwcnet.PWCNetBackward((p_t_last + 1) / 2, o_t_last_2_t)
                    x_t_warp_long = pwcnet.PWCNetBackward((x_0 + 1) / 2, o_t_first_2_t)
                    p_t_warp_long = pwcnet.PWCNetBackward((p_t_0 + 1) / 2, o_t_first_2_t)
                # Generator output
                p_t, lstm_state = generatorNet(x_t, p_t_last, lstm_state)
                lstm_state = utils.repackage_hidden(lstm_state)
                if iter_frame == 1:
                    mask_flow = torch.exp( -opt.mask_para * torch.sum((x_t + 1) / 2 - x_t_warp, dim=1).pow(2) ).unsqueeze(1)
                    loss_flow += criterion_L1(mask_flow * (p_t + 1) / 2, mask_flow * p_t_warp)
                elif iter_frame > 1:
                    mask_flow = torch.exp( -opt.mask_para * torch.sum((x_t + 1) / 2 - x_t_warp, dim=1).pow(2) ).unsqueeze(1)
                    loss_flow += criterion_L1(mask_flow * (p_t + 1) / 2, mask_flow * p_t_warp)
                    mask_flow_long = torch.exp( -opt.mask_para * torch.sum((x_t + 1) / 2 - x_t_warp_long, dim=1).pow(2) ).unsqueeze(1)
                    loss_flow_long += criterion_L1(mask_flow_long * (p_t + 1) / 2, mask_flow_long * p_t_warp_long)
                
                # Pixel-level loss
                loss_L1 += criterion_L1(p_t, y_t)

            # Overall Loss and optimize
            loss = loss_L1 + opt.lambda_flow * loss_flow + opt.lambda_flow_long * loss_flow_long
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + iteration
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.4f] [Flow Loss Short: %.8f] [Flow Loss Long: %.8f] Time_left: %s" %
                ((epoch + 1), opt.epochs, iteration, len(dataloader), loss_L1.item(), loss_flow.item(), loss_flow_long.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generatorNet)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
            
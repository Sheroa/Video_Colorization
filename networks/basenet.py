import torch
import torch.nn as nn
from .modules import *

# ----------------------------------------
#                Generator
# ----------------------------------------
# ConvLSTMGenerator contains 2 Auto-Encoders
class ConvLSTMGenerator_1in(nn.Module):
    def __init__(self, opt):
        super(ConvLSTMGenerator_1in, self).__init__()
        # The generator is U shaped
        # It means: input -> downsample -> upsample -> output
        # Encoder: a is for grayscale input and b & c is for other frames input
        # b: last frame y_t-1
        # c1: last frame x_t-1; c2: second last frame x_t-2; d1: next frame x_t+1; d2: second next frame x_t+2              Pendding!!!
        # Seperate Encoder layers
        self.E1_a = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none')
        self.E1_b = Conv2dLayer(opt.out_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none')
        self.E2_a = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E2_b = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E3_a = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E3_b = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        # Middle Encoder layers
        self.R1 = ResConv2dLayer(opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.R2 = ResConv2dLayer(opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.R3 = ResConv2dLayer(opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.R4 = ResConv2dLayer(opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.R5 = ResConv2dLayer(opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.ConvLSTM = ConvLSTM2d(opt.start_channels * 4, opt.start_channels * 4)
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 1, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D4 = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 7, 1, 3, pad_type = opt.pad, activation = 'tanh', norm = 'none')

    def forward(self, a, b, prev_state):
        # U-Net generator with skip connections from encoder to decoder
        # Note that: a is for grayscale input and b is for last frame input
        # Seperate Encoder layers
        E1_a = self.E1_a(a)                                     # out: batch * 64 * 256 * 256
        E1_b = self.E1_b(b)                                     # out: batch * 64 * 256 * 256
        E2_a = self.E2_a(E1_a)                                  # out: batch * 128 * 128 * 128
        E2_b = self.E2_b(E1_b)                                  # out: batch * 128 * 128 * 128
        E3_a = self.E3_a(E2_a)                                  # out: batch * 128 * 64 * 64
        E3_b = self.E3_b(E2_b)                                  # out: batch * 128 * 64 * 64
        E3 = torch.cat((E3_a, E3_b), 1)                         # out: batch * 256 * 64 * 64
        E4 = self.E4(E3)                                        # out: batch * 256 * 32 * 32
        # Middle Encoder layers
        E4 = self.R1(E4)                                        # out: batch * 256 * 32 * 32
        E4 = self.R2(E4)                                        # out: batch * 256 * 32 * 32
        E4 = self.R3(E4)                                        # out: batch * 256 * 32 * 32
        E4 = self.R4(E4)                                        # out: batch * 256 * 32 * 32
        E4 = self.R5(E4)                                        # out: batch * 256 * 32 * 32
        state = self.ConvLSTM(E4, prev_state)                   # out[0] | hidden: batch * 256 * 32 * 32; out[1] | cell: batch * 256 * 32 * 32
        # Decode the LSTM output
        D1 = self.D1(state[0])                                  # out: batch * 128 * 64 * 64
        C1 = torch.cat((D1, E3_a), 1)                           # out: batch * 256 * 64 * 64
        D2 = self.D2(C1)                                        # out: batch * 128 * 128 * 128
        C2 = torch.cat((D2, E2_a), 1)                           # out: batch * 256 * 128 * 128
        D3 = self.D3(C2)                                        # out: batch * 64 * 256 * 256
        C3 = torch.cat((D3, E1_a), 1)                           # out: batch * 128 * 256 * 256
        D4 = self.D4(C3)                                        # out: batch * 256 * 128 * 128

        return D4, state                                        # out[0]: the system output; out[1]: the current state (undetached)

# ----------------------------------------
#               Discriminator
# ----------------------------------------
# PatchDiscriminator70: PatchGAN discriminator for Pix2Pix
# Usage: Initialize PatchGAN in training code like:
#        discriminator = PatchDiscriminator70()
# This is a kind of PatchGAN. Patch is implied in the output. This is 70 * 70 PatchGAN
class PatchDiscriminator70(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator70, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.in_channels + opt.out_channels, 64, 4, 2, 1, pad_type = opt.pad, norm = 'none')
        self.block2 = Conv2dLayer(64, 128, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.block3 = Conv2dLayer(128, 256, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = Conv2dLayer(256, 512, 4, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.final2 = Conv2dLayer(512, 1, 4, 1, 1, pad_type = opt.pad, norm = 'none', activation = 'none')

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        # img_A: grayscale input; img_B: ab embedding output
        x = torch.cat((img_A, img_B), 1)                        # out: batch * 3 * 256 * 256
        x = self.block1(x)                                      # out: batch * 64 * 128 * 128
        x = self.block2(x)                                      # out: batch * 128 * 64 * 64
        x = self.block3(x)                                      # out: batch * 256 * 32 * 32
        x = self.final1(x)                                      # out: batch * 512 * 31 * 31
        x = self.final2(x)                                      # out: batch * 1 * 30 * 30
        return x

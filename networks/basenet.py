import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
#from .modules import *

class ConvLSTM2d(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3):
        super(ConvLSTM2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding)

    def forward(self, input_, prev_state = None):

        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)).cuda(),
                Variable(torch.zeros(state_size)).cuda()
            )

        # prev_state has two components
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


# ----------------------------------------
#           Spectral Norm Block
# ----------------------------------------
def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

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
        self.E1_a = nn.Sequential(
            nn.ReflectionPad2d(3),
            SpectralNorm(nn.Conv2d(opt.in_channels, opt.start_channels, 7, 1, padding = 0, dilation = 1)),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.E1_b = nn.Sequential(
            nn.ReflectionPad2d(3),
            SpectralNorm(nn.Conv2d(opt.in_channels, opt.start_channels, 7, 1, padding = 0, dilation = 1)),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.E2_a = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(opt.start_channels, opt.start_channels * 2, 4, 2, padding = 0, dilation = 1)),
            nn.InstanceNorm2d(opt.start_channels * 2),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.E2_b = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(opt.start_channels, opt.start_channels * 2, 4, 2, padding = 0, dilation = 1)),
            nn.InstanceNorm2d(opt.start_channels * 2),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.E3_a = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(opt.start_channels * 2, opt.start_channels * 2, 4, 2, padding = 0, dilation = 1)),
            nn.InstanceNorm2d(opt.start_channels * 2),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.E3_b = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(opt.start_channels * 2, opt.start_channels * 2, 4, 2, padding = 0, dilation = 1)),
            nn.InstanceNorm2d(opt.start_channels * 2),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.E4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(opt.start_channels * 4, opt.start_channels * 4, 4, 2, padding = 0, dilation = 1)),
            nn.InstanceNorm2d(opt.start_channels * 2),
            nn.LeakyReLU(0.2, inplace = True)
        )
        # Middle Encoder layers
        self.R1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(opt.start_channels * 4, opt.start_channels * 4, 3, 1, padding = 0, dilation = 1)),
            nn.InstanceNorm2d(opt.start_channels * 2),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.R2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(opt.start_channels * 4, opt.start_channels * 4, 3, 1, padding = 0, dilation = 1)),
            nn.InstanceNorm2d(opt.start_channels * 2),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.R3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(opt.start_channels * 4, opt.start_channels * 4, 3, 1, padding = 0, dilation = 1)),
            nn.InstanceNorm2d(opt.start_channels * 2),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.R4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(opt.start_channels * 4, opt.start_channels * 4, 3, 1, padding = 0, dilation = 1)),
            nn.InstanceNorm2d(opt.start_channels * 2),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.R5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(opt.start_channels * 4, opt.start_channels * 4, 3, 1, padding = 0, dilation = 1)),
            nn.InstanceNorm2d(opt.start_channels * 2),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.ConvLSTM = ConvLSTM2d(opt.start_channels * 4, opt.start_channels * 4)
        # Decoder
        self.D1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(opt.start_channels * 4, opt.start_channels * 2, 3, 1, padding = 0, dilation = 1)),
            nn.InstanceNorm2d(opt.start_channels * 2),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.D2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(opt.start_channels * 4, opt.start_channels * 2, 3, 1, padding = 0, dilation = 1)),
            nn.InstanceNorm2d(opt.start_channels * 2),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.D3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(opt.start_channels * 4, opt.start_channels, 3, 1, padding = 0, dilation = 1)),
            nn.InstanceNorm2d(opt.start_channels * 2),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.D4 = nn.Sequential(
            nn.ReflectionPad2d(3),
            SpectralNorm(nn.Conv2d(opt.start_channels * 2, opt.out_channels, 7, 1, padding = 0, dilation = 1)),
            nn.Tanh()
        )

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
        E4_1 = self.R1(E4)                                      # out: batch * 256 * 32 * 32
        E4 = E4 + E4_1
        E4_2 = self.R2(E4)                                        # out: batch * 256 * 32 * 32
        E4 = E4 + E4_2
        E4_3 = self.R3(E4)                                        # out: batch * 256 * 32 * 32
        E4 = E4 + E4_3
        E4_4 = self.R4(E4)                                        # out: batch * 256 * 32 * 32
        E4 = E4 + E4_4
        E4_5 = self.R5(E4)                                        # out: batch * 256 * 32 * 32
        E4 = E4 + E4_5
        state = self.ConvLSTM(E4, prev_state)                   # out[0] | hidden: batch * 256 * 32 * 32; out[1] | cell: batch * 256 * 32 * 32
        # Decode the LSTM output
        D1_in = F.interpolate(state[0], scale_factor = 2, mode = 'nearest')
        D1 = self.D1(D1_in)                                  # out: batch * 128 * 64 * 64
        C1 = torch.cat((D1, E3_a), 1)                           # out: batch * 256 * 64 * 64
        C1 = F.interpolate(C1, scale_factor = 2, mode = 'nearest')
        D2 = self.D2(C1)                                        # out: batch * 128 * 128 * 128
        C2 = torch.cat((D2, E2_a), 1)                           # out: batch * 256 * 128 * 128
        C2 = F.interpolate(C2, scale_factor = 2, mode = 'nearest')
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
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(opt.in_channels + opt.out_channels, opt.start_channels, 4, 2, padding = 0, dilation = 1)),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.block2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(opt.start_channels, opt.start_channels * 2, 4, 2, padding = 0, dilation = 1)),
            nn.InstanceNorm2d(opt.start_channels * 2),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.block3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(opt.start_channels * 2, opt.start_channels * 4, 4, 2, padding = 0, dilation = 1)),
            nn.InstanceNorm2d(opt.start_channels * 4),
            nn.LeakyReLU(0.2, inplace = True)
        )
        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(opt.start_channels * 4, opt.start_channels * 4, 4, 1, padding = 0, dilation = 1)),
            nn.InstanceNorm2d(opt.start_channels * 4),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.final2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(opt.start_channels * 4, 1, 4, 1, padding = 0, dilation = 1))
        )

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

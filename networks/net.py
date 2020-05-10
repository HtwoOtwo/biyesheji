from __future__ import absolute_import, print_function
import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Variable
from networks.skip import skip
from networks.CNN import ConvBlock

class GradDataFitting(Module):
    def __init__(self):
        super(GradDataFitting, self).__init__()

    def forward(self, x, y, k, kt):
        n_size = x.size()[0]
        k_channel = k.size()[1]
        k_size = k.size()[2]
        padding = int(k_size / 2)
        x1 = x.transpose(1, 0)  # x1: C x N x H x W
        y1 = y.transpose(1, 0)
        # k: N x 1 x Ksize x Ksize
        k = k.repeat(n_size,1,1,1)
        kt = kt.repeat(n_size,1,1,1)
        vk = Variable(k)
        vkt = Variable(kt)
        kx_y = F.conv2d(x1, vk, padding=padding, groups=n_size) # Ax
        kx_y.sub_(y1) # Ax-y
        ktkx_kty = F.conv2d(kx_y, vkt, padding=padding, groups=n_size) # A^T(Ax-y)
        res = ktkx_kty.transpose(1, 0)  # h3: N x C x H x W
        res = kx_y.transpose(1, 0)  # h3: N x C x H x W
        return res

class OptimizerNet(Module):
    """Gradient descent based optimizer model"""
    def __init__(self, num_steps,
                 use_grad_adj=True,
                 use_grad_scaler=True,
                 use_reg=True,
                 share_parameter=True,
                 use_cuda=True):

        super(OptimizerNet, self).__init__()
        #
        input_depth = 1
        pad = 'reflection'
        self.num_steps = num_steps
        self.momen = 0.8
        self.grad_datafitting_cal = GradDataFitting()
        self.use_grad_adj = use_grad_adj
        self.use_reg = use_reg
        self.use_grad_scaler = use_grad_scaler
        self.share_parameter = share_parameter
        if self.share_parameter:
            if(self.use_reg):
                self.rnet = skip( input_depth, 1,
                num_channels_down = [128, 128, 128, 128, 128],
                num_channels_up   = [128, 128, 128, 128, 128],
                num_channels_skip = [16, 16, 16, 16, 16],
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
            if(self.use_grad_adj):
                self.fnet = skip( input_depth, 1,
                num_channels_down = [128, 128, 128, 128, 128],
                num_channels_up   = [128, 128, 128, 128, 128],
                num_channels_skip = [16, 16, 16, 16, 16],
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
            if(self.use_grad_scaler):
                self.dnet = ConvBlock()

    def forward(self, y, A, At):
        # init x
        xcurrent = y
        # xcurrent = self.init_cal(xcurrent, y, k, kt)

        # output_list = []
        # optimization init
        for i in range(self.num_steps):
            ## single step operation
            grad_loss = self.grad_datafitting_cal(xcurrent, y, A, At)

            # F()
            if(self.use_grad_adj):
                if(self.share_parameter):
                    grad_adj = self.fnet(grad_loss)
            else:
                grad_adj = grad_loss

            # R(x)
            if(self.use_reg):
                if self.share_parameter:
                    grad_reg = self.rnet(xcurrent)
                grad_direc = grad_adj + grad_reg
            else:
                grad_direc = grad_adj

            # D()
            if(self.use_grad_scaler):
                if (self.share_parameter):
                    grad_scaled = self.dnet(grad_direc)
            else:
                grad_scaled = grad_direc

            ## update x
            xcurrent = self.momen * xcurrent + (1 - self.momen) * grad_scaled
            ## -end- single step operation

            # output
            #output_list += [xcurrent] #save the temp result

        #return output_list
        return xcurrent

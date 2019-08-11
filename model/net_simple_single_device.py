# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-05 12:45:28
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-08-10 20:20:47

import os
import sys
rootPath = os.path.dirname(sys.path[0])
sys.path.append(rootPath)
import numpy as np
import torch.nn as nn
import torch
from metalayers import *

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.noise_dim = params.noise_dims

        self.gkernel = gkern1D(params.gkernlen, params.gkernsig)

        self.FC = nn.Sequential(
            # ------------------------------------------------------
            nn.Linear(self.noise_dim + 2, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.Linear(256, 32 * 16, bias=False),
            nn.BatchNorm1d(32 * 16),
            nn.LeakyReLU(0.2),
        )

        self.CONV = nn.Sequential(
            # ------------------------------------------------------
            ConvTranspose1d_meta(16, 16, 5, stride=2, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            ConvTranspose1d_meta(16, 8, 5, stride=2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            ConvTranspose1d_meta(8, 4, 5, stride=2, bias=False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            ConvTranspose1d_meta(4, 1, 5, stride=1, bias=False),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2),
        )

        self.shortcut = nn.Sequential()

    def forward(self, z):

        net = self.FC(z)
        net = net.view(-1, 16, 32)
        net = self.CONV(net)
        net += self.shortcut(z[:, 2:].view_as(net))
        net = conv1d_meta(net, self.gkernel)
        net = torch.tanh(10 * net) * 1.05

        return net


if __name__ == '__main__':
    import utils
    import torchsummary

    params = utils.Params(os.path.join(rootPath, "results\\Params.json"))

    if torch.cuda.is_available():
        generator = Generator(params).cuda()
    else:
        generator = Generator(params)

    torchsummary.summary(generator, tuple([258]))

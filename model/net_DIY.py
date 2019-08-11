# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-05 12:45:28
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-08-09 23:53:51

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

        self.FC1_n = nn.Sequential(
            nn.Linear(self.noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
        )

        self.FC1_l = nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
        )

        self.FC2 = nn.Sequential(
            nn.Linear(256, 32 * 16, bias=False),
            nn.BatchNorm1d(32 * 16),
            nn.LeakyReLU(0.2),
        )

        self.CONV = nn.Sequential(
            # ------------------------------------------------------
            ConvTranspose1d_meta(64, 32, 5, stride=2, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            ConvTranspose1d_meta(32, 16, 5, stride=2, bias=False),
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

        self.shortcut = nn.Sequential(
            # Conv1d_meta(256, 512, 1, stride=2, bias=False),
            # nn.BatchNorm1d(512),
        )

    def forward(self, z):
        noise = z[:, 2:]
        label = z[:, 0:2]
        # print("noise", noise.shape)
        # print("label", label.shape)
        noiseFC = self.FC2(self.FC1_n(noise))
        labelFC = self.FC2(self.FC1_l(label))
        # print("noiseFC", noiseFC.shape)
        # print("labelFC", labelFC.shape)
        vetorFC = torch.cat((noiseFC, labelFC), 1)
        # print("vetorFC", vetorFC.shape)
        net = vetorFC.view(-1, 64, 16)
        # print("begin deconv", net.shape)
        net = self.CONV(net)
        # print("after deconv", net.shape)
        # noise = noise.unsqueeze(2)
        # noise_res = self.shortcut(noise)
        # print("noise res", noise_res.shape)
        # net += noise_res.transpose(1, 2)
        net += self.shortcut(noise.unsqueeze(1))
        # print("after shortcut", net.shape)
        net = conv1d_meta(net, self.gkernel)
        net = torch.tanh(10 * net) * 1.05
        # print("end", net.shape)

        return net


if __name__ == '__main__':
    import utils
    import torchsummary

    params = utils.Params(os.path.join(rootPath, "results\\Params.json"))
    generator = Generator(params)
    torchsummary.summary(generator, tuple([258]))

# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-07 18:20:26
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-08-08 12:56:04

import torch
import utils
import numpy

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
randconst = torch.rand(1).type(Tensor) * 2 - 1
lamdaconst = torch.rand(1).type(Tensor) * 600 + 300
thetaconst = torch.rand(1).type(Tensor) * 40 + 80
lamda = torch.rand(5, 1).type(Tensor) * 600 + 600
theta = torch.rand(5, 1).type(Tensor) * 40 + 40
json_path = '.\\results\\Params.json'
params = utils.Params(json_path)
noise = (torch.ones(5, params.noise_dims).type(Tensor) * randconst) * params.noise_amplitude
z = torch.cat((lamda, theta, noise), 1)
lamda = lamda[2:, :]
print(lamda.shape)
print(lamda)

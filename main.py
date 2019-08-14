import os
import sys
import logging
import argparse
import numpy as np
from train_and_evaluate import evaluate, train, test, test_group
from model.net_simple_single_device import Generator
import utils
import torch
from torchsummary import summary

import warnings
warnings.filterwarnings('ignore')

import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath('reticolo_allege'))
eng.addpath(eng.genpath('solvers'))

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='results',
                    help="Generated devices folder", type=str)
parser.add_argument('--restore_from', default=None, type=str,
                    help="Optional, directory or file containing weights to reload before training")
parser.add_argument('--wavelength', type=float, default=1200)
parser.add_argument('--angle', type=float, default=40)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--test_group', action='store_true', default=False)
parser.add_argument('--test_num', type=int, default=10, help="Number of pairs you want to feed into test model")
parser.add_argument('--generate_num', type=int, default=10,
                    help="Number of devices you want to generate for each test pair")
parser.add_argument('--heatmap', action='store_true', default=False)
parser.add_argument('--hwstart', type=int, default=600)
parser.add_argument('--hwend', type=int, default=1200)
parser.add_argument('--hwstride', type=int, default=25)
parser.add_argument('--hastart', type=int, default=40)
parser.add_argument('--haend', type=int, default=80)
parser.add_argument('--hastride', type=int, default=2)
parser.add_argument('--tensorboard', action='store_true', default=False)

if __name__ == '__main__':
    # Load the directory from commend line
    args = parser.parse_args()
    output_dir = args.output_dir + \
        '/w{}a{}'.format(args.wavelength, args.angle)

    os.makedirs(output_dir + '/outputs', exist_ok=True)
    os.makedirs(output_dir + '/figures', exist_ok=True)
    os.makedirs(output_dir + '/model', exist_ok=True)

    # Set the logger
    utils.set_logger(os.path.join(output_dir, 'train.log'))

    # Load parameters from json file
    json_path = os.path.join(args.output_dir, 'Params.json')
    assert os.path.isfile(
        json_path), "No json file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Add attributes to params
    params.output_dir = output_dir
    params.lambda_gp = 10.0
    params.n_critic = 1
    params.cuda = torch.cuda.is_available()
    params.restore_from = args.restore_from
    params.heatmap  = args.heatmap
    params.hwstart  = args.hwstart
    params.hwend    = args.hwend
    params.hwstride = args.hwstride
    params.hastart  = args.hastart
    params.haend    = args.haend
    params.hastride = args.hastride
    params.tensorboard = args.tensorboard
    params.w = args.wavelength
    params.a = args.angle

    # Define the models

    generator = Generator(params)
    if params.cuda:
        generator.cuda()

    # Define the optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=params.lr_gen,
                                   betas=(params.beta1_gen, params.beta2_gen))

    # Define the schedulers
    scheduler_G = torch.optim.lr_scheduler.StepLR(
        optimizer_G, step_size=params.step_size, gamma=params.gamma)

    # load model data
    if params.restore_from is not None:
        params.checkpoint = utils.load_checkpoint(
            params.restore_from, generator, optimizer_G, scheduler_G)
        logging.info('Model data loaded')

    if args.test:
        test(generator, eng, numImgs=args.generate_num, params=params)

    elif args.test_group:
        test_group(generator, eng, numImgs=args.generate_num, params=params, test_num=args.test_num)

    else:
        # train the model and save
        if params.numIter != 0:
            logging.info('Start training')
            train(generator, optimizer_G, scheduler_G, eng, params)

        # Generate images and save
        logging.info('Start generating devices for wavelength')
        evaluate(generator, eng, numImgs=args.generate_num, params=params)

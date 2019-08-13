import matlab.engine
import os
import logging
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image
import torch
import utils
import scipy.io as io
import numpy as np
from sklearn.decomposition import PCA
import random
import logger

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
randconst = torch.rand(1).type(Tensor) * 2 - 1


def make_figure_dir(folder):
    os.makedirs(folder + '/figures/scatter', exist_ok=True)
    os.makedirs(folder + '/figures/histogram', exist_ok=True)
    os.makedirs(folder + '/figures/deviceSamples', exist_ok=True)
    os.makedirs(folder + '/figures/scatter_and_histogram', exist_ok=True)


def PCA_model(data_path):
    pca = PCA(n_components=2, svd_solver='randomized')
    dataset = io.loadmat(data_path, struct_as_record=False, squeeze_me=True)
    data = dataset['new']
    pca.fit(data)
    return pca


def PCA_analysis(generator, pca, eng, params, numImgs=100):
    generator.eval()
    imgs = sample_images(generator, numImgs, params)
    generator.train()

    Efficiency = torch.zeros(numImgs)

    img = torch.squeeze(imgs[:, 0, :]).data.cpu().numpy()
    img = matlab.double(img.tolist())
    wavelength = matlab.double([params.w] * numImgs)
    desired_angle = matlab.double([params.a] * numImgs)

    abseffs = eng.Eval_Eff_1D_parallel(img, wavelength, desired_angle)
    Efficiency = torch.Tensor([abseffs]).data.cpu().numpy().reshape(-1)

    # img = img[np.where(Efficiency.reshape(-1) > 0), :]
    # Efficiency = Efficiency[Efficiency > 0]

    img_2 = pca.transform(img)

    fig_path = params.output_dir + \
        '/figures/scatter/Iter{}.png'.format(params.iter)
    utils.plot_scatter(img_2, Efficiency, params.iter, fig_path)

    fig_path = params.output_dir + \
        '/figures/histogram/Iter{}.png'.format(params.iter)
    utils.plot_histogram(Efficiency, params.iter, fig_path)

    imgs = imgs[:8, :, :].unsqueeze(2).repeat(1, 1, 64, 1)
    fig_path = params.output_dir + \
        '/figures/deviceSamples/Iter{}.png'.format(params.iter)
    save_image(imgs, fig_path, 2)

    '''
    grads = eng.GradientFromSolver_1D_parallel(img, wavelength, desired_angle)
    grad_2 = pca.transform(grads)
    if params.iter % 2 == 0:
        utils.plot_envolution(params.img_2_prev, params.eff_prev, params.grad_2_prev,
                              img_2, Efficiency, params.iter, params.output_dir)
    else:
        utils.plot_arrow(img_2, Efficiency, grad_2, params.iter, params.output_dir)
    params.img_2_prev = img_2
    params.eff_prev = Efficiency
    params.grad_2_prev = grad_2
    '''
    return img_2, Efficiency


def sample_images(generator, batch_size, params):

    if params.noise_constant == 1:
        noise = (torch.ones(batch_size, params.noise_dims).type(
            Tensor) * randconst) * params.noise_amplitude
    else:
        if params.noise_distribution == 'uniform':
            noise = (torch.rand(batch_size, params.noise_dims).type(
                Tensor) * 2. - 1.) * params.noise_amplitude
        else:
            noise = (torch.randn(batch_size, params.noise_dims).type(
                Tensor)) * params.noise_amplitude
    lamda = torch.ones(batch_size, 1).type(Tensor) * params.w
    theta = torch.ones(batch_size, 1).type(Tensor) * params.a
    z = torch.cat((lamda, theta, noise), 1)
    if params.cuda:
        z.cuda()
    return generator(z, params.binary_amp)


def evaluate(generator, eng, numImgs, params):
    generator.eval()

    filename = 'ccGAN_imgs_Si_w' + \
        str(params.w) + '_' + str(params.a) + 'deg.mat'
    images = sample_images(generator, numImgs, params)
    file_path = os.path.join(params.output_dir, 'outputs', filename)
    logging.info('Generation is done. \n')

    Efficiency = torch.zeros(numImgs)

    images = torch.sign(images)
    strucs = images.cpu().detach().numpy()
    img = torch.squeeze(images[:, 0, :]).data.cpu().numpy()
    img = matlab.double(img.tolist())
    wavelength = matlab.double([params.w] * numImgs)
    desired_angle = matlab.double([params.a] * numImgs)
    abseffs = eng.Eval_Eff_1D_parallel(img, wavelength, desired_angle)
    Efficiency = torch.Tensor([abseffs]).data.cpu().numpy().reshape(-1)
    max_eff_index = np.argmax(Efficiency)
    max_eff = Efficiency[max_eff_index]
    best_struc = strucs[max_eff_index, :, :].reshape(-1)

    fig_path = params.output_dir + '/figures/Efficiency.png'
    utils.plot_histogram(Efficiency, params.numIter, fig_path)

    print('{} {} {} {} {} {} {:.2f}'.format('The best efficiency for',
                                            'wavelength =', params.w, 'and angle =', params.a, 'is', max_eff))
    io.savemat(file_path, mdict={
               'strucs': strucs, 'effs': Efficiency, 'best_struc': best_struc,
               'max_eff_index': max_eff_index, 'max_eff': max_eff})


def test(generator, eng, numImgs, params):
    generator.eval()

    filename = 'ccGAN_imgs_Si_w' + \
        str(params.w) + '_' + str(params.a) + 'deg_test.mat'
    images = sample_images(generator, numImgs, params)
    file_path = os.path.join(params.output_dir, 'outputs', filename)
    logging.info('Test starts. \n')

    Efficiency = torch.zeros(numImgs)

    images = torch.sign(images)
    strucs = images.cpu().detach().numpy()
    img = torch.squeeze(images[:, 0, :]).data.cpu().numpy()
    img = matlab.double(img.tolist())
    wavelength = matlab.double([params.w] * numImgs)
    desired_angle = matlab.double([params.a] * numImgs)
    abseffs = eng.Eval_Eff_1D_parallel(img, wavelength, desired_angle)
    Efficiency = torch.Tensor([abseffs]).data.cpu().numpy().reshape(-1)
    max_eff_index = np.argmax(Efficiency)
    max_eff = Efficiency[max_eff_index]
    best_struc = strucs[max_eff_index, :, :].reshape(-1)

    print('{} {} {} {} {} {} {:.2f}'.format('The best efficiency for',
                                            'wavelength =', params.w, 'and angle =', params.a, 'is', max_eff))
    io.savemat(file_path, mdict={
               'strucs': strucs, 'effs': Efficiency, 'best_struc': best_struc,
               'max_eff_index': max_eff_index, 'max_eff': max_eff})


def test_group(generator, eng, numImgs, params, test_num):
    generator.eval()

    images = sample_images(generator, numImgs, params)
    logging.info('Test group starts. \n')

    Efficiency = torch.zeros(numImgs)

    images = torch.sign(images)
    strucs = images.cpu().detach().numpy()
    img = torch.squeeze(images[:, 0, :]).data.cpu().numpy()
    img = matlab.double(img.tolist())

    if params.heatmap:
        lamda_list = [600, 700, 800, 900, 1000, 1100, 1200]
        theta_list = [40, 50, 60, 70, 80]
        H = len(lamda_list)
        W = len(theta_list)
        heat_scores = np.zeros((H, W))
        with tqdm(total=H * W, ncols=70) as t:
            for lamda, i in zip(lamda_list[::-1], range(H)):
                for theta, j in zip(theta_list, range(W)):
                    wavelength = matlab.double([lamda] * numImgs)
                    desired_angle = matlab.double([theta] * numImgs)
                    abseffs = eng.Eval_Eff_1D_parallel(img, wavelength, desired_angle)
                    Efficiency = torch.Tensor([abseffs]).data.cpu().numpy().reshape(-1)
                    heat_scores[i, j] = np.max(Efficiency)
                    t.update()
        fig_path = params.output_dir + '/figures/heatmap_batch{}.png'.format(params.solver_batch_size_start)
        utils.plot_heatmap(lamda_list, theta_list, heat_scores, fig_path)
        print("Plot heatmap successfully!")

    else:
        max_eff_index = []
        max_eff = []
        best_struc = []
        with tqdm(total=test_num, ncols=70) as t:
            for i in range(test_num):
                lamda = random.uniform(600, 1200)
                theta = random.uniform(40, 80)

                wavelength = matlab.double([lamda] * numImgs)
                desired_angle = matlab.double([theta] * numImgs)
                abseffs = eng.Eval_Eff_1D_parallel(img, wavelength, desired_angle)
                Efficiency = torch.Tensor([abseffs]).data.cpu().numpy().reshape(-1)
                max_now = np.argmax(Efficiency)
                max_eff_index.append(max_now)
                max_eff.append(Efficiency[max_now])
                best_struc.append(strucs[max_now, :, :].reshape(-1))
                t.update()

        print('{} {:.2f} {} {:.2f} {} {:.2f} {} {:.2f} '.format('Lowest:', min(max_eff), 'Highest:', max(
            max_eff), 'Average:', np.mean(np.array(max_eff)), 'Var:', np.var(np.array(max_eff))))


def train(models, optimizers, schedulers, eng, params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    generator = models
    optimizer_G = optimizers
    scheduler_G = schedulers

    generator.train()

    pca = PCA_model("PCA.mat")

    make_figure_dir(params.output_dir)

    # lamda_list = [600, 700, 800, 900, 1000, 1100, 1200]
    # theta_list = [40, 50, 60, 70, 80]

    if params.restore_from is None:
        Eff_mean_history = []
        Binarization_history = []
        pattern_variance = []
        iter0 = 0
        imgs_2 = []
        Effs_2 = []
    else:
        Eff_mean_history = params.checkpoint['Eff_mean_history']
        iter0 = params.checkpoint['iter']
        Binarization_history = params.checkpoint['Binarization_history']
        pattern_variance = params.checkpoint['pattern_variance']
        imgs_2 = params.checkpoint['imgs_2']
        Effs_2 = params.checkpoint['Effs_2']

    if params.tensorboard:
        loss_logger = logger.set_logger(params.output_dir)

    with tqdm(total=params.numIter, leave=False, ncols=70) as t:

        for i in range(params.numIter):
            it = i + 1
            normIter = it / params.numIter
            params.iter = it + iter0

            scheduler_G.step()

            # binarization amplitude in the tanh function
            if params.iter < 1000:
                params.binary_amp = int(params.iter / 100) + 1

            # use solver and phyiscal gradient to update the Generator
            params.solver_batch_size = int(params.solver_batch_size_start + (params.solver_batch_size_end -
                                                                             params.solver_batch_size_start) * (1 - (1 - normIter)**params.solver_batch_size_power))
            if params.noise_constant == 1:
                noise = (torch.ones(params.solver_batch_size, params.noise_dims).type(
                    Tensor) * randconst) * params.noise_amplitude
            else:
                if params.noise_distribution == 'uniform':
                    noise = ((torch.rand(params.solver_batch_size, params.noise_dims).type(
                        Tensor) * 2. - 1.) * params.noise_amplitude)
                else:
                    noise = (torch.randn(params.solver_batch_size, params.noise_dims).type(
                        Tensor)) * params.noise_amplitude
            """
            batch equivalent
            """
            # lamdaconst = torch.rand(1).type(Tensor) * 600 + 600
            # thetaconst = torch.rand(1).type(Tensor) * 40 + 40
            # lamda = torch.ones(params.solver_batch_size,
            #                    1).type(Tensor) * lamdaconst
            # theta = torch.ones(params.solver_batch_size,
            #                    1).type(Tensor) * thetaconst

            """
            batch randomized
            """
            lamda = torch.rand(params.solver_batch_size, 1).type(Tensor) * 600 + 600
            theta = torch.rand(params.solver_batch_size, 1).type(Tensor) * 40 + 40

            z = torch.cat((lamda, theta, noise), 1)
            z = z.to(device)
            generator.to(device)
            gen_imgs = generator(z, params.binary_amp)

            img = torch.squeeze(gen_imgs[:, 0, :]).data.cpu().numpy()
            img = matlab.double(img.tolist())

            wavelength = matlab.double(lamda.cpu().numpy().tolist())
            desired_angle = matlab.double(theta.cpu().numpy().tolist())

            Grads_and_Effs = eng.GradientFromSolver_1D_parallel(
                img, wavelength, desired_angle)
            Grads_and_Effs = Tensor(Grads_and_Effs)
            grads = Grads_and_Effs[:, 1:]
            Efficiency_real = Grads_and_Effs[:, 0]

            Eff_max = torch.max(Efficiency_real.view(-1))
            Eff_reshape = Efficiency_real.view(-1, 1).unsqueeze(2)

            Gradients = Tensor(grads).unsqueeze(
                1) * gen_imgs * (1. / params.sigma * torch.exp((Eff_reshape - Eff_max) / params.sigma))

            # Train generator
            optimizer_G.zero_grad()

            binary_penalty = params.binary_penalty_start if params.iter < params.binary_step_iter else params.binary_penalty_end
            if params.binary == 1:
                g_loss_solver = - torch.mean(torch.mean(Gradients, dim=0).view(-1)) - torch.mean(
                    torch.abs(gen_imgs.view(-1)) * (2.0 - torch.abs(gen_imgs.view(-1)))) * binary_penalty
            else:
                g_loss_solver = - torch.mean(torch.mean(Gradients, dim=0).view(-1))

            g_loss_solver.backward()
            optimizer_G.step()

            if params.tensorboard:
                loss_logger.scalar_summary('loss', g_loss_solver.cpu().detach().numpy(), it)

            if it == 1 or it % params.save_iter == 0:

                # visualization

                generator.eval()
                outputs_imgs = sample_images(generator, 100, params)

                Binarization = torch.mean(torch.abs(outputs_imgs.view(-1)))
                Binarization_history.append(Binarization)

                diversity = torch.mean(torch.std(outputs_imgs, dim=0))
                pattern_variance.append(diversity.data)

                numImgs = 1 if params.noise_constant == 1 else 100

                img_2_tmp, Eff_2_tmp = PCA_analysis(
                    generator, pca, eng, params, numImgs)
                imgs_2.append(img_2_tmp)
                Effs_2.append(Eff_2_tmp)

                Eff_mean_history.append(np.mean(Eff_2_tmp))
                utils.plot_loss_history(
                    ([], [], Eff_mean_history, pattern_variance, Binarization_history), params.output_dir)

                generator.train()

                # save model

                model_dir = os.path.join(
                    params.output_dir, 'model', 'iter{}'.format(it + iter0))
                os.makedirs(model_dir, exist_ok=True)
                utils.save_checkpoint({'iter': it + iter0,
                                       'gen_state_dict': generator.state_dict(),
                                       'optim_G_state_dict': optimizer_G.state_dict(),
                                       'scheduler_G_state_dict': scheduler_G.state_dict(),
                                       'Eff_mean_history': Eff_mean_history,
                                       'Binarization_history': Binarization_history,
                                       'pattern_variance': pattern_variance,
                                       'Effs_2': Effs_2,
                                       'imgs_2': imgs_2
                                       },
                                      checkpoint=model_dir)

            if it == params.numIter:
                model_dir = os.path.join(params.output_dir, 'model')
                utils.save_checkpoint({'iter': it + iter0,
                                       'gen_state_dict': generator.state_dict(),
                                       'optim_G_state_dict': optimizer_G.state_dict(),
                                       'scheduler_G_state_dict': scheduler_G.state_dict(),
                                       'Eff_mean_history': Eff_mean_history,
                                       'Binarization_history': Binarization_history,
                                       'pattern_variance': pattern_variance,
                                       'Effs_2': Effs_2,
                                       'imgs_2': imgs_2
                                       },
                                      checkpoint=model_dir)

                io.savemat(params.output_dir + '/scatter.mat',
                           mdict={'imgs_2': np.asarray(imgs_2), 'Effs_2': np.asarray(Effs_2)})
                return

            t.update()

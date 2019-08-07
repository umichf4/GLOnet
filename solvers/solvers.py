# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-06 17:14:24
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-08-06 22:24:59

import scipy.io as io
import numpy as np
import math


class sym(object):
    """docstring for sym"""

    def __init__(self, dim):
        super(sym, self).__init__()
        if dim == 2:
            self.x = None
            self.y = None
            self.pol = 0
        else:
            self.x = None


class parm1(object):
    """docstring for parm1"""

    def __init__(self, dim):
        super(parm1, self).__init__()
        if dim == 2:
            self.angles = 1
            self.trace = 0
            self.xlimite = None
            self.ylimite = None
            self.nx = 100
            self.ny = 100
            self.calcul = 1
            self.champ = 0
            self.ftemp = 1
            self.fperm = None
            self.sog = 1
            self.li = 1
        else:
            self.trace = 0
            self.xlimite = None
            self.nx = 1000
            self.calcul = 1
            self.champ = 1
            self.ftemp = 0
            self.fperm = None
            self.sog = 1


class parm2(object):
    """docstring for parm2"""

    def __init__(self):
        super(parm2, self).__init__()
        self.cals = 1
        self.cale = 1
        self.calef = 1
        self.tolh = 1e-6
        self.tolb = 1e-6
        self.retss = 3
        self.retgg = 0
        self.result = 1


class parm3(object):
    """docstring for parm3"""

    def __init__(self):
        super(parm3, self).__init__()
        self.npts = 10
        self.cale = np.arange(1, 7, 1)
        self.calo = 'i'
        self.sens = 1
        self.catlab = 0
        self.gauss = 0
        self.gauss_x = 10
        self.gauss_y = np.NaN
        self.trace = 0
        self.champs = np.array([1, 2, 3, 4, 5, 6, 0])
        self.apod_champ = 0


class parm(object):
    """docstring for parm"""

    def __init__(self, dim, sym, parm1, parm2, parm3):
        super(parm, self).__init__()
        self.dim = dim
        self.sym = sym
        self.res1 = parm1
        self.res2 = parm2
        self.res3 = parm3


def res0(dim):
    material_sym = sym(dim)
    material_parm1 = parm1(dim)
    material_parm2 = parm2()
    material_parm3 = parm3()
    material = parm(dim, material_sym, material_parm1, material_parm2, material_parm3)
    return material


def Eval_Eff_1D(img=None, wavelength=900, angle=60):
    # img = img / 2 + 0.5
    n_air = 1
    n_glass = 1.45
    thickness = 325
    data = io.loadmat('p_Si.mat')
    WL, n = data['WL'], data['n']
    n_Si = np.interp(wavelength, WL.flatten(), n.flatten())
    angle_theta0 = 0
    k_parallel = n_air * math.sin(angle_theta0 * math.pi / 180)
    parm = res0(-1)
    parm.res1.champ = 1

    nn = 40
    period = math.abs(wavelength / math.sin(angle * math.pi / 180))

    N = len(img)
    dx = period / N
    x = np.arange(1, N + 1, 1) * dx - 0.5 * period
    nvec = img * (n_Si - n_air) + n_air


# Eval_Eff_1D(img=None, wavelength=900, angle=60)
print(res0(-1).res1.champ)

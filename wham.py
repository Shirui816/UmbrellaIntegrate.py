#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: shirui <shirui816@gmail.com>

import re
import warnings
import argparse
from scipy.constants import Boltzmann as KB
from scipy.constants import Avogadro as NA
import numpy as np
import pandas as pd
from argparse import RawTextHelpFormatter
from scipy.stats import gaussian_kde
from scipy.stats import circmean
from scipy.stats import multivariate_normal

description = """WHAM.py (Gaussian KDE version).
Written by Shirui shirui816@gmail.com
Bootstrap enhance sampling method is on the way.
I will aslo try GAN to enhance sampling.
### metafile format:
/window/data window_center sprint_konst [Temperature]
### window data file format:
time_step coordinate (1-dimentional)
"""

arg_parser = argparse.ArgumentParser(
    description=description, formatter_class=RawTextHelpFormatter)
arg_parser.add_argument('-P', '--period', nargs='+',
                        default=-1, metavar='val', dest='period', type=float,
                        help='Optional, nonzero val if periodic, 0 for '
                             'non-periodic system, default is 0.')
arg_parser.add_argument('-M', '--mode',
                        default="histogram", metavar='kde|gauss|histogram', dest='mode',
                        type=str, help='Optional, method to estimate the distribution. '
                                       'Defualt is Gaussian KDE method.',
                        choices=['kde', 'gauss', 'histogram'])
arg_parser.add_argument('-o', '--output',
                        metavar='Output free energy file',
                        default='free_py_wham.txt', dest='out_put',
                        help="Optional, use 'free_py_wham.txt' as default", )
arg_parser.add_argument('-R', '--reduced',
                        default=0, type=int, metavar='0|1', dest='is_reduced',
                        choices=[0, 1],
                        help='Is reduced units being used?')
arg_parser.add_argument('-B', '--max_bin', type=int, nargs='+', dest='max_bin',
                        help='How many bins were used in integration.')
arg_parser.add_argument('meta_file', help='Meta file name', type=str)
arg_parser.add_argument('temperature', metavar='Temperature', type=float,
                        help="Temperature.")
arg_parser.add_argument('range', nargs='+',
                        metavar='Range of xi', type=float,
                        help="Range of reaction coordinate.")
args = arg_parser.parse_args()
alvars = vars(args)


# Utils
def pbc(x, d):
    r"""Period boundary condition."""
    return x - d * np.floor(x / d + 0.5)


# Variables
temperature = alvars['temperature']
is_reduced = alvars['is_reduced']
max_bin = np.array(alvars['max_bin'])  # how many bins were used in integration
xi_range = np.array(alvars['range']).reshape(-1, 2)
n_dim = xi_range.shape[0]
period = np.array(alvars['period'])
if period is None:
    period = np.array([-1] * n_dim)
out_put_file = open(alvars['out_put'], 'w')
out_put_file.write('#r\tPMF\tP\n')
out_put_file.close()
out_put_file = open(alvars['out_put'], 'a')
mode = alvars['mode']
meta_file = open(alvars['meta_file'], 'r').readlines()
while '' in meta_file:
    meta_file.remove('')
meta_file = [_ for _ in meta_file if not re.search(r'^#', _)]
n_windows = len(meta_file)

kb = is_reduced or KB * NA
if not period.shape[0] == max_bin.shape[0] == n_dim:
    raise ValueError("Dimension is not correct!")
x0, xt = xi_range.T
x0 = x0.reshape(1, -1)
xt = xt.reshape(1, -1)
xi_range = np.array([(-period[_] / 2, period[_] / 2) if period[_] > 0 else xi_range[_] for _ in range(n_dim)])
pb_w_xis = np.empty((n_windows, *max_bin))
bias_w_xis = np.zeros((n_windows, *max_bin))
f_w = np.ones(n_windows) / n_windows  # initial F for each window
grid = np.meshgrid(*[np.linspace(_[0], _[1], __) for _, __ in zip(xi_range, max_bin)])
grid = np.array(grid)
xis = np.vstack([_.ravel() for _ in grid])
box = np.atleast_2d(period)


def pbc(x, d):
    r"""Period boundary condition."""
    return x - d * np.round(x / d)


for i, line in enumerate(meta_file):
    # loop for each window.
    line = re.split('\s+', line.strip())
    if len(line) != 4:
        warnings.warn("Temperature is not assigned for window %d,"
                      "I will use the default temperature!" % (i),
                      UserWarning)
    window_data = np.atleast_2d(pd.read_csv(line[0], header=None, squeeze=1,
                                            delim_whitespace=True, comment='#').values[:, 1:])
    # the data is always (n_sample, n_dim), (100, 1) is 1D case, for example.
    xi_center_w = np.array([float(line[_]) for _ in range(1, n_dim + 1)])
    k_w = float(line[1 + n_dim])  # bug: I shall enhance the reading part, currently
    # all dimensions share same spring constant.
    kbT_w = float(line[2 + n_dim]) if len(line) == 3 + n_dim else kb * temperature
    window_data = window_data - x0 + xi_range.T[0].reshape(1, -1)  # move data to -P/2, P/2
    xi_mean_w = np.diag(np.where(period > 0, circmean(window_data.T, high=box.T / 2, low=-box.T / 2, axis=1),
                                 np.mean(window_data.T, axis=1)))
    window_data_uwp = np.where(period > 0, pbc(window_data - xi_mean_w.T, period),
                               window_data - xi_mean_w.T)
    # unwrap data then treat circularly distributed data "flattly".
    # I should use the circular-midpoint method.
    # move the MEAN of the distribution to 0, only for relatively
    # symmetric distributions. If the distribution is heavily skewed,
    # the distribution should be "unwrapped" into a whole period and
    # PBC of _delta_xis should be removed, i.e., distances further than
    # half period is permitted. However, I don't think this case is physical...
    delta_xis = np.empty((*max_bin, n_dim))
    for d in range(n_dim):
        if period[d] > 0:
            delta_xis_ref_d = pbc(grid[d].T - xi_center_w[d], period[d])
            delta_xis[..., d] = pbc(grid[d].T - xi_mean_w[d], period[d])
        else:
            delta_xis_ref_d = grid[d].T - xi_center_w[d]
            delta_xis[..., d] = grid[d].T - xi_mean_w[d]
        bias_w_xis[i] += np.exp(-kbT_w * 0.5 * k_w * delta_xis_ref_d ** 2)
    if mode == 'kde':
        kde = gaussian_kde(window_data_uwp.T)
        positions = np.vstack([_.ravel() for _ in delta_xis.swapaxes(0, -1)])
        pb_w_xis[i] = np.reshape(kde(positions).T, grid[0].shape).T
        # not exactly same with the example on the docs of scipy, figure why
    if mode == 'histogram':
        pb_w_xis[i], _ = np.histogramdd(window_data, bins=max_bin, range=xi_range, density=True)
    if mode == 'gauss':
        xi_var_w = np.cov(window_data_uwp.T)
        pb_w_xis[i] = multivariate_normal.pdf(delta_xis, mean=[0] * n_dim, cov=xi_var_w)
        # kde and gauss mode share the same idea: calculate P(\xi) from "unwrapped" distribution
        # if the system is no periodic, "unwrapped" represents P(\xi-\overline{\xi})

pb_w_xis = pb_w_xis / pb_w_xis.sum(axis=tuple(range(1, n_dim + 1)), keepdims=True)
pu_xis_old = np.zeros(max_bin)

# WHAM iteration
counter = 0
while True:
    pu_xis = np.sum(pb_w_xis, axis=0) / np.einsum('i,i...->...', f_w, bias_w_xis)
    pu_xis = pu_xis / pu_xis.sum()
    f_w = 1 / np.sum(np.einsum('i...,...->i...', bias_w_xis, pu_xis),
                     axis=tuple(np.arange(1, n_dim + 1)))
    if counter % 1000 == 0:
        print("F for each window (%d):" % (counter))
        for i, line in enumerate(f_w):
            print(i, '%.4f' % (line))
    counter += 1
    if np.allclose(pu_xis_old, pu_xis, rtol=1e-5):
        break
    pu_xis_old = pu_xis
pmf = -kb * temperature * np.log(pu_xis_old)
np.savetxt(out_put_file, pmf - pmf.min())  # I should also improve the output part, x, y,... pmf for example
# np.savetxt(out_put_file, np.vstack([xis, pmf - pmf.min(), pu_xis]).T, fmt="%.6f")
out_put_file.close()

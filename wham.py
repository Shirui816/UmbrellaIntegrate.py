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

description = """WHAM.py (Gaussian KDE version).
Written by Shirui shirui816@gmail.com
Bootstrap enhance sampling method is on the way.
I will aslo try GAN to enhance sampling.
Multi-dimension version is also on the way.
### metafile format:
/window/data window_center sprint_konst [Temperature]
### window data file format:
time_step coordinate (1-dimentional)
"""

arg_parser = argparse.ArgumentParser(
    description=description, formatter_class=RawTextHelpFormatter)
arg_parser.add_argument('-P', '--period',
                        default=0, metavar='val', dest='period', type=float,
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
arg_parser.add_argument('meta_file', help='Meta file name')
arg_parser.add_argument('range', nargs=2, default=None,
                        metavar='Range of xi', type=float,
                        help="Range of reaction coordinate.")
arg_parser.add_argument('max_bin', type=int,
                        help='How many bins were used in integration.')
arg_parser.add_argument('temperature', metavar='Temperature', type=float,
                        help="Temperature.")
args = arg_parser.parse_args()
alvars = vars(args)


# Utils
def pbc(x, d):
    r"""Period boundary condition."""
    return x - d * np.floor(x / d + 0.5)


# Variables
period = alvars['period']
temperature = alvars['temperature']
is_reduced = alvars['is_reduced']
max_bin = alvars['max_bin']  # how many bins were used in integration
xi_range = alvars['range']
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

if period > 0:
    x0, xt = xi_range
    if not np.allclose(xt - x0, period):
        raise ValueError("The data range is not equal to period!")
    print("Peroid is set to %.2f, the data range is set to (-%.2f, %.2f]!"
          % (period, period / 2, period / 2))
    xi_range[0] = -period / 2
    xi_range[1] = period / 2

xis = np.linspace(xi_range[0], xi_range[1], max_bin)
pb_w_xis = np.empty((n_windows, max_bin))
bias_w_xis = np.empty((n_windows, max_bin))
f_w = np.ones(n_windows) / n_windows  # initial F for each window
window_info = []


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
    window_data = pd.read_csv(line[0], header=None, squeeze=1,
                              delim_whitespace=True, comment='#').values[:, 1]
    xi_center_w = float(line[1])
    k_w = float(line[2])
    kbT_w = float(line[3]) if len(line) == 4 else kb * temperature
    if period > 0:
        window_data = window_data - x0 + xi_range[0]  # move data to -P/2, P/2
        xi_mean_w = circmean(window_data, high=period / 2, low=-period / 2)
        window_data_uwp = pbc(window_data - xi_mean_w, period)
        # unwrap data then treat circularly distributed data "flattly".
        # I should use the circular-midpoint method.
        # move the MEAN of the distribution to 0, only for relatively
        # symmetric distributions. If the distribution is heavily skewed,
        # the distribution should be "unwrapped" into a whole period and
        # PBC of _delta_xis should be removed, i.e., distances further than
        # half period is permitted. However, I don't think this case is physical...
        xi_var_w = np.mean(window_data_uwp ** 2)
        delta_xis = pbc(xis - xi_mean_w, period)
        delta_xis_ref = pbc(xis - xi_center_w, period)
    else:
        xi_mean_w = window_data.mean()
        xi_var_w = window_data.var()
        window_data_uwp = window_data - xi_mean_w
        delta_xis = xis - xi_mean_w
        delta_xis_ref = xis - xi_center_w
    bias_w_xis[i] = np.exp(-kbT_w * 0.5 * k_w * delta_xis_ref ** 2)
    if mode == 'kde':
        kde = gaussian_kde(window_data_uwp, bw_method=0.1)
        pb_w_xis[i] = kde(delta_xis)
    if mode == 'histogram':
        pb_w_xis[i], _ = np.histogram(window_data, bins=max_bin, range=xi_range, density=True)
    if mode == 'gauss':
        pb_w_xis[i] = 1 / np.sqrt(2 * np.pi) * 1 / np.sqrt(xi_var_w) * \
                      np.exp(-0.5 * delta_xis ** 2 / xi_var_w)

pb_w_xis = pb_w_xis / pb_w_xis.sum(axis=1)[:, None]
pu_xis_old = np.zeros(max_bin)

# WHAM iteration
counter = 0
while True:
    pu_xis = np.sum(pb_w_xis / (f_w.dot(bias_w_xis)), axis=0)
    pu_xis = pu_xis / pu_xis.sum()
    f_w = 1 / (bias_w_xis.dot(pu_xis))
    if counter % 1000 == 0:
        print("F for each window (%d):" % (counter))
        for i, line in enumerate(f_w):
            print(i, '%.4f' % (line))
    counter += 1
    if np.allclose(pu_xis_old, pu_xis, rtol=1e-5):
        break
    pu_xis_old = pu_xis
pmf = -kb * temperature * np.log(pu_xis_old)
np.savetxt(out_put_file, np.vstack([xis, pmf - pmf.min(), pu_xis]).T, fmt="%.6f")
out_put_file.close()

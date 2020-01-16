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
from scipy.integrate import simps
from scipy.stats import gaussian_kde
from scipy.stats import circmean
from scipy.stats import kstat

# use gaussian_kde and fit exp(-(\sum_i a_i x_i^(i-1)); periodic data
# the numerical stability of exp(-(\sum_i a_i x_i^(i-1)) is the key, make sure
# that fitting range is large enough so that mean force does not diverge.

description = """An Umbrella Integration program (Gaussian KDE version).
Written by Shirui shirui816@gmail.com
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
                        default="kde", metavar='kde|kastner', dest='mode', type=str,
                        help='Optional, method to estimate the distribution. '
                             'Defualt is Gaussian KDE method.',
                        choices=['kde', 'kastner'])
arg_parser.add_argument('-o', '--output',
                        metavar='Output free energy file',
                        default='free_py.txt', dest='out_put',
                        help="Optional, use 'free_py.txt' as default", )
arg_parser.add_argument('-R', '--reduced',
                        default=0, type=int, metavar='0|1', dest='is_reduced',
                        choices=[0, 1],
                        help='Is reduced units being used?')
arg_parser.add_argument('-Q', '--order',
                        default=2, metavar='order',
                        dest='order', type=int,
                        help="Order of probability function, 0 for pure KDE method.")
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
out_put_file.write('#r PMF MF\n')
out_put_file.close()
out_put_file = open(alvars['out_put'], 'a')
order = alvars['order']
mode = alvars['mode']
meta_file = open(alvars['meta_file'], 'r')

kb = is_reduced or KB * NA

if period > 0:
    x0, xt = xi_range
    if not np.allclose(xt - x0, period):
        raise ValueError("The data range is not equal to period!")
    print("Peroid is set to %.2f, the data range is set to (-%.2f, %.2f]!"
          % (period, period / 2, period / 2))
    xi_range[0] = -period / 2
    xi_range[1] = period / 2

if order > 4 and mode == 'kastner':
    raise ValueError("The order of kastner method is up to 4!")

min_ = []
d = xi_range[1] - xi_range[0]
x = np.linspace(-d, d, max_bin * 2)
xis = np.linspace(xi_range[0], xi_range[1], max_bin)
dxi = np.diff(xis)[0]
dAu_dxis_pb_w = np.zeros((max_bin,))
pb_xi = np.zeros((max_bin,))


def pbc(x, d):
    r"""Period boundary condition."""
    return x - d * np.round(x / d)


for line in meta_file:
    # loop for each window.
    if not re.search('^#', line) is None:
        continue
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
        window_data = pbc(window_data - xi_mean_w, period)
        # unwrap data
        # move the MEAN of the distribution to 0, only for relatively
        # symmetric distributions. If the distribution is heavily skewed,
        # the distribution should be "unwrapped" into a whole period and
        # PBC of _delta_xis should be removed, i.e., distances further than
        # half period is permitted. However, I don't think this case is physical...
        xi_var_w = np.mean(window_data ** 2)
        delta_xis = pbc(xis - xi_mean_w, period)
        delta_xis_ref = pbc(xis - xi_center_w, period)
    else:
        xi_mean_w = window_data.mean()
        xi_var_w = window_data.var()
        window_data = window_data - xi_mean_w
        delta_xis = xis - xi_mean_w
        delta_xis_ref = xis - xi_center_w
    if mode == 'kde':
        kde = gaussian_kde(window_data, bw_method=0.1)
        pi_w = kde(x)
        pi_w = np.where(pi_w > 0, pi_w, 1e-100)  # avoid 0 for np.log
    if order == 2:
        # evaluate \partial A^{ub}_w / \partial \xi \times P^b_w(\xi)
        # and summation of p^b_w
        tmp = 1 / np.sqrt(2 * np.pi) * 1 / np.sqrt(xi_var_w) * \
              np.exp(-0.5 * delta_xis ** 2 / xi_var_w)
        n_tmp = 1.
        dAu_dxis_pb_w += (kbT_w * delta_xis / xi_var_w -
                          k_w * delta_xis_ref) * tmp
    elif mode == 'kde' and order > 0:
        z_ = np.polyfit(x, -kbT_w * np.log(pi_w), order, w=pi_w / pi_w.max())
        # Fit the probability if the extended results of kde is not trusted
        # weights are set to be the probability itself, the fitting is
        # in well accord within the data range. PDF out of the data range
        # extended by Gaussian KDE is very close to 0.
        Z_ = np.poly1d(z_)
        dz_ = np.poly1d(z_[:-1] * np.arange(order, 0, -1))
        tmp = np.exp(-Z_(delta_xis) / kbT_w)
        n_tmp = np.sum(tmp)  # normalization factor, simple summation.
        dAu_dxis_pb_w += (kbT_w * dz_(delta_xis) -
                          k_w * delta_xis_ref) * tmp / n_tmp
    elif mode == 'kde' and order == 0:
        # "pure" kde method
        if period == 0:
            delta_xis_ext = np.r_[delta_xis, delta_xis[-1] + dxi]
        else:
            delta_xis_ext = np.r_[delta_xis, pbc(xis[-1] + dxi - xi_mean_w, period)]
        tmp = kde(delta_xis_ext)[:-1]
        n_tmp = np.sum(tmp)
        dAu_dxis_pb_w += (-kbT_w * np.diff(np.log(kde(delta_xis_ext))) / dxi
                          - k_w * delta_xis_ref) * tmp / n_tmp
    elif mode == 'kastner':
        m1 = xi_mean_w
        m2 = xi_var_w
        m3 = np.mean(window_data ** 3)
        m4 = np.mean(window_data ** 4)
        gamma1 = m3 / m2 ** 1.5
        gamma2 = m4 / m2 ** 2 - 3
        a1_w = kbT_w * (0.5 * m3 / m2 ** 2 - m1 / m2)
        a2_w = kbT_w * (0.5 / m2)
        a3_w = kbT_w * (-m3 / (6 * m2 ** 3))
        a4_w = np.abs(kbT_w * (-gamma2 / (24 * m2 ** 2) + m3 ** 2 / (8 * m2 ** 5)))
        xi_k2_w = kstat(window_data, n=2)
        # data is unwrapped in the whole period with zero mean,
        # for lightly skewed data.
        xi_k3_w = kstat(window_data, n=3)
        xi_k4_w = kstat(window_data, n=4)
        xi_g2_w = xi_k4_w / xi_k2_w ** 2
        if order == 3:
            tmp = np.exp(-(a1_w * delta_xis + a2_w * delta_xis ** 2 +
                           a3_w * delta_xis ** 3) / kbT_w)
            n_tmp = np.sum(tmp)
            dAu_dxis_pb_w += (kbT_w * delta_xis / xi_k2_w +
                              0.5 * kbT_w * xi_k3_w / xi_k2_w ** 2 *
                              (1 - delta_xis ** 2 / xi_k2_w) - k_w * delta_xis_ref) * tmp / n_tmp
        if order == 4:
            tmp = np.exp(-(a1_w * delta_xis + a2_w * delta_xis ** 2 +
                           a3_w * delta_xis ** 3 + a4_w * delta_xis ** 4) / kbT_w)
            n_tmp = np.sum(tmp)
            dAu_dxis_pb_w += (kbT_w * delta_xis / xi_k2_w + 0.5 * kbT_w * xi_k3_w / xi_k2_w ** 2 *
                              (1 - delta_xis ** 2 / xi_k2_w) +
                              kbT_w * (0.5 * xi_k3_w ** 2 / xi_k2_w ** 5 -
                                       1 / 6 * xi_g2_w / xi_k2_w ** 2) *
                              delta_xis ** 3 - k_w * delta_xis_ref) * tmp / n_tmp
    pb_xi += tmp / n_tmp
    min_.append(window_data.min() + xi_mean_w)
    min_.append(window_data.max() + xi_mean_w)

if min(min_) > xi_range[0] or max(min_) < xi_range[1]:
    warnings.warn("Warning, xi range exceeds the sample range!",
                  UserWarning)

dAu_dxis = dAu_dxis_pb_w / pb_xi
if period > 0:
    dAu_dxis -= dAu_dxis.mean()  # remove the drifting
pmf = np.array([simps(dAu_dxis[xis <= r], xis[xis <= r]) for r in xis])
np.savetxt(out_put_file, np.vstack([xis, pmf, dAu_dxis]).T, fmt="%.6f")
out_put_file.close()

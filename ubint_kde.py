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
Periodic function is not fully tested!
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
arg_parser.add_argument('-T', '--temperature',
                        metavar='Temperature',
                        dest='temperature', default=-1, type=float,
                        help="Optional, set a default temperature globally.")
arg_parser.add_argument('-R', '--reduced',
                        default=0, type=int, metavar='0|1', dest='is_reduced',
                        choices=[0, 1],
                        help='Is reduced units being used?')
arg_parser.add_argument('-Q', '--order',
                        default=2, metavar='order',
                        dest='order', type=int,
                        help="Order of probability function.")
arg_parser.add_argument('meta_file',
                        nargs=None,
                        help='Meta file name')
arg_parser.add_argument('range', nargs=2, default=None,
                        metavar='Range of xi', type=float,
                        help="Range of reaction coordinate.")
arg_parser.add_argument('max_bin',
                        type=int, nargs=None,
                        help='How many bins were used in integration.')

args = arg_parser.parse_args()
alvars = vars(args)


# Utils
def pbc(x, d):
    r"""Period boundary condition."""
    return x - d * np.round(x / d)


# Variables
_period = alvars['period']
_temperature = alvars['temperature']
_is_reduced = alvars['is_reduced']
_max_bin = alvars['max_bin']  # how many bins were used in integration
_xi_range = alvars['range']
_out_put_file = open(alvars['out_put'], 'w')
_out_put_file.write('#r PMF MF\n')
_out_put_file.close()
_out_put_file = open(alvars['out_put'], 'a')
_order = alvars['order']
_mode = alvars['mode']

_kb = _is_reduced or KB * NA

if _period > 0:
    _xi_range[0] = -_period / 2
    _xi_range[1] = _period / 2
    print("Peroid is set to %.2f, the data range is set to (-%.2f, %.2f]!"
          % (_period, _period / 2, _period / 2))

if _order > 4 and _mode == 'kastner':
    raise ValueError("The order of kastner method is up to 4!")

_meta_file = open(alvars['meta_file'], 'r')
_min = []
_d = _xi_range[1] - _xi_range[0]
_x = np.linspace(-_d, _d, _max_bin * 2)
_xis = np.linspace(_xi_range[0], _xi_range[1], _max_bin)
_dAu_dxis_pb_w = np.zeros((_max_bin,))
_pb_xi = np.zeros((_max_bin,))


class NoTemperatureError(Exception):
    r"""No temperature error."""
    pass


def pbc(x, d):
    r"""Period boundary condition."""
    return x - d * np.round(x / d)


for _line in _meta_file:
    # loop for each window.
    if not re.search('^#', _line) is None:
        continue
    _line = re.split('\s+', _line.strip())
    if _temperature == -1 and len(_line) != 4:
        raise NoTemperatureError("You have not set temperature for this "
                                 "window or a global temperature!")
    _window_data = pd.read_csv(_line[0], header=None, squeeze=1,
                               delim_whitespace=True, comment='#').values[:, 1]
    _xi_center_w = float(_line[1])
    _k_w = float(_line[2])
    _kbT_w = float(_line[3]) if len(_line) == 4 else _kb * _temperature
    if _period > 0:
        _xi_mean_w = circmean(_window_data, high=_period / 2, low=-_period / 2)
        _window_data = pbc(_window_data - _xi_mean_w, _period)
        # move the MEAN of the distribution to 0, only for relatively
        # symmetric distributions. If the distribution is heavily skewed,
        # the distribution should be "unwrapped" into a whole period and
        # PBC of _delta_xis should be removed, i.e., distances further than
        # half period is permitted. However, I don't think this case is physical...
        _xi_var_w = np.mean(_window_data ** 2)
        _delta_xis = pbc(_xis - _xi_mean_w, _period)
        _delta_xis_ref = pbc(_xis - _xi_center_w, _period)
    else:
        _xi_mean_w = _window_data.mean()
        _xi_var_w = _window_data.var()
        _window_data = _window_data - _xi_mean_w
        _delta_xis = _xis - _xi_mean_w
        _delta_xis_ref = _xis - _xi_center_w
    _kde = gaussian_kde(_window_data)
    _pi_w = _kde(_x)

    if _order == 2:
        # evaluate \partial A^{ub}_w / \partial \xi \times P^b_w(\xi)
        # and summation of p^b_w
        _tmp = 1 / np.sqrt(2 * np.pi) * 1 / np.sqrt(_xi_var_w) * \
               np.exp(-0.5 * _delta_xis ** 2 / _xi_var_w)
        _ntmp = 1
        _dAu_dxis_pb_w += (_kbT_w * _delta_xis / _xi_var_w -
                           _k_w * _delta_xis_ref) * _tmp
    else:
        if _mode == 'kde':
            _z = np.polyfit(_x, -_kbT_w * np.log(_pi_w), _order, w=_pi_w)
            _Z = np.poly1d(_z)
            _dz = np.poly1d(_z[:-1] * np.arange(_order, 0, -1))
            _tmp = np.exp(-_Z(_delta_xis) / _kbT_w)
            _ntmp = np.sum(_tmp)  # normalization factor, simple summation.
            _dAu_dxis_pb_w += (_kbT_w * _dz(_delta_xis) -
                               _k_w * _delta_xis_ref) * _tmp / _ntmp
        else:
            m1 = _xi_mean_w
            m2 = _xi_var_w
            m3 = np.mean((_window_data) ** 3)
            m4 = np.mean((_window_data) ** 4)
            gamma1 = m3 / m2 ** 1.5
            gamma2 = m4 / m2 ** 2 - 3
            _a1_w = _kbT_w * (0.5 * m3 / m2 ** 2 - m1 / m2)
            _a2_w = _kbT_w * (0.5 / m2)
            _a3_w = _kbT_w * (-m3 / (6 * m2 ** 3))
            _a4_w = np.abs(_kbT_w * (-gamma2 / (24 * m2 ** 2) + m3 ** 2 / (8 * m2 ** 5)))
            _xi_k2_w = kstat(_window_data, n=2)
            # data is unwrapped in the whole period with zero mean,
            # for lightly skewed data.
            _xi_k3_w = kstat(_window_data, n=3)
            _xi_k4_w = kstat(_window_data, n=4)
            _xi_g2_w = _xi_k4_w / _xi_k2_w ** 2
            if _order == 3:
                _tmp = np.exp(-(_a1_w * _delta_xis + _a2_w * _delta_xis ** 2 +
                                _a3_w * _delta_xis ** 3) / _kbT_w)
                _ntmp = np.sum(_tmp)
                _dAu_dxis_pb_w += (_kbT_w * _delta_xis / _xi_k2_w +
                                   0.5 * _kbT_w * _xi_k3_w / _xi_k2_w ** 2 *
                                   (1 - _delta_xis ** 2 / _xi_k2_w) - _k_w * _delta_xis_ref) * _tmp / _ntmp
            if _order == 4:
                _tmp = np.exp(-(_a1_w * _delta_xis + _a2_w * _delta_xis ** 2 +
                                _a3_w * _delta_xis ** 3 + _a4_w * _delta_xis ** 4) / _kbT_w)
                _ntmp = np.sum(_tmp)
                _dAu_dxis_pb_w += (_kbT_w * _delta_xis / _xi_k2_w + 0.5 * _kbT_w * _xi_k3_w / _xi_k2_w ** 2 *
                             (1 - _delta_xis ** 2 / _xi_k2_w) +
                             _kbT_w * (0.5 * _xi_k3_w ** 2 / _xi_k2_w ** 5 -
                                       1 / 6 * _xi_g2_w / _xi_k2_w ** 2) *
                             _delta_xis ** 3 - _k_w * _delta_xis_ref) * _tmp / _ntmp
    _pb_xi += _tmp / _ntmp
    _min.append(_window_data.min() + _xi_mean_w)
    _min.append(_window_data.max() + _xi_mean_w)

if min(_min) > _xi_range[0] or max(_min) < _xi_range[1]:
    warnings.warn("Warning, xi range exceeds the sample range!",
                  UserWarning)

if _period != 0 and max(_min) - min(_min) < _period:
    warnings.warn("Your sampled data is lesser than 1 period!",
                  UserWarning)

_dAu_dxis = _dAu_dxis_pb_w / _pb_xi
_pmf = np.array([simps(_dAu_dxis[_xis <= r], _xis[_xis <= r]) for r in _xis])
np.savetxt(_out_put_file, np.vstack([_xis, _pmf, _dAu_dxis]).T, fmt="%.6f")
_out_put_file.close()

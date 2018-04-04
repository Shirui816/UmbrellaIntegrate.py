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
from scipy.integrate import trapz

description = """An Umbrella Integration program.
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
arg_parser.add_argument('-o', '--output',
                        metavar='Output free energy file',
                        default='free_py.txt', dest='out_put',
                        help="Optional, use 'free_py.txt' as default",)
arg_parser.add_argument('-T', '--temperature',
                        metavar='Temperature',
                        dest='temperature', default=-1, type=float,
                        help="Optional, set a default temperature globally.")
arg_parser.add_argument('-R', '--reduced',
                        default=0, type=int, metavar='0|1', dest='is_reduced',
                        help='Is reduced units being used?')
arg_parser.add_argument('-X', '--range',
                        nargs=2, default=None, metavar='Range of xi',
                        dest='range', type=float,
                        help="Range of reaction coordinate.")
arg_parser.add_argument('meta_file',
                        nargs=None,
                        help='Meta file name')
arg_parser.add_argument('max_bin',
                        type=int, nargs=None,
                        help='How many bins were used in integration.')

args = arg_parser.parse_args()
alvars = vars(args)

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

_kb = _is_reduced or KB * NA
if _xi_range:
    assert _xi_range[0] < _xi_range[1], "Give the rigth range!"

_meta_file = open(alvars['meta_file'], 'r')
_window_info, _min = [], []


class NoTemperatureError(Exception):
    r"""No temperature error."""

    pass


def pbc(x, d):
    r"""Period boundary condition."""
    return x - d * np.round(x / d)


for _line in _meta_file:
    if not re.search('^#', _line) is None:
        continue
    _line = re.split('\s+', _line.strip())
    if _temperature == -1 and len(_line) != 4:
        raise NoTemperatureError("You have not set temperature for this "
                                 "window or a global temperature!")
    _window_data = pd.read_csv(_line[0], header=None, squeeze=1,
                               delim_whitespace=True, comment='#').values[:, 1]
    center_ = float(_line[1])
    spring_konst = float(_line[2])
    kbT = float(_line[3]) if len(_line) == 4 else _kb * _temperature
    _window_info.append([_window_data.mean(), _window_data.var(),
                         center_, spring_konst, kbT])
    _min.append(_window_data.min())
    _min.append(_window_data.max())

_window_info = np.array(_window_info)
_window_info = _window_info[np.argsort(_window_info.T[2])]

if _period != 0 and max(_min) - min(_min) < _period:
    warnings.warn("Your sampled data is lesser than 1 period!",
                  UserWarning)

if _xi_range:
    if min(_min) < _xi_range[0] or max(_min) > _xi_range[1]:
        warnings.warn("Warning, xi range exceeds the sample range!",
                      UserWarning)

_xi_range = _xi_range or [min(_min), max(_min)]
_xis = np.linspace(_xi_range[0], _xi_range[1], _max_bin)
_xi_mean_w = _window_info.T[0][:, np.newaxis]
_xi_var_w = _window_info.T[1][:, np.newaxis]
_xi_center_w = _window_info.T[2][:, np.newaxis]
_k_w = _window_info.T[3][:, np.newaxis]
_kbT_w = _window_info.T[4][:, np.newaxis]
_pbc_xis = _xis
if _period != 0:
    _pbc_xis = pbc(_xis - _xi_mean_w, _period) + _xi_center_w
_dAu_dxis = _kbT_w * (_pbc_xis - _xi_mean_w) / _xi_var_w -\
        _k_w * (_pbc_xis - _xi_center_w)
_pb_i = 1/np.sqrt(2 * np.pi) * 1 / np.sqrt(_xi_var_w) *\
        np.exp(-0.5 * (_pbc_xis - _xi_mean_w) ** 2 / _xi_var_w)
_dA_dxis = np.sum(_dAu_dxis * _pb_i, axis=0)
_pb_xi = np.sum(_pb_i, axis=0)
_dA_dxis /= _pb_xi

_pmf = np.array([trapz(_dA_dxis[_xis <= r], _xis[_xis <= r]) for r in _xis])
np.savetxt(_out_put_file, np.vstack([_xis, _pmf, _dA_dxis]).T, fmt="%.6f")
_out_put_file.close()

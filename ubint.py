from sys import argv
from pylab import *
from scipy.constants import Boltzmann as kb
from scipy.constants import Avogadro as NA
import numpy
import pandas as pd
from pandas import DataFrame as df
import argparse
from argparse import RawTextHelpFormatter
description="""
An Umbrella Integration program.
Written by Shirui shirui816@gmail.com
### metafile format:
/window/data window_center sprint_konst [Temperature]
### window data file format:
time_step coordinate (1-dimentional)
"""
parser = argparse.ArgumentParser(description=description,formatter_class=RawTextHelpFormatter)
parser.add_argument('-P','--period',type=float, help='Optional, nonzero val if periodic, 0 for non-periodic system, default is 0', default=0, metavar='val', dest='peri')
parser.add_argument('metafile', nargs=None, help='Meta file name') # nargs = 1 for a list
parser.add_argument('max_bin', nargs=None, help='How many bins were used in integration', type=int)
parser.add_argument('-o','--output', metavar='FreeEnergyFile', help="Optional, use 'free_py.txt' as default", default='free_py.txt', dest='outp')
parser.add_argument('-T','--temperature', metavar='Temperature', dest='T', default=-1, type=float, help="Optional, set a default temperature globally")
parser.add_argument('-R','--reduced',metavar='0|1', dest='R',default=0, type=int, help='Is reduced units being used?')
args = parser.parse_args()
alvars = vars(args)

# Variables

Peri = alvars['peri']
O = alvars['outp']
T = alvars['T']
R = alvars['R']
kbT = 'nil'
if R != 0:
    kb, NA = 1, 1000 # reduce kb * NA/1000 to 1
if T != -1:
    kbT = kb * T * NA / 1000
MAX_BINS = alvars['max_bin'] # how many bins were used in integration

class NoTemperatureError(Exception):
	pass

class MultiPeriodError(Exception):
	pass

def pbc(r, d):
	return(r-d*round(r/d))

def loadmeta(filename):
    res = []
    o = open(filename,'r')
    for l in o:
        d = {}
        t = l.split(' ')
        for m in t:
            d['file'] = t[0]
            d['bin'] = float(t[1])
            d['K'] = float(t[2])
            try:
                d['kbT'] = float(t[3]) * kb * NA / 1000
            except:
                d['kbT'] = kbT
                if kbT == 'nil':
                	raise NoTemperatureError("Sorry, no temperature was given by '-T' option nor the meta file!")
        res.append(d)
    return df(res)

conf = loadmeta(alvars['metafile'])
conf.sort_values(by='bin')

def a_u(x, xbin, ximean, xivar, K, kbT): # $\frac{\partial{A^u}}{\partial{\xi}$
    return kbT * (x - ximean)/xivar - K * (x - xbin)

def pbc_a_u(x, xbin, ximean, xivar, K, kbT):
	pbcx = pbc(x-xbin,Peri) + xbin
	return(kbT * (pbcx-ximean)/xivar - K * (pbcx-xbin))

from math import sqrt, pi
from numpy import exp
def P(x, ximean, xivar): # Gaussian distribution
    return 1/(sqrt(2*pi* xivar)) * exp(-0.5 * (x - ximean)**2 / xivar)

para = []
def get_para():
    for xifile, K, t, xibin in zip(conf['file'], conf['K'], conf['kbT'], conf['bin']): # t is kbT here
        #wxis = loadtxt(xifile).T[1]
        window_data = pd.read_csv(xifile, delimiter='\s+', names=['time_step', 'xi'])
        wxis = window_data['xi']
        ximean = wxis.mean()
        xivar = wxis.var()
        #print((xibin, ximean, xivar, K))
        para.append((xibin, ximean, xivar, K, t))
        ############   0      1       2    3  4
get_para() # this is the most time-consuming part, reading all coordiante files.

m, M = conf['bin'].min(), conf['bin'].max()
if  Peri != 0 and Peri < M-m:
	raise(MultiPeriodError("Only 1 perid data is available!"))

xis = linspace(m, M, MAX_BINS)

def Pi(xi, p, paras):
    ai = P(xi, p[1], p[2])
    alsum = 0
    for pa in paras:
        alsum += P(xi, pa[1], pa[2])
    return ai / alsum

def pbcPi(xi, p, paras):
	ai = P(xi, p[1], p[2])
	alsum = 0
	for pa in paras:
		pbcxi = pbc(xi- pa[0], Peri)+pa[0]
		alsum += P(pbcxi, pa[1], pa[2])
	return(ai/alsum)

if not Peri==0:
	Pi = pbcPi
	a_u = pbc_a_u

dau_dxis = [] # $\frac{\partial{A^u}}{\partial{\xi}}$

def get_dau_dxis():
    for xi in xis:
        dau_dxi = 0
        for p in para:
            dau_dxi += a_u(xi, p[0], p[1], p[2], p[3], p[4]) * Pi(xi, p, para)
        dau_dxis.append(dau_dxi)
get_dau_dxis()

dau_dxis = array(dau_dxis)
from scipy.integrate import trapz
PMF = []

for r in xis:
    PMF.append(trapz(dau_dxis[xis<=r], xis[xis<=r]))

o = open(O,'w')
for i,j in zip(xis, PMF):
    o.write('%s %s\n' % (i,j))
o.close()

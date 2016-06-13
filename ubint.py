from sys import argv
from pylab import *
from scipy.constants import Boltzmann as kb
from scipy.constants import Avogadro as NA
import numpy
from pandas import DataFrame as df

# Variables

T = 300
kbT = kb * T * NA / 1000
MAX_BINS = 200 # how many bins were used in integration

### Read metafile
## metafile format:
# /path/to/file/of/reaction/coordinates/of/a/window window_center sprint_konst [Temperature]
## coordinates file format:
# time_step coordinate (1-dimentional)

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
        res.append(d)
    return df(res)


conf = loadmeta(argv[1])
conf.sort_values(by='bin')

def a_u(x, xbin, ximean, xivar, K, kbT): # $\frac{\partial{A^u}}{\partial{\xi}$
    return kbT * (x - ximean)/xivar - K * (x - xbin)

from math import sqrt, pi
from numpy import exp
def P(x, ximean, xivar): # Gaussian distribution
    return 1/(sqrt(2*pi* xivar)) * exp(-0.5 * (x - ximean)**2 / xivar)

para = []
def get_para():
    for xifile, K, t, xibin in zip(conf['file'], conf['K'], conf['kbT'], conf['bin']): # t is kbT here
        wxis = loadtxt(xifile).T[1]
        ximean = wxis.mean()
        xivar = wxis.var()
        #print((xibin, ximean, xivar, K))
        para.append((xibin, ximean, xivar, K, t))
get_para() # this is the most time-consuming part, reading all coordiante files.

xis = linspace(conf['bin'].min(), conf['bin'].max(), MAX_BINS)

def Pi(xi, p, paras):
    ai = P(xi, p[1], p[2])
    alsum = 0
    for pa in paras:
        alsum += P(xi, pa[1], pa[2])
    return ai / alsum

dau_dxis = [] # $\frac{\partial{A^u}}{\partial{\xi}}$

def get_dau_dxis():
    for xi in xis:
        dau_dxi = 0
        for p in para:
            dau_dxi += a_u(xi, p[0], p[1], p[2], p[3], p[4]) * Pi(xi, p, para)
        dau_dxis.append(dau_dxi)
get_dau_dxis()

dau_dxis = array(dau_dxis)
from scipy.integrate import simps
PMF = []

for r in xis:
    PMF.append(simps(dau_dxis[xis<=r], xis[xis<=r]))

o = open('free_py.txt','w')
for i,j in zip(xis, PMF):
    o.write('%s %s\n' % (i,j))
o.close()

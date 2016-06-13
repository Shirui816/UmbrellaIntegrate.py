from sys import argv
from pylab import *
from scipy.constants import Boltzmann as kb
from scipy.constants import Avogadro as NA
import numpy
from pandas import DataFrame as df
from scipy.stats import kstat, moment
from scipy.integrate import simps

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

mconf, Mconf = conf['bin'].min(), conf['bin'].max()

xis = linspace(mconf, Mconf, MAX_BINS)

def a_u4(x, xbin, k1, k2, k3, k4, G2, K, kbT): # $\frac{\partial{A^u}}{\partial{\xi}$
    return(kbT*(x - k1)/k2 + kbT * k3/(2*k2**2) * (1-(x-k1)**2/k2) + kbT*(k3**2/(2*k2**5)-G2/(6*k2**2))*(x-k1)**3-K*(x-xbin))

def a_u3(x, xbin, k1, k2, k3, k4, G2, K, kbT):
    return(kbT*(x - k1)/k2+kbT * k3/(2*k2**2) * (1-(x-k1)**2/k2)-K*(x-xbin))

def a_u2(x, xbin, k1, k2, k3, k4, G2, K, kbT): # Gives the same value as ubint.py
    return(kbT*(x - k1)/k2-K*(x-xbin))


from math import sqrt, pi
from numpy import exp

def EP4(xirange, t, a1, a2, a3, a4):
    return exp(-1/t * (a1*(xirange)+a2*(xirange)**2+a3*(xirange)**3+a4*(xirange)**4))

def EP3(xirange, t, a1, a2, a3, a4):
    return exp(-1/t * (a1*(xirange)+a2*(xirange)**2+a3*(xirange)**3))

def EP2(xirange, t, a1, a2, a3, a4):
    return(exp(-1/t * (a1*(xirange)+a2*(xirange)**2)))

def P4(xi, t, a1, a2, a3, a4, NF):
    EXP = exp(-1/t * (a1*(xi)+a2*(xi)**2+a3*(xi)**3+a4*(xi)**4))
    return EXP/NF

def P3(xi, t, a1, a2, a3, a4, NF):
    EXP = exp(-1/t * (a1*(xi)+a2*(xi)**2+a3*(xi)**3))
    return EXP/NF

def P2(xi, t, a1, a2, a3, a4, NF):
    EXP = exp(-1/t * (a1*(xi)+a2*(xi)**2))
    return EXP/NF

INT_PATH = linspace((mconf* 2-2 * Mconf), (2*Mconf-mconf), MAX_BINS * 4)

para = []
def get_para():
    for xifile, K, t, xibin in zip(conf['file'], conf['K'], conf['kbT'], conf['bin']): # t is kbT here
        wxis = loadtxt(xifile).T[1]
        k1, k2, k3, k4 = kstat(wxis, 1), kstat(wxis, 2), kstat(wxis, 3), kstat(wxis, 4)
        #k1, k2, k3, k4 = moment(wxis, 1), moment(wxis, 2), moment(wxis, 3), moment(wxis, 4)
        G2 = k4/k2**2
        #G2 = k4/k2**2 - 3
        a1 = t * k3/(2*k2**2) - t * k1/k2
        a2 = t / (2 * k2)
        a3 = abs(-t * k3/(6 * k2 ** 3))
        a4 = abs(-G2 * t/(24* k2**2) + t *k3 ** 2/(8 * k2 ** 5))
        NF = simps(EP4(INT_PATH, t, a1, a2, a3, a4), INT_PATH) # Normalization factor of P_i
        NF = nan_to_num(NF)
        #print(NF, xibin)
        para.append((xibin, a1, a2, a3, a4, k1, k2, k3, k4, G2, K, t, NF))
        ############# 0      1   2   3   4   5   6   7   8   9  10 11 12
get_para() # this is the most time-consuming part, reading all coordiante files.

def Pi(xi, p, paras):
    ai = P4(xi, p[11], p[1], p[2], p[3], p[4], p[12])
    alsum = 0
    for pa in paras:
        alsum += P4(xi, pa[11], pa[1], pa[2], pa[3], pa[4], pa[12])
    return ai / alsum

dau_dxis = [] # $\frac{\partial{A^u}}{\partial{\xi}}$


def get_dau_dxis():
    for xi in xis:
        dau_dxi = 0
        for p in para:
            dau_dxi += a_u4(xi, p[0], p[5], p[6], p[7], p[8], p[9], p[10], p[11]) * Pi(xi, p, para)
        dau_dxis.append(dau_dxi)
get_dau_dxis()

dau_dxis = array(dau_dxis)
PMF = []

for r in xis:
    PMF.append(simps(dau_dxis[xis<=r], xis[xis<=r]))

o = open('free_ho_py.txt','w')
for i,j in zip(xis, PMF):
    o.write('%s %s\n' % (i,j))
o.close()

o = open('mean_force_ho_py.txt', 'w')
for i, j in zip(xis, dau_dxis):
	o.write("%s %s\n" % (i, j))
o.close()

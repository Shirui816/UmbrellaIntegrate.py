from sys import argv
from pylab import *
from scipy.constants import Boltzmann as kb
from scipy.constants import Avogadro as NA
import numpy
from pandas import DataFrame as df
import pandas as pd
from scipy.stats import kstat, moment
from scipy.integrate import simps
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
parser.add_argument('-T','--temperature', metavar='Temperature', dest='T', default=-1, type=float, help="Optional, set a default temperature globally")
parser.add_argument('-o', '--order', metavar='order', dest='O', default=4, type=int, help="The order of A, default 4")
args = parser.parse_args()
alvars = vars(args)

# Variables

Peri = alvars['peri']
O = alvars['O']
T = alvars['T']
kbT = 'nil'
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

mconf, Mconf = conf['bin'].min(), conf['bin'].max()
if Peri < Mconf-mconf:
    raise(MultiPeriodError("Only 1 perid data is available!"))

xis = linspace(mconf, Mconf, MAX_BINS)

def a_u4(x, xbin, k1, k2, k3, k4, G2, K, kbT): # $\frac{\partial{A^u}}{\partial{\xi}$
    return(kbT*(x - k1)/k2 + kbT * k3/(2*k2**2) * (1-(x-k1)**2/k2) + kbT*(k3**2/(2*k2**5)-G2/(6*k2**2))*(x-k1)**3-K*(x-xbin))

def a_u3(x, xbin, k1, k2, k3, k4, G2, K, kbT):
    return(kbT*(x - k1)/k2+kbT * k3/(2*k2**2) * (1-(x-k1)**2/k2)-K*(x-xbin))

def a_u2(x, xbin, k1, k2, k3, k4, G2, K, kbT):
    return(kbT*(x - k1)/k2-K*(x-xbin))

def a_u4_pbc(x, xbin, k1, k2, k3, k4, G2, K, kbT): # $\frac{\partial{A^u}}{\partial{\xi}$
    pbcx = pbc(x-xbin,Peri) + xbin
    return(kbT*(pbcx - k1)/k2 + kbT * k3/(2*k2**2) * (1-(pbcx-k1)**2/k2) + kbT*(k3**2/(2*k2**5)-G2/(6*k2**2))*(pbcx-k1)**3-K*(pbcx-xbin))

def a_u3_pbc(x, xbin, k1, k2, k3, k4, G2, K, kbT):
    pbcx = pbc(x-xbin,Peri) + xbin
    return(kbT*(pbcx - k1)/k2+kbT * k3/(2*k2**2) * (1-(pbcx-k1)**2/k2)-K*(pbcx-xbin))

def a_u2_pbc(x, xbin, k1, k2, k3, k4, G2, K, kbT):
    pbcx = pbc(x-xbin,Peri) + xbin
    return(kbT*(pbcx - k1)/k2-K*(pbcx-xbin))



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


INT_PATH = linspace((mconf* 2-2 * Mconf), (2*Mconf-mconf), MAX_BINS * 4) # Enlarge the integration zone

para = []
def get_para():
    for xifile, K, t, xibin in zip(conf['file'], conf['K'], conf['kbT'], conf['bin']): # t is kbT here
        #wxis = loadtxt(xifile).T[1]
        window_data = pd.read_csv(xifile, delimiter='\s+', names=['time_step', 'xi'])
        wxis = window_data['xi']
        k1, k2, k3, k4 = kstat(wxis, 1), kstat(wxis, 2), kstat(wxis, 3), kstat(wxis, 4)
        #k1, k2, k3, k4 = moment(wxis, 1), moment(wxis, 2), moment(wxis, 3), moment(wxis, 4)
        G2 = k4/k2**2
        #G2 = k4/k2**2 - 3
        a1 = t * k3/(2*k2**2) - t * k1/k2
        a2 = t / (2 * k2)
        a3 = abs(-t * k3/(6 * k2 ** 3))
        a4 = abs(-G2 * t/(24* k2**2) + t *k3 ** 2/(8 * k2 ** 5))
        eepp = eval("EP%s" % (O))
        NF = simps(eepp(INT_PATH, t, a1, a2, a3, a4), INT_PATH) # Normalization factor of P_i
        NF = nan_to_num(NF)
        print(NF, xibin)
        para.append((xibin, a1, a2, a3, a4, k1, k2, k3, k4, G2, K, t, NF))
        ############# 0      1   2   3   4   5   6   7   8   9  10 11 12
get_para() # this is the most time-consuming part, reading all coordiante files.

PP = eval("P%s" % (O))

def Pi(xi, p, paras):
    ai = PP(xi, p[11], p[1], p[2], p[3], p[4], p[12])
    alsum = 0
    for pa in paras:
        alsum += PP(xi, pa[11], pa[1], pa[2], pa[3], pa[4], pa[12])
    return ai / alsum

def Pi_pbc(xi, p, paras):
    ai = PP(xi, p[11], p[1], p[2], p[3], p[4], p[12])
    alsum = 0
    for pa in paras:
        pbcxi = pbc(xi- pa[0], Peri)+pa[0]
        alsum += PP(xi, pa[11], pa[1], pa[2], pa[3], pa[4], pa[12])
    return ai / alsum

if not Peri == 0:
    Pi = Pi_pbc

AU = eval('a_u%s%s' % (O, '_pbc' if Peri else ''))

dau_dxis = [] # $\frac{\partial{A^u}}{\partial{\xi}}$


def get_dau_dxis():
    for xi in xis:
        dau_dxi = 0
        for p in para:
            dau_dxi += AU(xi, p[0], p[5], p[6], p[7], p[8], p[9], p[10], p[11]) * Pi(xi, p, para)
            #dau_dxi += aa_u4(xi, p[0], p[1], p[2], p[3], p[4], p[5], p[10]) * Pi(xi, p, para) # Same with above
        dau_dxis.append(dau_dxi)
get_dau_dxis()

dau_dxis = array(dau_dxis)
PMF = []

for r in xis:
    PMF.append(simps(dau_dxis[xis<=r], xis[xis<=r]))

o = open("%sth_order_free_energy.txt" % (O),'w')
for i,j in zip(xis, PMF):
    o.write('%s %s\n' % (i,j))
o.close()

o = open("%sth_order_mean_force.txt" % (O), 'w')
for i, j in zip(xis, dau_dxis):
	o.write("%s %s\n" % (i, j))
o.close()

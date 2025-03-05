import numpy as np
import math
import mpmath
import numba
from numba import jit
from math import factorial
import os
import psutil
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import pyfftw
import numexpr as ne
from astropy.io import fits

import pdb

import pickle
from joblib import Parallel, delayed

from scipy.interpolate import griddata

from ao_cockpit_syn import plt_plot
from ao_cockpit_syn import plt_imshow

from ao_cockpit_syn import plt_imshow_expa


import importlib

import ao_cockpit_syn as aou
importlib.reload(aou)

# from aotools.turbulence import ft_phase_screen



# file2exec='func_read.py'
# exec(open(file2exec).read())


# file2exec='func_forward_wfs.py'
# exec(open(file2exec).read())

def ca():
    plt.close('all')


file2exec='SYN_FUNC_EDGE.py'
exec(open(file2exec).read())


file2exec='FUNC_PYR_EDGE.py'
exec(open(file2exec).read())




def ONE_HF(TFcase):
    TFcase_0 = TFcase*0.
    TFcase_0[ SZ//2-2*idx_fc:SZ//2+2*idx_fc, SZ//2-2*idx_fc:SZ//2+2*idx_fc] =TFcase[ SZ//2-2*idx_fc:SZ//2+2*idx_fc, SZ//2-2*idx_fc:SZ//2+2*idx_fc]
    TFcase_0[ SZ//2-idx_fc:SZ//2+idx_fc , SZ//2-idx_fc:SZ//2+idx_fc] =0
    return TFcase_0

def ONE_LF(TFcase):
    TFcase_0 = TFcase*0.
    TFcase_0[ SZ//2-idx_fc:SZ//2+idx_fc , SZ//2-idx_fc:SZ//2+idx_fc] = TFcase[ SZ//2-idx_fc:SZ//2+idx_fc , SZ//2-idx_fc:SZ//2+idx_fc]
    return TFcase_0

###### DEFINITIONS
nwfs=4

diameter = 40.

sz = 1024
SZ = 2*sz

diam_max = 40.
dxo = diam_max/sz


diameter=38.542 # Note 'diameter' is indicative and is not applied on the 2D phase maps
cobs=0.28
GEO=aou.mkp(diam_max*2.0,SZ,diameter,cobs)

pupil=GEO.pupil[SZ//2-sz//2:SZ//2+sz//2,SZ//2-sz//2:SZ//2+sz//2]

RAD = np.sqrt(GEO.xx**2+GEO.yy**2)

PSD1, df1,P1 = aou.VK_DSP_up(sz*dxo, 1.0, 30.0, SZ, sz, 0., pupil)


ffx,ffy,ffr = build_spatial_freq(SZ,df1)

uu=ffx.copy()
vv=ffy.copy()


dd=4./(1./(2.*dxo))

fc = 1./(2.*dd)

idx_fc = np.min(np.where(ffx[SZ//2,SZ//2:]>=fc))



##### PYR SENSORS TF

modulation_angle = 3.0
TFx_pyr,TFy_pyr,M1O,M2O,M3O,M4O, M1,M2,M3,M4, Omega, PSF, Omega_0 = build_sensor_syn_c1(modulation_angle,SZ,sz,pupil=pupil,PSF=None,convolvPSF=True,TYPE=None)


TFx_pyr_dd = TFx_pyr*np.sinc(ffx*dd)*np.sinc(ffy*dd)
TFy_pyr_dd = TFy_pyr*np.sinc(ffx*dd)*np.sinc(ffy*dd)


TF_pyr_1 = 1j * (conv(M4,M1O) - conv(M1,M4O))
TF_pyr_2 = 1j * (conv(M3,M2O) - conv(M2,M3O))
TF_pyr_3 = 1j * (conv(M2,M3O) - conv(M3,M2O))
TF_pyr_4 = 1j * (conv(M1,M4O) - conv(M4,M1O))

# TFx_pyr,TFy_pyr,Mvp,Mvm,Mhp,Mhm, Omega, PSF, Omega = build_roofx2_syn_c1(modulation_angle,SZ,sz,pupil=pupil,PSF=None,convolvPSF=True,TYPE=None)


# MvpO = Mvp*Omega
# MvmO = Mvm*Omega
# MhpO = Mhp*Omega
# MhmO = Mhm*Omega

# TFx_pyr_dd = TFx_pyr*np.sinc(ffx*dd)*np.sinc(ffy*dd)
# TFy_pyr_dd = TFy_pyr*np.sinc(ffx*dd)*np.sinc(ffy*dd)


# TF_Vp_pyr = 1j * (conv(Mvp,MvmO) - conv(Mvm,MvpO))
# TF_Vm_pyr = 1j * (conv(Mvm,MvpO) - conv(Mvp,MvmO))


# TF_Hp_pyr = 1j * (conv(Mhp,MhmO) - conv(Mhm,MhpO))
# TF_Hm_pyr = 1j * (conv(Mhm,MhpO) - conv(Mhp,MhmO))

####

TF_pyr_1_dd = TF_pyr_1*np.sinc(ffx*dd)*np.sinc(ffy*dd)
TF_pyr_2_dd = TF_pyr_2*np.sinc(ffx*dd)*np.sinc(ffy*dd)
TF_pyr_3_dd = TF_pyr_3*np.sinc(ffx*dd)*np.sinc(ffy*dd)
TF_pyr_4_dd = TF_pyr_4*np.sinc(ffx*dd)*np.sinc(ffy*dd)

TFdd_pyr_n = np.zeros([SZ,SZ,nwfs],dtype=np.complex128)

TFdd_pyr_n[:,:,0] = TF_pyr_1_dd #*np.exp(-2j*np.pi*delta_*(uu*s_x[0]+vv*s_y[0]))
TFdd_pyr_n[:,:,1] = TF_pyr_2_dd #*np.exp(-2j*np.pi*delta_*(uu*s_x[1]+vv*s_y[1]))
TFdd_pyr_n[:,:,2] = TF_pyr_3_dd  #*np.exp(-2j*np.pi*delta_*(uu*s_x[2]+vv*s_y[2]))
TFdd_pyr_n[:,:,3] = TF_pyr_4_dd #*np.exp(-2j*np.pi*delta_*(uu*s_x[3]+vv*s_y[3]))

######## SUPER - RESOLUTION

def rshift(arr,a,b):
    return aou.myshift2D(arr,b,a)

                
px_1sd = np.int64(np.round(1./dd/df1))

s_x=np.asarray([-1,1,-1,1])
s_y=np.asarray([1,1,-1,-1])

delta_ = dd/4.

LF_ZONE = np.zeros([SZ,SZ])

LF_ZONE[SZ//2-idx_fc:SZ//2+idx_fc , SZ//2-idx_fc:SZ//2+idx_fc] = 1.0


nc=12


COR_S = np.zeros([SZ,SZ,nc])



x_COR = np.asarray([1.5,1.5,1.5,1.5,0.5,-0.5,-1.5,-1.5,-1.5,-1.5,-0.5,0.5])*idx_fc
y_COR = np.asarray([1.5,0.5,-0.5,-1.5,-1.5,-1.5,-1.5,-0.5,0.5,1.5,1.5,1.5])*idx_fc

px_1sd = np.int64(np.round(1./dd/df1))

s1sd_x = x_COR.copy()*px_1sd
s1sd_y = y_COR.copy()*px_1sd


for k in range(0,nc):
    cey = SZ//2 + np.int64(x_COR[k])
    cex = SZ//2 + np.int64(y_COR[k])
    COR_S[:,:,k][cex-idx_fc//2:cex+idx_fc//2 , cey-idx_fc//2:cey+idx_fc//2] = 1.0



HF_ZONE = np.sum(COR_S,axis=2)

nc2 = 6

COR_2S = np.zeros([SZ,SZ,nc2])

COR_2S[:,:,0] = COR_S[:,:,0] + COR_S[:,:,6]
COR_2S[:,:,1] = COR_S[:,:,1] + COR_S[:,:,7]
COR_2S[:,:,2] = COR_S[:,:,2] + COR_S[:,:,8]
COR_2S[:,:,3] = COR_S[:,:,3] + COR_S[:,:,9]
COR_2S[:,:,4] = COR_S[:,:,4] + COR_S[:,:,10]
COR_2S[:,:,5] = COR_S[:,:,5] + COR_S[:,:,11]

SZr = idx_fc*2*2*2

HF_ZONEr = HF_ZONE[ SZ//2-SZr//2 : SZ//2+SZr//2 , SZ//2-SZr//2 : SZ//2+SZr//2]

LF_ZONEr = LF_ZONE[ SZ//2-SZr//2 : SZ//2+SZr//2 , SZ//2-SZr//2 : SZ//2+SZr//2]

COR_2Sr = COR_2S[ SZ//2-SZr//2 : SZ//2+SZr//2 , SZ//2-SZr//2 : SZ//2+SZr//2 , :]

TFdd_pyr_nr = TFdd_pyr_n[SZ//2-SZr//2 : SZ//2+SZr//2 , SZ//2-SZr//2 : SZ//2+SZr//2,:]

### REC BF HF


S2M1_pyrr = np.zeros([SZr,SZr])

for k in range(0,nwfs):
    S2M1_pyrr = S2M1_pyrr + np.conj(TFdd_pyr_nr[:,:,k])*TFdd_pyr_nr[:,:,k]

S2M1_pyrr = (1./S2M1_pyrr).real

S2M1_pyrr[SZr//2,SZr//2] = 0.

##### PYR NO SUPER
s_x=np.asarray([-1,1,-1,1])*0.
s_y=np.asarray([1,1,-1,-1])*0.


HFinLFr_ = np.zeros([SZr,SZr,nwfs,nc2],dtype=np.complex128)

HFinLFr = np.zeros([SZr,SZr,nc2],dtype=np.complex128)

boundary = 1

for j in range(0,nc2):
    for k in range(0,nwfs):
        for n in range(-boundary, boundary+1 ):
            print(n, ' ', end='\r', flush=True)
            for m in range(-boundary, boundary+1 ):
                if (m!=0) or (n!=0) :
                    HFinLFr_[:,:,k,j] = np.conj(TFdd_pyr_nr[:,:,k])*rshift(TFdd_pyr_nr[:,:,k]*COR_2Sr[:,:,j],n*px_1sd,m*px_1sd)*np.exp(2j*np.pi*delta_/dd*(n*s_x[k]+m*s_y[k]))*LF_ZONEr
                    HFinLFr[:,:,j] = HFinLFr[:,:,j] + np.conj(TFdd_pyr_nr[:,:,k])*rshift(TFdd_pyr_nr[:,:,k]*COR_2Sr[:,:,j],n*px_1sd,m*px_1sd)*np.exp(2j*np.pi*delta_/dd*(n*s_x[k]+m*s_y[k]))*LF_ZONEr
                 
for j in range(nc2):
    print(k,np.max(np.abs(HFinLFr[:,:,j])))

CONF_PYR0 = HFinLFr*0.

for k in range(0,nc2):
    CONF_PYR0[:,:,k] = S2M1_pyrr*HFinLFr[:,:,k]


CONF_PYR0_2 = np.zeros([SZr,SZr])


for k in range(0,nc2):
    CONF_PYR0_2 = CONF_PYR0_2 + np.abs(CONF_PYR0[:,:,k])**2



CONF_PYR0_RMS = np.sqrt(CONF_PYR0_2)


##### PYR  SUPER
s_x=np.asarray([-1,1,-1,1])
s_y=np.asarray([1,1,-1,-1])


HFinLFr_ = np.zeros([SZr,SZr,nwfs,nc2],dtype=np.complex128)

HFinLFr = np.zeros([SZr,SZr,nc2],dtype=np.complex128)

boundary = 1

for j in range(0,nc2):
    for k in range(0,nwfs):
        for n in range(-boundary, boundary+1 ):
            print(n, ' ', end='\r', flush=True)
            for m in range(-boundary, boundary+1 ):
                if (m!=0) or (n!=0) :
                    HFinLFr_[:,:,k,j] = np.conj(TFdd_pyr_nr[:,:,k])*rshift(TFdd_pyr_nr[:,:,k]*COR_2Sr[:,:,j],n*px_1sd,m*px_1sd)*np.exp(2j*np.pi*delta_/dd*(n*s_x[k]+m*s_y[k]))*LF_ZONEr
                    HFinLFr[:,:,j] = HFinLFr[:,:,j] + np.conj(TFdd_pyr_nr[:,:,k])*rshift(TFdd_pyr_nr[:,:,k]*COR_2Sr[:,:,j],n*px_1sd,m*px_1sd)*np.exp(2j*np.pi*delta_/dd*(n*s_x[k]+m*s_y[k]))*LF_ZONEr
                 
for j in range(nc2):
    print(k,np.max(np.abs(HFinLFr[:,:,j])))

CONF_PYR1 = HFinLFr*0.

for k in range(0,nc2):
    CONF_PYR1[:,:,k] = S2M1_pyrr*HFinLFr[:,:,k]


CONF_PYR1_2 = np.zeros([SZr,SZr])


for k in range(0,nc2):
    CONF_PYR1_2 = CONF_PYR1_2 + np.abs(CONF_PYR1[:,:,k])**2



CONF_PYR1_RMS = np.sqrt(CONF_PYR1_2)


##### FIGURES
liim = idx_fc*2*df1

plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)


plt.figure()
plt.title('Confusion map for PYRAMID',fontsize=16)
plt.imshow(CONF_PYR0_RMS[SZr//2-idx_fc*2 : SZr//2+idx_fc*2 ,SZr//2-idx_fc*2 : SZr//2+idx_fc*2 ],origin='lower',extent=[-liim,liim,-liim,liim])
plt.xlabel('m^-1',fontsize=14)
plt.colorbar()
plt.show(block=False)

# plt.savefig('FIGURES_OK/SUPER_R/CONFUSION_MAP_PYR.pdf')

plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)


plt.figure()
plt.title('Aliasing map for PYRAMID with SUPER-R')
plt.imshow(CONF_PYR1_RMS[SZr//2-idx_fc*2 : SZr//2+idx_fc*2 ,SZr//2-idx_fc*2 : SZr//2+idx_fc*2 ],origin='lower',extent=[-liim,liim,-liim,liim])
plt.xlabel('m^-1')
plt.colorbar()
plt.show(block=False)

# plt.savefig('FIGURES_OK/SUPER_R/CONFUSION_MAP_PYR_SUPER_R.pdf')


TOT_PYR0 = np.sqrt(np.sum(CONF_PYR0_RMS**2*df1))

TOT_PYR1 = np.sqrt(np.sum(CONF_PYR1_RMS**2*df1))


##### FIGURES BIGGER
liim = idx_fc*2*df1/(1/dd)

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)


plt.figure()
plt.title('ALIASING: PYRAMID',fontsize=18)
plt.imshow(CONF_PYR0_RMS[SZr//2-idx_fc : SZr//2+idx_fc ,SZr//2-idx_fc : SZr//2+idx_fc ],origin='lower',extent=[-liim/2,liim/2,-liim/2,liim/2])
#plt.xlabel(r'$m^{-1}$',fontsize=18)
plt.xlabel('[1/d]',fontsize=15)
plt.colorbar()
plt.show(block=False)

# plt.savefig('FIGURES_OK/SUPER_R/CONFUSION_MAP_PYR_REV.pdf')

liim = idx_fc*2*df1/(1/dd)

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)


plt.figure()
plt.title('ALIASING: PYRAMID \n +SUPER-RESOLUTION',fontsize=18)
plt.imshow(CONF_PYR1_RMS[SZr//2-idx_fc : SZr//2+idx_fc ,SZr//2-idx_fc : SZr//2+idx_fc ],origin='lower',extent=[-liim/2,liim/2,-liim/2,liim/2])
#plt.xlabel(r'$m^{-1}$',fontsize=18)
plt.xlabel('[1/d]',fontsize=15)
plt.colorbar()
plt.show(block=False)

# plt.savefig('FIGURES_OK/SUPER_R/CONFUSION_MAP_PYR_SUPER_R_REV.pdf')

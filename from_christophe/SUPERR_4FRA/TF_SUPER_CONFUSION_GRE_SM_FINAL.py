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

#%%
importlib.reload(aou)

#%%

#from aotools.turbulence import ft_phase_screen



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

#%%


def ONE_HF(TFcase):
    TFcase_0 = TFcase*0.
    TFcase_0[ SZ//2-2*idx_fc:SZ//2+2*idx_fc, SZ//2-2*idx_fc:SZ//2+2*idx_fc] =TFcase[ SZ//2-2*idx_fc:SZ//2+2*idx_fc, SZ//2-2*idx_fc:SZ//2+2*idx_fc]
    TFcase_0[ SZ//2-idx_fc:SZ//2+idx_fc , SZ//2-idx_fc:SZ//2+idx_fc] =0
    return TFcase_0

def ONE_LF(TFcase):
    TFcase_0 = TFcase*0.
    TFcase_0[ SZ//2-idx_fc:SZ//2+idx_fc , SZ//2-idx_fc:SZ//2+idx_fc] = TFcase[ SZ//2-idx_fc:SZ//2+idx_fc , SZ//2-idx_fc:SZ//2+idx_fc]
    return TFcase_0

#%%

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



##### BIO SENSORS TF

modulation_angle = 3.0

TFx_bio,TFy_bio,Mvp,Mvm,Mhp,Mhm, Omega, PSF, Omega0 = build_roofx2_syn_c1(modulation_angle,SZ,sz,pupil=pupil,PSF=None,convolvPSF=True,TYPE=None)

TFx_bio0,TFy_bio0,Mvp0,Mvm0,Mhp0,Mhm0, Omega0, PSF0, Omega0 = build_roofx2_syn_c1(modulation_angle*0.,SZ,sz,pupil=pupil,PSF=None,convolvPSF=True,TYPE=None)

modu_eff_blunt = modulation_angle/np.sqrt(2.0)

TFx_bio,TFy_bio,Mvp_,Mvm_,Mhp_,Mhm_, Omega_gre, PSF_gre, Omega0_gre =  build_bio_blunt_syn_c1(modu_eff_blunt,0.,SZ,sz,pupil=pupil,PSF=None,convolvPSF=True,TYPE=None)

Mvp = Mvp0.copy()
Mvm = Mvm0.copy()
Mhp = Mhp0.copy()
Mhm = Mhm0.copy()

aa = np.int64(modulation_angle*2*SZ/sz)

# Mvp[SZ//2-aa//2:SZ//2+aa//2 , SZ//2-aa//2:SZ//2+aa//2 ] = Mvp_[SZ//2-aa//2:SZ//2+aa//2 , SZ//2-aa//2:SZ//2+aa//2 ] 

# Mvm[SZ//2-aa//2:SZ//2+aa//2 , SZ//2-aa//2:SZ//2+aa//2 ] = Mvm_[SZ//2-aa//2:SZ//2+aa//2 , SZ//2-aa//2:SZ//2+aa//2 ] 

# Mhp[SZ//2-aa//2:SZ//2+aa//2 , SZ//2-aa//2:SZ//2+aa//2 ] = Mhp_[SZ//2-aa//2:SZ//2+aa//2 , SZ//2-aa//2:SZ//2+aa//2 ] 

# Mhm[SZ//2-aa//2:SZ//2+aa//2 , SZ//2-aa//2:SZ//2+aa//2 ] = Mhp_[SZ//2-aa//2:SZ//2+aa//2 , SZ//2-aa//2:SZ//2+aa//2 ] 

Mvp[SZ//2-aa//2:SZ//2+aa//2-1 , SZ//2-aa//2:SZ//2+aa//2 ] = Mvp_[SZ//2-aa//2:SZ//2+aa//2-1 , SZ//2-aa//2:SZ//2+aa//2 ] 

Mvm[SZ//2-aa//2:SZ//2+aa//2-1 , SZ//2-aa//2:SZ//2+aa//2 ] = Mvm_[SZ//2-aa//2:SZ//2+aa//2-1 , SZ//2-aa//2:SZ//2+aa//2 ] 

Mhp[SZ//2-aa//2:SZ//2+aa//2 , SZ//2-aa//2:SZ//2+aa//2-1 ] = Mhp_[SZ//2-aa//2:SZ//2+aa//2 , SZ//2-aa//2:SZ//2+aa//2-1 ] 

Mhm[SZ//2-aa//2:SZ//2+aa//2 , SZ//2-aa//2:SZ//2+aa//2-1 ] = Mhm_[SZ//2-aa//2:SZ//2+aa//2 , SZ//2-aa//2:SZ//2+aa//2-1 ] 

MvpO = Mvp*Omega_gre
MvmO = Mvm*Omega_gre
MhpO = Mhp*Omega_gre
MhmO = Mhm*Omega_gre

TFx_bio_dd = TFx_bio*np.sinc(ffx*dd)*np.sinc(ffy*dd)
TFy_bio_dd = TFy_bio*np.sinc(ffx*dd)*np.sinc(ffy*dd)


TF_Vp_bio = 1j * (conv(Mvp,MvmO) - conv(Mvm,MvpO))
TF_Vm_bio = 1j * (conv(Mvm,MvpO) - conv(Mvp,MvmO))


TF_Hp_bio = 1j * (conv(Mhp,MhmO) - conv(Mhm,MhpO))
TF_Hm_bio = 1j * (conv(Mhm,MhpO) - conv(Mhp,MhmO))

####

TF_Vp_bio_dd = TF_Vp_bio*np.sinc(ffx*dd)*np.sinc(ffy*dd)
TF_Vm_bio_dd = TF_Vm_bio*np.sinc(ffx*dd)*np.sinc(ffy*dd)
TF_Hp_bio_dd = TF_Hp_bio*np.sinc(ffx*dd)*np.sinc(ffy*dd)
TF_Hm_bio_dd = TF_Hm_bio*np.sinc(ffx*dd)*np.sinc(ffy*dd)

TFdd_bio_n = np.zeros([SZ,SZ,nwfs],dtype=np.complex128)

TFdd_bio_n[:,:,0] = TF_Vp_bio_dd #*np.exp(-2j*np.pi*delta_*(uu*s_x[0]+vv*s_y[0]))
TFdd_bio_n[:,:,1] = TF_Hp_bio_dd #*np.exp(-2j*np.pi*delta_*(uu*s_x[1]+vv*s_y[1]))
TFdd_bio_n[:,:,2] = TF_Hm_bio_dd  #*np.exp(-2j*np.pi*delta_*(uu*s_x[2]+vv*s_y[2]))
TFdd_bio_n[:,:,3] = TF_Vm_bio_dd #*np.exp(-2j*np.pi*delta_*(uu*s_x[3]+vv*s_y[3]))

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

TFdd_bio_nr = TFdd_bio_n[SZ//2-SZr//2 : SZ//2+SZr//2 , SZ//2-SZr//2 : SZ//2+SZr//2,:]

### REC BF HF


S2M1_bior = np.zeros([SZr,SZr])

for k in range(0,nwfs):
    S2M1_bior = S2M1_bior + np.conj(TFdd_bio_nr[:,:,k])*TFdd_bio_nr[:,:,k]

S2M1_bior = (1./S2M1_bior).real

S2M1_bior[SZr//2,SZr//2] = 0.

##### BIO NO SUPER
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
                    HFinLFr_[:,:,k,j] = np.conj(TFdd_bio_nr[:,:,k])*rshift(TFdd_bio_nr[:,:,k]*COR_2Sr[:,:,j],n*px_1sd,m*px_1sd)*np.exp(2j*np.pi*delta_/dd*(n*s_x[k]+m*s_y[k]))*LF_ZONEr
                    HFinLFr[:,:,j] = HFinLFr[:,:,j] + np.conj(TFdd_bio_nr[:,:,k])*rshift(TFdd_bio_nr[:,:,k]*COR_2Sr[:,:,j],n*px_1sd,m*px_1sd)*np.exp(2j*np.pi*delta_/dd*(n*s_x[k]+m*s_y[k]))*LF_ZONEr
                 
for j in range(nc2):
    print(k,np.max(np.abs(HFinLFr[:,:,j])))

CONF_BIO0 = HFinLFr*0.

for k in range(0,nc2):
    CONF_BIO0[:,:,k] = S2M1_bior*HFinLFr[:,:,k]


CONF_BIO0_2 = np.zeros([SZr,SZr])


for k in range(0,nc2):
    CONF_BIO0_2 = CONF_BIO0_2 + np.abs(CONF_BIO0[:,:,k])**2



CONF_BIO0_RMS = np.sqrt(CONF_BIO0_2)


##### BIO  SUPER
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
                    HFinLFr_[:,:,k,j] = np.conj(TFdd_bio_nr[:,:,k])*rshift(TFdd_bio_nr[:,:,k]*COR_2Sr[:,:,j],n*px_1sd,m*px_1sd)*np.exp(2j*np.pi*delta_/dd*(n*s_x[k]+m*s_y[k]))*LF_ZONEr
                    HFinLFr[:,:,j] = HFinLFr[:,:,j] + np.conj(TFdd_bio_nr[:,:,k])*rshift(TFdd_bio_nr[:,:,k]*COR_2Sr[:,:,j],n*px_1sd,m*px_1sd)*np.exp(2j*np.pi*delta_/dd*(n*s_x[k]+m*s_y[k]))*LF_ZONEr
                 
for j in range(nc2):
    print(k,np.max(np.abs(HFinLFr[:,:,j])))

CONF_BIO1 = HFinLFr*0.

for k in range(0,nc2):
    CONF_BIO1[:,:,k] = S2M1_bior*HFinLFr[:,:,k]


CONF_BIO1_2 = np.zeros([SZr,SZr])


for k in range(0,nc2):
    CONF_BIO1_2 = CONF_BIO1_2 + np.abs(CONF_BIO1[:,:,k])**2



CONF_BIO1_RMS = np.sqrt(CONF_BIO1_2)


##### FIGURES
liim = idx_fc*2*df1

plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)


plt.figure()
plt.title('Confusion map for BIOEDGE')
plt.imshow(CONF_BIO0_RMS[SZr//2-idx_fc*2 : SZr//2+idx_fc*2 ,SZr//2-idx_fc*2 : SZr//2+idx_fc*2 ],origin='lower',extent=[-liim,liim,-liim,liim])
plt.xlabel('m^-1')
plt.colorbar()
plt.show(block=False)

#plt.savefig('FIGURES_OK/SUPER_R/CONFUSION_MAP_BIO.pdf')

plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)


plt.figure()
plt.title('Confusion map for BIOEDGE with SUPER-R')
plt.imshow(CONF_BIO1_RMS[SZr//2-idx_fc*2 : SZr//2+idx_fc*2 ,SZr//2-idx_fc*2 : SZr//2+idx_fc*2 ],origin='lower',extent=[-liim,liim,-liim,liim])
plt.xlabel('m^-1')
plt.colorbar()
plt.show(block=False)

#plt.savefig('FIGURES_OK/SUPER_R/CONFUSION_MAP_BIO_SUPER_R.pdf')


TOT_BIO0 = np.sqrt(np.sum(CONF_BIO0_RMS**2*df1))

TOT_BIO1 = np.sqrt(np.sum(CONF_BIO1_RMS**2*df1))

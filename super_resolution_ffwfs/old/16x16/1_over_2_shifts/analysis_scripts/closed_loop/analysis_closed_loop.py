# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 16:19:11 2025

@author: fleroux
"""
#%%

import pathlib
import dill

import numpy as np
import matplotlib.pyplot as plt

from OOPAO.tools.displayTools import cl_plot

#%%

path = pathlib.Path(__file__).parent.parent.parent

#%% load objects

path_object = path / 'data_object'
filename_object = 'object_sr_noise_prop_I2_band_16x16_KL_basis.pkl'

dill.load_session(path_object / filename_object)

#%% load calibrations

pathlib.PosixPath = pathlib.WindowsPath

path_calibration = path / 'data_calibration'
filename_calibration = 'calibration_gbioedge_sr_noise_prop_I2_band_16x16_KL_basis.pkl'

dill.load_session(path_calibration / filename_calibration)

#%% Reconstructor

n_modes_to_keep = int(gbioedge.nSignal/4 - 50)

R = np.linalg.pinv(calib_gbioedge.D[:,:n_modes_to_keep])

#%%

reconstructor = M2C[:, :n_modes_to_keep]@R

#%% Setup

tel.computePSF()

loop_gain = 0.5
n_iter = 1000

total = np.zeros(n_iter)
residual = np.zeros(n_iter)
strehl = np.zeros(n_iter)

atm.initializeAtmosphere(tel)
tel+atm

dm.coefs = 0
gbioedge_measure = 0*gbioedge.signal

ngs*tel*dm*gbioedge

#%% Close the loop

n = 200

SE_PSF = []
LE_PSF = np.log10(tel.PSF)[n:-n,n:-n]

plot_obj = cl_plot(list_fig          = [atm.OPD,tel.mean_removed_OPD,gbioedge.cam.frame,[[0,0],[0,0]],[dm.coordinates[:,0],np.flip(dm.coordinates[:,1]),dm.coefs],
                                        np.log10(tel.PSF),np.log10(tel.PSF)],\
                        type_fig          = ['imshow','imshow','imshow','plot','scatter','imshow','imshow'],\
                        list_title        = ['Turbulence OPD','Residual OPD','bio Detector',None,None,None,None],\
                        list_lim          = [None,None,None,None,None,[2,6],[2,6]],\
                        list_label        = [None,None,None,['Time','WFE [nm]'],['DM Commands',''],['Short Exposure PSF',''],['Long Exposure_PSF','']],\
                        n_subplot         = [4,2],\
                        list_display_axis = [None,None,None,True,None,None,None],\
                        list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)

for k in range(n_iter):
    
    atm.update()
    total[k] = np.std(tel.OPD[np.where(tel.pupil==1)])
    phase_turb = tel.src.phase
    
    tel*dm*gbioedge
    
    gbioedge_measure = gbioedge.signal # tune delay (1 frames here)
    
    dm.coefs = dm.coefs - loop_gain * np.matmul(reconstructor, gbioedge_measure)
    
    # gbioedge_measure = gbioedge.signal # tune delay (2 frames here)
    
    strehl[k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
    
    
    if k>15:
        tel.computePSF(4)
        
        SE_PSF.append(np.log10(tel.PSF)[n:-n,n:-n])
        LE_PSF = np.mean(SE_PSF, axis=0)
        cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD, gbioedge.cam.frame,[np.arange(k+1),residual[:k+1]],dm.coefs,SE_PSF[-1], LE_PSF],
                list_lim =[None,None,None,None,None,[SE_PSF[-1].max()-4,SE_PSF[-1].max()],[LE_PSF.max()-4,LE_PSF.max()]],
                plt_obj = plot_obj)
        plt.pause(0.1)
        
        if plot_obj.keep_going is False:
                break
            
    strehl[k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
    residual[k]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
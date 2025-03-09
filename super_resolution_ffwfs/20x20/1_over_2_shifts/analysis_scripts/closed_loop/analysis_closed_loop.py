# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 16:19:11 2025

@author: fleroux
"""
#%%

import pathlib
import sys
import platform
import dill

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from OOPAO.tools.displayTools import cl_plot

#%%

path = pathlib.Path(__file__).parent.parent.parent # location of parameter_file.py

#%% import parameter file from a distinct repository than the one of this file (relou)

# weird method from https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path

import importlib.util
import sys
spec = importlib.util.spec_from_file_location("get_parameters", path / "parameter_file.py")
foo = importlib.util.module_from_spec(spec)
sys.modules["parameter_file"] = foo
spec.loader.exec_module(foo)

#%%

param = foo.get_parameters()

#%% path type compatibility issues

if platform.system() == 'Windows':
    temp = deepcopy(pathlib.PosixPath)
    pathlib.PosixPath = pathlib.WindowsPath
elif platform.system() == 'Linux':
    temp = deepcopy(pathlib.WindowsPath)
    pathlib.WindowsPath = pathlib.PosixPath

#%% load objects

dill.load_session(param['path_object'] / pathlib.Path('object'+str(param['filename'])+'.pkl'))

#%% load calibrations

dill.load_session(param['path_calibration'] / pathlib.Path('calibration_gbioedge'+str(param['filename'])+'.pkl'))

#%% reconstructors

n_modes_to_keep_gbioedge = 250 #int(gbioedge.nSignal/4 - 50)
n_modes_to_keep_gbioedge_sr = 850
n_modes_to_keep_gbioedge_oversampled = 980

R_gbioedge = np.linalg.pinv(calib_gbioedge.D[:,:n_modes_to_keep_gbioedge])
R_gbioedge_sr = np.linalg.pinv(calib_gbioedge_sr.D[:,:n_modes_to_keep_gbioedge_sr])
R_gbioedge_oversampled = np.linalg.pinv(calib_gbioedge_oversampled.D[:,:n_modes_to_keep_gbioedge_oversampled])

reconstructor_gbioedge = M2C[:, :n_modes_to_keep_gbioedge]@R_gbioedge
reconstructor_gbioedge_sr = M2C[:, :n_modes_to_keep_gbioedge_sr]@R_gbioedge_sr
reconstructor_gbioedge_oversampled = M2C[:, :n_modes_to_keep_gbioedge_oversampled]@R_gbioedge_oversampled

#%% Close the loop - gbioedge

# Setup

tel.computePSF()

loop_gain = 0.5
n_iter = 2000

total_gbioedge = np.zeros(n_iter)
residual_gbioedge = np.zeros(n_iter)
strehl_gbioedge = np.zeros(n_iter)

atm.initializeAtmosphere(tel)
tel+atm

dm.coefs = 0
gbioedge_measure = 0*gbioedge.signal

ngs*tel*dm*gbioedge

n = 200

SE_PSF = []
LE_PSF = np.log10(tel.PSF)[n:-n,n:-n]

plot_obj = cl_plot(list_fig          = [atm.OPD,tel.mean_removed_OPD,gbioedge.cam.frame,[[0,0],[0,0]],
                                        [dm.coordinates[:,0],np.flip(dm.coordinates[:,1]),dm.coefs],
                                        np.log10(tel.PSF),np.log10(tel.PSF)],\
                        type_fig          = ['imshow','imshow','imshow','plot','scatter','imshow','imshow'],\
                        list_title        = ['Turbulence OPD','Residual OPD','bio Detector',None,None,None,None],\
                        list_lim          = [None,None,None,None,None,[2,6],[2,6]],\
                        list_label        = [None,None,None,['Time','WFE [nm]'],['DM Commands',''],['Short Exposure PSF',''],
                                             ['Long Exposure_PSF','']],\
                        n_subplot         = [4,2],\
                        list_display_axis = [None,None,None,True,None,None,None],\
                        list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)
    

display = False

for k in range(n_iter):
    
    atm.update()
    total_gbioedge[k] = np.std(tel.OPD[np.where(tel.pupil==1)])
    phase_turb = tel.src.phase
    
    tel*dm*gbioedge
    
    gbioedge_measure = gbioedge.signal # tune delay (1 frames here)
    
    dm.coefs = dm.coefs - loop_gain * np.matmul(reconstructor_gbioedge, gbioedge_measure)
    
    # gbioedge_measure = gbioedge.signal # tune delay (2 frames here)
    
    strehl_gbioedge[k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
    
    
    if k>15 and display:
        tel.computePSF(4)
        
        SE_PSF.append(np.log10(tel.PSF)[n:-n,n:-n])
        LE_PSF = np.mean(SE_PSF, axis=0)
        cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD, gbioedge.cam.frame,[np.arange(k+1),residual_gbioedge[:k+1]],dm.coefs,SE_PSF[-1], LE_PSF],
                list_lim =[None,None,None,None,None,[SE_PSF[-1].max()-4,SE_PSF[-1].max()],[LE_PSF.max()-4,LE_PSF.max()]],plt_obj = plot_obj)
        plt.pause(0.1)
        if plot_obj.keep_going is False:
                break
            
    strehl_gbioedge[k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
    residual_gbioedge[k]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    
    
#%% Close the loop - gbioedge_sr

# Setup

tel.computePSF()

loop_gain = 0.5
n_iter = 2000

total_gbioedge_sr = np.zeros(n_iter)
residual_gbioedge_sr = np.zeros(n_iter)
strehl_gbioedge_sr = np.zeros(n_iter)

atm.initializeAtmosphere(tel)
tel+atm

dm.coefs = 0
gbioedge_sr_measure = 0*gbioedge_sr.signal

ngs*tel*dm*gbioedge_sr

n = 200

SE_PSF = []
LE_PSF = np.log10(tel.PSF)[n:-n,n:-n]

plot_obj = cl_plot(list_fig          = [atm.OPD,tel.mean_removed_OPD,gbioedge_sr.cam.frame,[[0,0],[0,0]],
                                        [dm.coordinates[:,0],np.flip(dm.coordinates[:,1]),dm.coefs],
                                        np.log10(tel.PSF),np.log10(tel.PSF)],\
                        type_fig          = ['imshow','imshow','imshow','plot','scatter','imshow','imshow'],\
                        list_title        = ['Turbulence OPD','Residual OPD','bio Detector',None,None,None,None],\
                        list_lim          = [None,None,None,None,None,[2,6],[2,6]],\
                        list_label        = [None,None,None,['Time','WFE [nm]'],['DM Commands',''],['Short Exposure PSF',''],
                                             ['Long Exposure_PSF','']],\
                        n_subplot         = [4,2],\
                        list_display_axis = [None,None,None,True,None,None,None],\
                        list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)
    

display = False

for k in range(n_iter):
    
    atm.update()
    total_gbioedge_sr[k] = np.std(tel.OPD[np.where(tel.pupil==1)])
    phase_turb = tel.src.phase
    
    tel*dm*gbioedge_sr
    
    gbioedge_sr_measure = gbioedge_sr.signal # tune delay (1 frames here)
    
    dm.coefs = dm.coefs - loop_gain * np.matmul(reconstructor_gbioedge_sr, gbioedge_sr_measure)
    
    # gbioedge_sr_measure = gbioedge_sr.signal # tune delay (2 frames here)
    
    strehl_gbioedge_sr[k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
    
    
    if k>15 and display:
        tel.computePSF(4)
        
        SE_PSF.append(np.log10(tel.PSF)[n:-n,n:-n])
        LE_PSF = np.mean(SE_PSF, axis=0)
        cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD, gbioedge_sr.cam.frame,[np.arange(k+1),residual_gbioedge_sr[:k+1]],dm.coefs,SE_PSF[-1], LE_PSF],
                list_lim =[None,None,None,None,None,[SE_PSF[-1].max()-4,SE_PSF[-1].max()],[LE_PSF.max()-4,LE_PSF.max()]],plt_obj = plot_obj)
        plt.pause(0.1)
        if plot_obj.keep_going is False:
                break
            
    strehl_gbioedge_sr[k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
    residual_gbioedge_sr[k]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    
    
#%%

plt.plot(residual_gbioedge, 'b')
plt.plot(residual_gbioedge_sr, 'r')
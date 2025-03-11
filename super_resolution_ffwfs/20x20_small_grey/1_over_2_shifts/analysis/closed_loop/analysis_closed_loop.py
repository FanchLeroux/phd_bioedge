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

#%% Load parameter file

param = foo.get_parameters()

#%% Load calibration

path_calibration = param['path_calibration']

dill.load_session(path_calibration / pathlib.Path('calibration_gbioedge'+str(param['filename'])+'.pkl'))
dill.load_session(path_calibration / pathlib.Path('calibration_sgbioedge'+str(param['filename'])+'.pkl'))

#%% Load parameter file

param = foo.get_parameters()

#%% reconstructors - gbioedge

M2C_original = deepcopy(M2C)
M2C = np.load(path/"M2C.npy")

M2C_sr = deepcopy(M2C)
M2C_sr[:, param['param['n_modes_to_control_sr']']:] = 0.

# ------------------- gbioedge -------------------- #

R_gbioedge = np.linalg.pinv(calib_gbioedge.D[:,:param['n_modes_to_show']])
R_gbioedge_sr = np.linalg.pinv(calib_gbioedge_sr.D[:,:param['n_modes_to_show_sr']])
R_gbioedge_oversampled = np.linalg.pinv(calib_gbioedge_oversampled.D[:,:param['n_modes_to_show_oversampled']])

reconstructor_gbioedge = M2C[:, :param['n_modes_to_show']]@R_gbioedge
reconstructor_gbioedge_sr = M2C_sr[:, :param['n_modes_to_show_sr']]@R_gbioedge_sr
reconstructor_gbioedge_oversampled = M2C[:, :param['n_modes_to_show_oversampled']]@R_gbioedge_oversampled

#%% reconstructors - sgbioedge

#M2C = np.load(path/"M2C.npy")
M2C = M2C_original

M2C_sr = deepcopy(M2C)
M2C_sr[:, param['param['n_modes_to_control_sr']']:] = 0.

# ------------------- sgbioedge -------------------- #

R_sgbioedge = np.linalg.pinv(calib_sgbioedge.D[:,:param['n_modes_to_show']])
R_sgbioedge_sr = np.linalg.pinv(calib_sgbioedge_sr.D[:,:param['n_modes_to_show_sr']])
R_sgbioedge_oversampled = np.linalg.pinv(calib_sgbioedge_oversampled.D[:,:param['n_modes_to_show_oversampled']])

reconstructor_sgbioedge = M2C[:, :param['n_modes_to_show']]@R_sgbioedge
reconstructor_sgbioedge_sr = M2C_sr[:, :param['n_modes_to_show_sr']]@R_sgbioedge_sr
reconstructor_sgbioedge_oversampled = M2C[:, :param['n_modes_to_show_oversampled']]@R_sgbioedge_oversampled

#%% check
# dm.coefs = M2C[:,10]*1e-9

# ngs*tel*dm*sgbioedge

# plt.figure(),plt.plot(R_sgbioedge@sgbioedge.signal)

# #%%
# dm.coefs = M2C[:,:20]

# from OOPAO.tools.displayTools import displayMap
# displayMap(dm.OPD)


#%%
seed = 10 # 0 is bad, 10 is ok

#%% Close the loop - gbioedge

# Setup

tel.computePSF()

total_gbioedge = np.zeros(param['n_iter'])
residual_gbioedge = np.zeros(param['n_iter'])
strehl_gbioedge = np.zeros(param['n_iter'])

atm.initializeAtmosphere(tel)
atm.generateNewPhaseScreen(seed = seed)
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

for k in range(param['n_iter']):
    
    atm.update()
    total_gbioedge[k] = np.std(tel.OPD[np.where(tel.pupil==1)])
    phase_turb = tel.src.phase
    
    tel*dm*gbioedge
    
    gbioedge_measure = gbioedge.signal # tune delay (1 frames here)
    
    dm.coefs = dm.coefs - param['loop_gain'] * np.matmul(reconstructor_gbioedge, gbioedge_measure)
    
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

#param['loop_gain'] = 0.5
#param['n_iter'] = 2000

total_gbioedge_sr = np.zeros(param['n_iter'])
residual_gbioedge_sr = np.zeros(param['n_iter'])
strehl_gbioedge_sr = np.zeros(param['n_iter'])

atm.initializeAtmosphere(tel)
atm.generateNewPhaseScreen(seed = seed)
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

for k in range(param['n_iter']):
    
    atm.update()
    total_gbioedge_sr[k] = np.std(tel.OPD[np.where(tel.pupil==1)])
    phase_turb = tel.src.phase
    
    tel*dm*gbioedge_sr
    
    gbioedge_sr_measure = gbioedge_sr.signal # tune delay (1 frames here)
    
    dm.coefs = dm.coefs - param['loop_gain'] * np.matmul(reconstructor_gbioedge_sr, gbioedge_sr_measure)
    
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
    
#%% Close the loop - gbioedge_oversampled

ref = True

if ref:

# Setup

    tel.computePSF()

    #param['loop_gain'] = 0.5
    #param['n_iter'] = 2000
    
    total_gbioedge_oversampled = np.zeros(param['n_iter'])
    residual_gbioedge_oversampled = np.zeros(param['n_iter'])
    strehl_gbioedge_oversampled = np.zeros(param['n_iter'])
    
    atm.initializeAtmosphere(tel)
    atm.generateNewPhaseScreen(seed = seed)
    tel+atm
    
    dm.coefs = 0
    gbioedge_oversampled_measure = 0*gbioedge_oversampled.signal
    
    ngs*tel*dm*gbioedge_oversampled
    
    n = 200
    
    SE_PSF = []
    LE_PSF = np.log10(tel.PSF)[n:-n,n:-n]
    
    plot_obj = cl_plot(list_fig          = [atm.OPD,tel.mean_removed_OPD,gbioedge_oversampled.cam.frame,[[0,0],[0,0]],
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
    
    for k in range(param['n_iter']):
        
        atm.update()
        total_gbioedge_oversampled[k] = np.std(tel.OPD[np.where(tel.pupil==1)])
        phase_turb = tel.src.phase
        
        tel*dm*gbioedge_oversampled
        
        gbioedge_oversampled_measure = gbioedge_oversampled.signal # tune delay (1 frames here)
        
        dm.coefs = dm.coefs - param['loop_gain'] * np.matmul(reconstructor_gbioedge_oversampled, gbioedge_oversampled_measure)
        
        # gbioedge_oversampled_measure = gbioedge_oversampled.signal # tune delay (2 frames here)
        
        strehl_gbioedge_oversampled[k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
        
        
        if k>15 and display:
            tel.computePSF(4)
            
            SE_PSF.append(np.log10(tel.PSF)[n:-n,n:-n])
            LE_PSF = np.mean(SE_PSF, axis=0)
            cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD, gbioedge_oversampled.cam.frame,[np.arange(k+1),residual_gbioedge_oversampled[:k+1]],dm.coefs,SE_PSF[-1], LE_PSF],
                    list_lim =[None,None,None,None,None,[SE_PSF[-1].max()-4,SE_PSF[-1].max()],[LE_PSF.max()-4,LE_PSF.max()]],plt_obj = plot_obj)
            plt.pause(0.1)
            if plot_obj.keep_going is False:
                    break
                
        strehl_gbioedge_oversampled[k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
        residual_gbioedge_oversampled[k]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9    

#%% Close the loop - sgbioedge

# Setup

tel.computePSF()

total_sgbioedge = np.zeros(param['n_iter'])
residual_sgbioedge = np.zeros(param['n_iter'])
strehl_sgbioedge = np.zeros(param['n_iter'])

atm.initializeAtmosphere(tel)
atm.generateNewPhaseScreen(seed = seed)
tel+atm

dm.coefs = 0
sgbioedge_measure = 0*sgbioedge.signal

ngs*tel*dm*sgbioedge

n = 200

SE_PSF = []
LE_PSF = np.log10(tel.PSF)[n:-n,n:-n]

plot_obj = cl_plot(list_fig          = [atm.OPD,tel.mean_removed_OPD,sgbioedge.cam.frame,[[0,0],[0,0]],
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

for k in range(param['n_iter']):
    
    atm.update()
    total_sgbioedge[k] = np.std(tel.OPD[np.where(tel.pupil==1)])
    phase_turb = tel.src.phase
    
    tel*dm*sgbioedge
    
    sgbioedge_measure = sgbioedge.signal # tune delay (1 frames here)
    
    dm.coefs = dm.coefs - param['loop_gain'] * np.matmul(reconstructor_sgbioedge, sgbioedge_measure)
    
    # sgbioedge_measure = sgbioedge.signal # tune delay (2 frames here)
    
    strehl_sgbioedge[k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
    
    
    if k>15 and display:
        tel.computePSF(4)
        
        SE_PSF.append(np.log10(tel.PSF)[n:-n,n:-n])
        LE_PSF = np.mean(SE_PSF, axis=0)
        cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD, sgbioedge.cam.frame,[np.arange(k+1),residual_sgbioedge[:k+1]],dm.coefs,SE_PSF[-1], LE_PSF],
                list_lim =[None,None,None,None,None,[SE_PSF[-1].max()-4,SE_PSF[-1].max()],[LE_PSF.max()-4,LE_PSF.max()]],plt_obj = plot_obj)
        plt.pause(0.1)
        if plot_obj.keep_going is False:
                break
            
    strehl_sgbioedge[k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
    residual_sgbioedge[k]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    
    
#%% Close the loop - sgbioedge_sr

# Setup

tel.computePSF()

#param['loop_gain'] = 0.5
#param['n_iter'] = 2000

total_sgbioedge_sr = np.zeros(param['n_iter'])
residual_sgbioedge_sr = np.zeros(param['n_iter'])
strehl_sgbioedge_sr = np.zeros(param['n_iter'])

atm.initializeAtmosphere(tel)
atm.generateNewPhaseScreen(seed = seed)
tel+atm

dm.coefs = 0
sgbioedge_sr_measure = 0*sgbioedge_sr.signal

ngs*tel*dm*sgbioedge_sr

n = 200

SE_PSF = []
LE_PSF = np.log10(tel.PSF)[n:-n,n:-n]

plot_obj = cl_plot(list_fig          = [atm.OPD,tel.mean_removed_OPD,sgbioedge_sr.cam.frame,[[0,0],[0,0]],
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

for k in range(param['n_iter']):
    
    atm.update()
    total_sgbioedge_sr[k] = np.std(tel.OPD[np.where(tel.pupil==1)])
    phase_turb = tel.src.phase
    
    tel*dm*sgbioedge_sr
    
    sgbioedge_sr_measure = sgbioedge_sr.signal # tune delay (1 frames here)
    
    dm.coefs = dm.coefs - param['loop_gain'] * np.matmul(reconstructor_sgbioedge_sr, sgbioedge_sr_measure)
    
    # sgbioedge_sr_measure = sgbioedge_sr.signal # tune delay (2 frames here)
    
    strehl_sgbioedge_sr[k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
    
    
    if k>15 and display:
        tel.computePSF(4)
        
        SE_PSF.append(np.log10(tel.PSF)[n:-n,n:-n])
        LE_PSF = np.mean(SE_PSF, axis=0)
        cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD, sgbioedge_sr.cam.frame,[np.arange(k+1),residual_sgbioedge_sr[:k+1]],dm.coefs,SE_PSF[-1], LE_PSF],
                list_lim =[None,None,None,None,None,[SE_PSF[-1].max()-4,SE_PSF[-1].max()],[LE_PSF.max()-4,LE_PSF.max()]],plt_obj = plot_obj)
        plt.pause(0.1)
        if plot_obj.keep_going is False:
                break
            
    strehl_sgbioedge_sr[k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
    residual_sgbioedge_sr[k]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    
#%% Close the loop - sgbioedge_oversampled

ref = True

if ref:

# Setup

    tel.computePSF()

    #param['loop_gain'] = 0.5
    #param['n_iter'] = 2000
    
    total_sgbioedge_oversampled = np.zeros(param['n_iter'])
    residual_sgbioedge_oversampled = np.zeros(param['n_iter'])
    strehl_sgbioedge_oversampled = np.zeros(param['n_iter'])
    
    atm.initializeAtmosphere(tel)
    atm.generateNewPhaseScreen(seed = seed)
    tel+atm
    
    dm.coefs = 0
    sgbioedge_oversampled_measure = 0*sgbioedge_oversampled.signal
    
    ngs*tel*dm*sgbioedge_oversampled
    
    n = 200
    
    SE_PSF = []
    LE_PSF = np.log10(tel.PSF)[n:-n,n:-n]
    
    plot_obj = cl_plot(list_fig          = [atm.OPD,tel.mean_removed_OPD,sgbioedge_oversampled.cam.frame,[[0,0],[0,0]],
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
    
    for k in range(param['n_iter']):
        
        atm.update()
        total_sgbioedge_oversampled[k] = np.std(tel.OPD[np.where(tel.pupil==1)])
        phase_turb = tel.src.phase
        
        tel*dm*sgbioedge_oversampled
        
        sgbioedge_oversampled_measure = sgbioedge_oversampled.signal # tune delay (1 frames here)
        
        dm.coefs = dm.coefs - param['loop_gain'] * np.matmul(reconstructor_sgbioedge_oversampled, sgbioedge_oversampled_measure)
        
        # sgbioedge_oversampled_measure = sgbioedge_oversampled.signal # tune delay (2 frames here)
        
        strehl_sgbioedge_oversampled[k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
        
        
        if k>15 and display:
            tel.computePSF(4)
            
            SE_PSF.append(np.log10(tel.PSF)[n:-n,n:-n])
            LE_PSF = np.mean(SE_PSF, axis=0)
            cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD, sgbioedge_oversampled.cam.frame,[np.arange(k+1),residual_sgbioedge_oversampled[:k+1]],dm.coefs,SE_PSF[-1], LE_PSF],
                    list_lim =[None,None,None,None,None,[SE_PSF[-1].max()-4,SE_PSF[-1].max()],[LE_PSF.max()-4,LE_PSF.max()]],plt_obj = plot_obj)
            plt.pause(0.1)
            if plot_obj.keep_going is False:
                    break
                
        strehl_sgbioedge_oversampled[k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
        residual_sgbioedge_oversampled[k]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9    

# %%

np.save(path/("residual_gbiodege_"+str(param['n_iter'])+"_iter.npy"), residual_gbioedge)
np.save(path/("residual_gbiodege_sr_"+str(param['n_iter'])+"_iter.npy"), residual_gbioedge_sr)
np.save(path/("residual_gbioedge_oversampled"+str(param['n_iter'])+"_iter.npy"), residual_gbioedge_oversampled)

#%% plots - gbioedge

plt.close('all')

plt.figure()
plt.plot(residual_gbioedge, 'b', label=str(param['n_subaperture'])+'x'+
         str(param['n_subaperture'])+', no SR '+str(param['n_modes_to_show'] )+' modes')
plt.plot(residual_gbioedge_sr, 'r', label=str(param['n_subaperture'])+'x'+
         str(param['n_subaperture'])+', SR ' +str(param['n_modes_to_show_sr'] )+' modes shown\n'
         +str(param['n_modes_to_control_sr'])+' modes controlled')
plt.plot(residual_gbioedge_oversampled, 'k', label=str(2*param['n_subaperture'])+'x'+
         str(2*param['n_subaperture'])+', '+str(param['n_modes_to_show_oversampled'] )+' modes')
plt.title('Closed Loop residuals gbioedge\n'
          'loop frequency : '+str(np.round(1/tel.samplingTime/1e3, 1))+'kHz\n'
          'Telescope diameter: '+str(tel.D) + ' m\n'
          'Half grey width : '+str(param['modulation'])+' lambda/D\n'
          'seed : '+str(seed))
plt.xlabel('Iteration')
plt.ylabel('residuals (nm)')
plt.legend()

plt.savefig(pathlib.Path(__file__).parent / "plots" / pathlib.Path("residual_gbiodege_sr_"+str(param['n_iter'])+"_iter.png"), 
            bbox_inches = 'tight')

# zoom
plt.figure()
plt.plot(residual_gbioedge, 'b', label=str(param['n_subaperture'])+'x'+
         str(param['n_subaperture'])+', no SR '+str(param['n_modes_to_show'] )+' modes')
plt.plot(residual_gbioedge_sr, 'r', label=str(param['n_subaperture'])+'x'+
         str(param['n_subaperture'])+', SR ' +str(param['n_modes_to_show_sr'] )+' modes shown\n'
         +str(param['n_modes_to_control_sr'])+' modes controlled')
plt.plot(residual_gbioedge_oversampled, 'k', label=str(2*param['n_subaperture'])+'x'+
         str(2*param['n_subaperture'])+', '+str(param['n_modes_to_show_oversampled'] )+' modes')
plt.title('zoom Closed Loop residuals gbioedge\n'
          'loop frequency : '+str(np.round(1/tel.samplingTime/1e3, 1))+'kHz\n'
          'Telescope diameter: '+str(tel.D) + ' m\n'
          'Half grey width : '+str(param['modulation'])+' lambda/D\n'
          'seed : '+str(seed))
plt.xlabel('Iteration')
plt.ylabel('residuals (nm)')
plt.xlim(4000, 6000)
plt.ylim(0, 400)
plt.legend()

plt.savefig(pathlib.Path(__file__).parent / "plots" / pathlib.Path("zoom_residual_gbiodege_sr_"+str(param['n_iter'])+"_iter.png"), 
            bbox_inches = 'tight')

#%% plots - sgbioedge

plt.close('all')

plt.figure()
plt.plot(residual_sgbioedge, 'b', label=str(param['n_subaperture'])+'x'+
         str(param['n_subaperture'])+', no SR '+str(param['n_modes_to_show'])+' modes')
plt.plot(residual_sgbioedge_sr, 'r', label=str(param['n_subaperture'])+'x'+
         str(param['n_subaperture'])+', SR ' +str(param['n_modes_to_show_sr'])+' modes shown\n'
         +str(param['n_modes_to_control_sr'])+' modes controlled')
plt.plot(residual_sgbioedge_oversampled, 'k', label=str(2*param['n_subaperture'])+'x'+
         str(2*param['n_subaperture'])+', '+str(param['n_modes_to_show_oversampled'])+' modes')
plt.title('Closed Loop residuals sgbioedge\n'
          'loop frequency : '+str(np.round(1/tel.samplingTime/1e3, 1))+'kHz\n'
          'Telescope diameter: '+str(tel.D) + ' m\n'
          'Half grey width : '+str(param['modulation'])+' lambda/D\n'
          'seed : '+str(seed))
plt.xlabel('Iteration')
plt.ylabel('residuals (nm)')
plt.legend()

plt.savefig(pathlib.Path(__file__).parent / "plots" / pathlib.Path("residual_sgbiodege_sr_"+str(param['n_iter'])+"_iter.png"), 
            bbox_inches = 'tight')

# zoom
plt.figure()
plt.plot(residual_sgbioedge, 'b', label=str(param['n_subaperture'])+'x'+
         str(param['n_subaperture'])+', no SR '+str(param['n_modes_to_show'])+' modes')
plt.plot(residual_sgbioedge_sr, 'r', label=str(param['n_subaperture'])+'x'+
         str(param['n_subaperture'])+', SR ' +str(param['n_modes_to_show_sr'])+' modes shown\n'
         +str(param['n_modes_to_control_sr'])+' modes controlled')
plt.plot(residual_sgbioedge_oversampled, 'k', label=str(2*param['n_subaperture'])+'x'+
         str(2*param['n_subaperture'])+', '+str(param['n_modes_to_show_oversampled'])+' modes')
plt.title('Zoom Closed Loop residuals sgbioedge\n'
          'loop frequency : '+str(np.round(1/tel.samplingTime/1e3, 1))+'kHz\n'
          'Telescope diameter: '+str(tel.D) + ' m\n'
          'Half grey width : '+str(param['modulation'])+' lambda/D\n'
          'seed : '+str(seed))
plt.xlabel('Iteration')
plt.ylabel('residuals (nm)')
plt.xlim(4000, 6000)
plt.ylim(0, 400)
plt.legend()

plt.savefig(pathlib.Path(__file__).parent / "plots" / pathlib.Path("zoom_residual_sgbiodege_sr_"+str(param['n_iter'])+"_iter.png"), 
            bbox_inches = 'tight')
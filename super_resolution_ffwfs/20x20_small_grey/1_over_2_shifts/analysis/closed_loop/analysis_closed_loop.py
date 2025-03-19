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

from fanch.tools.save_load import save_vars, load_vars

from OOPAO.tools.displayTools import cl_plot

#%% Get parameter file

path_parameter_file = pathlib.Path(__file__).parent.parent.parent / "parameter_file.pkl"
load_vars(path_parameter_file, ['param'])

#%% path type compatibility issues

if platform.system() == 'Windows':
    temp = deepcopy(pathlib.PosixPath)
    pathlib.PosixPath = pathlib.WindowsPath
elif platform.system() == 'Linux':
    temp = deepcopy(pathlib.WindowsPath)
    pathlib.WindowsPath = pathlib.PosixPath

#%% Load objects computed in build_object_file.py 

load_vars(param['path_object'] / pathlib.Path('object'+str(param['filename'])+'.pkl'), 
          ['parameters_object', 'origin_object',\
           'tel','atm', 'dm', 'ngs', 'M2C',\
           #'pyramid', 'pyramid_sr', 'pyramid_oversampled',\
           #'sbioedge', 'sbioedge_sr', 'sbioedge_oversampled',\
           'gbioedge', 'gbioedge_sr', 'gbioedge_oversampled',\
           'sgbioedge', 'sgbioedge_sr','sgbioedge_oversampled'])

#%% load calibrations

#load_vars(param['path_calibration'] / pathlib.Path('calibration_all_wfs'+param['filename']+'.pkl'))

# load_vars(param['path_calibration'] / pathlib.Path('calibration_pyramid'+param['filename']+'.pkl'))
#load_vars(param['path_calibration'] / pathlib.Path('calibration_sbioedge'+param['filename']+'.pkl'))
load_vars(param['path_calibration'] / pathlib.Path('calibration_gbioedge'+param['filename']+'.pkl'))
load_vars(param['path_calibration'] / pathlib.Path('calibration_sgbioedge'+param['filename']+'.pkl'))

#%% Modal Basis

M2C_sr = deepcopy(M2C)
M2C_sr[:, param['n_modes_to_control_sr']:] = 0.

#%% reconstructors - gbioedge

R_gbioedge = np.linalg.pinv(calib_gbioedge.D[:,:param['n_modes_to_show']])
R_gbioedge_sr = np.linalg.pinv(calib_gbioedge_sr.D[:,:param['n_modes_to_show_sr']])
R_gbioedge_oversampled = np.linalg.pinv(calib_gbioedge_oversampled.D[:,:param['n_modes_to_show_oversampled']])

reconstructor_gbioedge = M2C[:, :param['n_modes_to_show']]@R_gbioedge
reconstructor_gbioedge_sr = M2C_sr[:, :param['n_modes_to_show_sr']]@R_gbioedge_sr
reconstructor_gbioedge_oversampled = M2C[:, :param['n_modes_to_show_oversampled']]@R_gbioedge_oversampled

#%% reconstructors - sgbioedge

R_sgbioedge = np.linalg.pinv(calib_sgbioedge.D[:,:param['n_modes_to_show']])
R_sgbioedge_sr = np.linalg.pinv(calib_sgbioedge_sr.D[:,:param['n_modes_to_show_sr']])
R_sgbioedge_oversampled = np.linalg.pinv(calib_sgbioedge_oversampled.D[:,:param['n_modes_to_show_oversampled']])

reconstructor_sgbioedge = M2C[:, :param['n_modes_to_show']]@R_sgbioedge
reconstructor_sgbioedge_sr = M2C_sr[:, :param['n_modes_to_show_sr']]@R_sgbioedge_sr
reconstructor_sgbioedge_oversampled = M2C[:, :param['n_modes_to_show_oversampled']]@R_sgbioedge_oversampled

#%% Allocate memory

total_gbioedge = np.zeros((len(param['seeds']), param['n_iter']))
residual_gbioedge = np.zeros((len(param['seeds']), param['n_iter']))
strehl_gbioedge = np.zeros((len(param['seeds']), param['n_iter']))

total_gbioedge_sr = np.zeros((len(param['seeds']), param['n_iter']))
residual_gbioedge_sr = np.zeros((len(param['seeds']), param['n_iter']))
strehl_gbioedge_sr = np.zeros((len(param['seeds']), param['n_iter']))         
                              
total_gbioedge_oversampled = np.zeros((len(param['seeds']), param['n_iter']))
residual_gbioedge_oversampled = np.zeros((len(param['seeds']), param['n_iter']))
strehl_gbioedge_oversampled = np.zeros((len(param['seeds']), param['n_iter']))
                                       
total_sgbioedge = np.zeros((len(param['seeds']), param['n_iter']))
residual_sgbioedge = np.zeros((len(param['seeds']), param['n_iter']))
strehl_sgbioedge = np.zeros((len(param['seeds']), param['n_iter']))

total_sgbioedge_sr = np.zeros((len(param['seeds']), param['n_iter']))
residual_sgbioedge_sr = np.zeros((len(param['seeds']), param['n_iter']))
strehl_sgbioedge_sr = np.zeros((len(param['seeds']), param['n_iter']))
                               
total_sgbioedge_oversampled = np.zeros((len(param['seeds']), param['n_iter']))
residual_sgbioedge_oversampled = np.zeros((len(param['seeds']), param['n_iter']))
strehl_sgbioedge_oversampled = np.zeros((len(param['seeds']), param['n_iter']))

#%%

display = False

for m in range(len(param['seeds'])):
    seed = param['seeds'][m]

    #% Close the loop - gbioedge
    
    # Setup
    
    tel.computePSF()
    
    atm.initializeAtmosphere(tel)
    atm.generateNewPhaseScreen(seed = seed)
    tel+atm
    
    dm.coefs = 0
    gbioedge_measure = 0*gbioedge.signal
    
    ngs*tel*dm*gbioedge
    
    n = 200
    
    SE_PSF = []
    LE_PSF = np.log10(tel.PSF)[n:-n,n:-n]
    
    plot_obj = cl_plot(list_fig          = [atm.OPD,tel.mean_removed_OPD, gbioedge.cam.frame,[[0,0],[0,0]],
                                            [dm.coordinates[:,0], np.flip(dm.coordinates[:,1]), dm.coefs],
                                            np.log10(tel.PSF),np.log10(tel.PSF)],\
                            type_fig          = ['imshow','imshow','imshow','plot','scatter','imshow','imshow'],\
                            list_title        = ['Turbulence OPD','Residual OPD','bio Detector',None,None,None,None],\
                            list_lim          = [None,None,None,None,None,[2,6],[2,6]],\
                            list_label        = [None,None,None,['Time','WFE [nm]'],['DM Commands',''],['Short Exposure PSF',''],
                                                  ['Long Exposure_PSF','']],\
                            n_subplot         = [4,2],\
                            list_display_axis = [None,None,None,True,None,None,None],\
                            list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)
        
    
    
    
    # close the loop
    
    for k in range(param['n_iter']):
        
        atm.update()
        total_gbioedge[m, k] = np.std(tel.OPD[np.where(tel.pupil==1)])*1e9
        phase_turb = tel.src.phase
        
        tel*dm*gbioedge
        
        gbioedge_measure = gbioedge.signal # tune delay (1 frames here)
        
        dm.coefs = dm.coefs - param['loop_gain'] * np.matmul(reconstructor_gbioedge, gbioedge_measure)
        
        # gbioedge_measure = gbioedge.signal # tune delay (2 frames here)
        
        strehl_gbioedge[m, k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
        
        
        if k>15 and display:
            tel.computePSF(4)
            
            SE_PSF.append(np.log10(tel.PSF)[n:-n,n:-n])
            LE_PSF = np.mean(SE_PSF, axis=0)
            cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD, gbioedge.cam.frame,[np.arange(k+1),residual_gbioedge[:k+1]],dm.coefs,SE_PSF[-1], LE_PSF],
                    list_lim =[None,None,None,None,None,[SE_PSF[-1].max()-4,SE_PSF[-1].max()],[LE_PSF.max()-4,LE_PSF.max()]],plt_obj = plot_obj)
            plt.pause(0.1)
            if plot_obj.keep_going is False:
                    break
                
        strehl_gbioedge[m, k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
        residual_gbioedge[m, k]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
        
        
    #% Close the loop - gbioedge_sr
    
    # Setup
    
    tel.computePSF()
    
    
    
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
    
    # close the loop
    
    for k in range(param['n_iter']):
        
        atm.update()
        phase_turb = tel.src.phase
        
        tel*dm*gbioedge_sr
        
        gbioedge_sr_measure = gbioedge_sr.signal # tune delay (1 frames here)
        
        dm.coefs = dm.coefs - param['loop_gain'] * np.matmul(reconstructor_gbioedge_sr, gbioedge_sr_measure)
        
        # gbioedge_sr_measure = gbioedge_sr.signal # tune delay (2 frames here)
        
        strehl_gbioedge_sr[m, k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
        
        
        if k>15 and display:
            tel.computePSF(4)
            
            SE_PSF.append(np.log10(tel.PSF)[n:-n,n:-n])
            LE_PSF = np.mean(SE_PSF, axis=0)
            cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD, gbioedge_sr.cam.frame,[np.arange(k+1),residual_gbioedge_sr[:k+1]],dm.coefs,SE_PSF[-1], LE_PSF],
                    list_lim =[None,None,None,None,None,[SE_PSF[-1].max()-4,SE_PSF[-1].max()],[LE_PSF.max()-4,LE_PSF.max()]],plt_obj = plot_obj)
            plt.pause(0.1)
            if plot_obj.keep_going is False:
                    break
                
        strehl_gbioedge_sr[m, k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
        residual_gbioedge_sr[m, k]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
        
    #% Close the loop - gbioedge_oversampled
    
    # Setup
    
    tel.computePSF()
    
    
    
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
    
    # close the loop
    
    for k in range(param['n_iter']):
        
        atm.update()
        total_gbioedge_oversampled[m, k] = np.std(tel.OPD[np.where(tel.pupil==1)])
        phase_turb = tel.src.phase
        
        tel*dm*gbioedge_oversampled
        
        gbioedge_oversampled_measure = gbioedge_oversampled.signal # tune delay (1 frames here)
        
        dm.coefs = dm.coefs - param['loop_gain'] * np.matmul(reconstructor_gbioedge_oversampled, gbioedge_oversampled_measure)
        
        # gbioedge_oversampled_measure = gbioedge_oversampled.signal # tune delay (2 frames here)
        
        strehl_gbioedge_oversampled[m, k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
        
        
        if k>15 and display:
            tel.computePSF(4)
            
            SE_PSF.append(np.log10(tel.PSF)[n:-n,n:-n])
            LE_PSF = np.mean(SE_PSF, axis=0)
            cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD, gbioedge_oversampled.cam.frame,[np.arange(k+1),residual_gbioedge_oversampled[:k+1]],dm.coefs,SE_PSF[-1], LE_PSF],
                    list_lim =[None,None,None,None,None,[SE_PSF[-1].max()-4,SE_PSF[-1].max()],[LE_PSF.max()-4,LE_PSF.max()]],plt_obj = plot_obj)
            plt.pause(0.1)
            if plot_obj.keep_going is False:
                    break
                
        strehl_gbioedge_oversampled[m, k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
        residual_gbioedge_oversampled[m, k]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9    
    
    #% Close the loop - sgbioedge
    
    # Setup
    
    tel.computePSF()
    
    
    
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
    
    for k in range(param['n_iter']):
        
        atm.update()
        total_sgbioedge[m, k] = np.std(tel.OPD[np.where(tel.pupil==1)])
        phase_turb = tel.src.phase
        
        tel*dm*sgbioedge
        
        sgbioedge_measure = sgbioedge.signal # tune delay (1 frames here)
        
        dm.coefs = dm.coefs - param['loop_gain'] * np.matmul(reconstructor_sgbioedge, sgbioedge_measure)
        
        # sgbioedge_measure = sgbioedge.signal # tune delay (2 frames here)
        
        strehl_sgbioedge[m, k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
        
        
        if k>15 and display:
            tel.computePSF(4)
            
            SE_PSF.append(np.log10(tel.PSF)[n:-n,n:-n])
            LE_PSF = np.mean(SE_PSF, axis=0)
            cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD, sgbioedge.cam.frame,[np.arange(k+1),residual_sgbioedge[:k+1]],dm.coefs,SE_PSF[-1], LE_PSF],
                    list_lim =[None,None,None,None,None,[SE_PSF[-1].max()-4,SE_PSF[-1].max()],[LE_PSF.max()-4,LE_PSF.max()]],plt_obj = plot_obj)
            plt.pause(0.1)
            if plot_obj.keep_going is False:
                    break
                
        strehl_sgbioedge[m, k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
        residual_sgbioedge[m, k]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
        
        
    #% Close the loop - sgbioedge_sr
    
    # Setup
    
    tel.computePSF()
    
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
    
    for k in range(param['n_iter']):
        
        atm.update()
        total_sgbioedge_sr[m, k] = np.std(tel.OPD[np.where(tel.pupil==1)])
        phase_turb = tel.src.phase
        
        tel*dm*sgbioedge_sr
        
        sgbioedge_sr_measure = sgbioedge_sr.signal # tune delay (1 frames here)
        
        dm.coefs = dm.coefs - param['loop_gain'] * np.matmul(reconstructor_sgbioedge_sr, sgbioedge_sr_measure)
        
        # sgbioedge_sr_measure = sgbioedge_sr.signal # tune delay (2 frames here)
        
        strehl_sgbioedge_sr[m, k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
        
        
        if k>15 and display:
            tel.computePSF(4)
            
            SE_PSF.append(np.log10(tel.PSF)[n:-n,n:-n])
            LE_PSF = np.mean(SE_PSF, axis=0)
            cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD, sgbioedge_sr.cam.frame,[np.arange(k+1),residual_sgbioedge_sr[:k+1]],dm.coefs,SE_PSF[-1], LE_PSF],
                    list_lim =[None,None,None,None,None,[SE_PSF[-1].max()-4,SE_PSF[-1].max()],[LE_PSF.max()-4,LE_PSF.max()]],plt_obj = plot_obj)
            plt.pause(0.1)
            if plot_obj.keep_going is False:
                    break
                
        strehl_sgbioedge_sr[m, k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
        residual_sgbioedge_sr[m, k]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
        
    #% Close the loop - sgbioedge_oversampled
    
    # Setup
    
    tel.computePSF()
    
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
        
    for k in range(param['n_iter']):
        
        atm.update()
        total_sgbioedge_oversampled[m, k] = np.std(tel.OPD[np.where(tel.pupil==1)])
        phase_turb = tel.src.phase
        
        tel*dm*sgbioedge_oversampled
        
        sgbioedge_oversampled_measure = sgbioedge_oversampled.signal # tune delay (1 frames here)
        
        dm.coefs = dm.coefs - param['loop_gain'] * np.matmul(reconstructor_sgbioedge_oversampled, sgbioedge_oversampled_measure)
        
        # sgbioedge_oversampled_measure = sgbioedge_oversampled.signal # tune delay (2 frames here)
        
        strehl_sgbioedge_oversampled[m, k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
        
        
        if k>15 and display:
            tel.computePSF(4)
            
            SE_PSF.append(np.log10(tel.PSF)[n:-n,n:-n])
            LE_PSF = np.mean(SE_PSF, axis=0)
            cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD, sgbioedge_oversampled.cam.frame,[np.arange(k+1),residual_sgbioedge_oversampled[:k+1]],
                                  dm.coefs,SE_PSF[-1], LE_PSF],
                    list_lim =[None,None,None,None,None,[SE_PSF[-1].max()-4,SE_PSF[-1].max()],[LE_PSF.max()-4,LE_PSF.max()]],plt_obj = plot_obj)
            plt.pause(0.1)
            if plot_obj.keep_going is False:
                    break
                
        strehl_sgbioedge_oversampled[m, k]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
        residual_sgbioedge_oversampled[m, k]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9    

#%% Save analysis results

path_analysis_data = pathlib.Path(__file__).parent / 'data_analysis'

parameters_analysis = deepcopy(param)

save_vars(path_analysis_data / pathlib.Path('analysis_closed_loop' + param['filename']), 
          ['parameters_analysis',\
           'total_gbioedge', 'residual_gbioedge', 'strehl_gbioedge',\
           'residual_gbioedge_sr', 'strehl_gbioedge_sr',\
           'residual_gbioedge_oversampled', 'strehl_gbioedge_oversampled',\
           'residual_sgbioedge', 'strehl_sgbioedge',\
           'residual_sgbioedge_sr', 'strehl_sgbioedge_sr',\
           'residual_sgbioedge_oversampled', 'strehl_sgbioedge_oversampled'])


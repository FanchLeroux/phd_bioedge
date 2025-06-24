# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 14:05:03 2025

@author: lgs
"""

from plico_dm import deformableMirror
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.mask import CircularMask
import numpy as np
from slm_4lgs_prototype.utils.my_tools import reshape_map2vector

def create_device():
    '''
    Return the object that allows to control and interface the SLM
    Before running this function you should run the script plico_dm_controller_#.exe to interface the device.
    It is located in the following path:
    D:\anaconda3\envs\slm\Scripts
    '''
    # hostname and port defined in the configuration file (server1)
    # C:\Users\lgs\AppData\Local\INAF Arcetri Adaptive Optics\inaf.arcetri.ao.plico_dm_server\plico_dm_server.conf
    hostname = 'localhost'
    port = 7000
    slm = deformableMirror(hostname, port)
    
    return slm

def apply_defocus_on_slm(amp=500e-9): # amp : m rms
    
    '''
    This function display a defocus (Z4) of amplitude 500nm rms on a circular mask
    of center (517,875) pixel and radius 571pixel, in the frame of the SLM
    '''
    # building circular mask
    frame_shape = (1152, 1920)
    mask_radius = 550
    centerYX = (540, 920)
    cmask_obj = CircularMask(frame_shape, mask_radius, centerYX)
    
    #building Zernike polynomial
    zg = ZernikeGenerator(cmask_obj)
    
    j = 4 # Defocus index Noll
    wavefront2display = amp * zg.getZernike(index = j)
    
    #create slm device
    slm = create_device()
    #reshape 2D wavefront in 1D vector
    cmd = np.reshape(wavefront2display, (1152*1920,), 'C')
    #cmd = reshape_map2vector(wavefront2display, 1152*1920, 'C')
    #apply command on slm
    #need a 1D array
    slm.set_shape(command = cmd)
     
    
    
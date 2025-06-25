# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 17:02:10 2025

@author: lgs
"""

import datetime
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt

#%%

dirc = pathlib.Path(__file__).parent.parent.parent.parent.parent / "data" / "orca"

#%% ORCA 

from pylablib.devices import DCAM
from astropy.io import fits

#%% MEADOWLARK

from plico_dm import deformableMirror
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.mask import CircularMask
import numpy as np
from slm_4lgs_prototype.utils.my_tools import reshape_map2vector

#%% Functions declarations - ORCA

def acquire(cam, n_frames, exp_time, roi=False, dirc = False, overwrite=False):
    
    # roi = [xcenter, ycenter, xwidth, ywidth]
    
    cam.set_exposure(exp_time)
    t = str(datetime.datetime.now())
    image = np.double(cam.grab(n_frames))
    
    if roi != False:
        image = image[:, roi[1]-roi[3]//2:roi[1]+roi[3]//2, roi[0]-roi[2]//2:roi[0]+roi[2]//2]
    
    if dirc != False:
        
        hdu = fits.PrimaryHDU(data = image)
        hdr=hdu.header
        tmp = cam.get_acquisition_parameters()
        hdr['NFRAMES']      = (tmp['nframes'],'Size of the data cube')
                
        tmp = cam.get_all_attribute_values()
        hdr['EXP_TIME']     = (tmp['exposure_time'],'Exposure time in s')
        hdr['FPS']          = (tmp['internal_frame_rate'],'Frame rate in Hz')
        hdr['INTERVAL']     = (tmp['internal_frame_interval'],'Delay between two successive acquisitions in s')
        hdr['HPOS']         = (tmp['subarray_hpos'],'X-position of the ROI')
        hdr['YPOS']         = (tmp['subarray_vpos'],'Y-position of the ROI')
        hdr['TIME']         = (t,'Local Time of Acquisition')
    
        file_name = dirc / pathlib.Path(str(cam.ID)+"_exp_" + 
                                        str(np.round(cam.get_exposure()*1000, 3)) + '_nframes_' + 
                                        str(n_frames) + '.fits')
        
        hdu.writeto(file_name, overwrite=overwrite)
    
    return image

#%% Functions declarations - MEADOWLARK

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
    mask_radius = 500
    centerYX = (551, 910)
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

#%% Link camera

cam = DCAM.DCAMCamera()

#%%

# initialize settings

cam.exp_time = 4e-3    # exposure time (s)
cam.n_frames = 10      # acquire cubes of n_frames images
cam.ID = 0             # ID for the data saved

roi = False
#roi = [1058, 1037, 100, 100]

#%% Aquire image

data = acquire(cam, cam.n_frames, cam.exp_time, roi=roi, dirc = dirc, overwrite=True)

if data.max() > 65000:
    print("Image is saturated")

#%% Post processing

img = np.median(data, axis=0)

#%% Pots

fig, axs = plt.subplots(nrows=1,ncols=1)
im = axs.imshow(img)

axs.set_title("ORCA frame")
plt.colorbar(im, ax=axs, fraction=0.046, pad=0.04)

plt.figure()

plt.plot(img[50,:])
#plt.yscale('log')

#%% SLM MEADOWLARK

apply_defocus_on_slm(amp=10*500e-9)

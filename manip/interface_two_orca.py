# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 10:10:15 2025

@author: lgs
"""

import datetime
import pathlib

import numpy as np

import matplotlib.pyplot as plt

from pylablib.devices import DCAM

from astropy.io import fits

#%%

def get_utc_now():
    return datetime.datetime.utcnow().strftime("utc_%Y-%m-%d_%H-%M-%S")

# Get Frames with ORCA
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

# display ORCA frames in real time
def live_view(get_frame_func, cam, roi, dirc = False, overwrite=True, interval=0.005):
    
    cam.n_frames = 1
    
    plt.ion()  # Turn on interactive mode

    # First frame for setup
    frame = get_frame_func(cam, cam.n_frames, cam.exp_time, roi=roi, dirc = False, overwrite=True)[0,:,:,]
    is_color = frame.ndim == 3 and frame.shape[2] == 3

    fig, ax = plt.subplots()
    im = ax.imshow(frame, cmap='viridis' if not is_color else None)
    plt.colorbar(im, ax=ax)
    title = fig.suptitle(f"Max value: {np.max(frame):.2f}", fontsize=24)

    while plt.fignum_exists(fig.number):  # Loop while window is open
        frame = get_frame_func(cam, cam.n_frames, cam.exp_time, roi=roi, dirc = False, overwrite=True)[0,:,:,]
        im.set_data(frame)
        im.set_clim(vmin=np.min(frame), vmax=np.max(frame))  # Adjust color scale
        title.set_text(f"Max value: {np.max(frame):.2f}")  # Update title
        plt.pause(interval)

#%%

dirc_data = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent / "data"

#%%

DCAM.get_cameras_number()

#%%

cam1 = DCAM.DCAMCamera(idx=0)
cam2 = DCAM.DCAMCamera(idx=1)

#%%

cam1.get_device_info()

#%%

cam2.get_device_info()

#%% Link camera ORCA

cam = DCAM.DCAMCamera()

#%% Setup camera 1

# initialize settings
cam1.exp_time = 5e-3    # exposure time (s)
cam1.n_frames = 10      # acquire cubes of n_frames images
cam1.ID = 0             # ID for the data saved
roi = False

#%% Setup camera 2

# initialize settings
cam2.exp_time = 5e-3    # exposure time (s)
cam2.n_frames = 10      # acquire cubes of n_frames images
cam2.ID = 0             # ID for the data saved
roi = False

#%% Enter roi

roi = [900, 830, 500, 500] # roi[0] is x coordinate, i.e column number

#%% Check roi

live_view(acquire, cam, roi)

#%% Acquire image cam1

data_orca_cam1 = acquire(cam1, n_frames=10, exp_time=5e-3, roi=roi,
                    dirc = False, overwrite=True)

plt.figure()
plt.imshow(np.median(data_orca_cam1, axis=0))

#%% Acquire image cam2

data_orca_cam2 = acquire(cam2, n_frames=10, exp_time=5e-3, roi=roi,
                    dirc = False, overwrite=True)

plt.figure()
plt.imshow(np.median(data_orca_cam2, axis=0))

#%% Live view

live_view(acquire, cam1, roi)

#%%

cam1.close()
cam2.close()


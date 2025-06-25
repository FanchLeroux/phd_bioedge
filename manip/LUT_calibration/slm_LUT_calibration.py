# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 17:02:10 2025

@author: lgs
"""

import datetime
import pathlib

import numpy as np
import matplotlib.pyplot as plt

import ctypes as ct

from PIL import Image

#%% ORCA 

from pylablib.devices import DCAM
from astropy.io import fits

#%% MEADOWLARK

# load slm library
ct.cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\Blink_C_wrapper")
slm_lib = ct.CDLL("Blink_C_wrapper")

# load image generation library
ct.cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\ImageGen")
image_lib = ct.CDLL("ImageGen")

#%% Basic parameters for calling Create_SDK

bit_depth = ct.c_uint(12)
num_boards_found = ct.c_uint(0)
constructed_okay = ct.c_uint(-1)
is_nematic_type = ct.c_bool(1)
RAM_write_enable = ct.c_bool(1)
use_GPU = ct.c_bool(1)
max_transients = ct.c_uint(20)
board_number = ct.c_uint(1)
wait_For_Trigger = ct.c_uint(0)
timeout_ms = ct.c_uint(5000)
OutputPulseImageFlip = ct.c_uint(0)
OutputPulseImageRefresh = ct.c_uint(0); #only supported on 1920x1152, FW rev 1.8.

#%% Call the Create_SDK constructor

# Returns a handle that's passed to subsequent SDK calls
slm_lib.Create_SDK(bit_depth, ct.byref(num_boards_found), ct.byref(constructed_okay), is_nematic_type, RAM_write_enable, use_GPU, max_transients, 0)

if constructed_okay.value == 0:
    print ("Blink SDK did not construct successfully");
    # Python ctypes assumes the return value is always int
    # We need to tell it the return type by setting restype
    slm_lib.Get_last_error_message.restype = ct.c_char_p
    print (slm_lib.Get_last_error_message());

if num_boards_found.value == 1:
    print ("Blink SDK was successfully constructed");
    print ("Found %s SLM controller(s)" % num_boards_found.value);
    height = ct.c_uint(slm_lib.Get_image_height(board_number));
    width = ct.c_uint(slm_lib.Get_image_width(board_number));
    center_x = ct.c_uint(width.value//2);
    center_y = ct.c_uint(height.value//2);

#%% Load LUT

slm_lib.Load_LUT_file(board_number,
                      b"C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\12bit_linear.LUT");

#%%

slm_lib.Load_LUT_file(board_number,
                      b"C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm5758_at675.LUT");

#%% Get flat

slm_flat = np.asarray(Image.open(
    b"C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\Flat Files\\slm5758_at675.bmp"))

slm_flat = np.reshape(slm_flat, [width.value*height.value], 'C');

#%% Load flat on SLM

slm_lib.Write_image(board_number, slm_flat.ctypes.data_as(ct.POINTER(ct.c_ubyte)), height.value*width.value, 
                    wait_For_Trigger, OutputPulseImageFlip, OutputPulseImageRefresh,timeout_ms)
slm_lib.ImageWriteComplete(board_number, timeout_ms);

#%% close sdk

# Always call Delete_SDK before exiting
slm_lib.Delete_SDK()

#%%

dirc = pathlib.Path(__file__).parent

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
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

from time import sleep
from PIL import Image


from pylablib.devices import DCAM
from astropy.io import fits

from fanch.plots import make_gif

#%%

dirc_data = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent / "data" / "slm" / "LUT_calibration"

#%% Function declaration

# utc datetime now

def utc_now():
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

#%% Link camera ORCA

cam = DCAM.DCAMCamera()

#%% Link slm MEADOWLARK

# load slm library
ct.cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\Blink_C_wrapper")
slm_lib = ct.CDLL("Blink_C_wrapper")

# load image generation library
ct.cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\ImageGen")
image_lib = ct.CDLL("ImageGen")

# Basic parameters for calling Create_SDK

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

# Call the Create_SDK constructor

# Returns a handle that's passed to subsequent SDK calls
slm_lib.Create_SDK(bit_depth, ct.byref(num_boards_found), ct.byref(constructed_okay), 
                   is_nematic_type, RAM_write_enable, use_GPU, max_transients, 0)

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

# Load LUT

slm_lib.Load_LUT_file(board_number,
                      b"C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\12bit_linear.LUT");

# slm_lib.Load_LUT_file(board_number,
#                       b"C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm5758_at675.LUT");

#%% Get slm flat @ 675 nm

slm_flat = np.asarray(Image.open(
    b"C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\Flat Files\\slm5758_at675.bmp"))

slm_flat = np.reshape(slm_flat, [width.value*height.value], 'C');

#%% Load flat on SLM

slm_lib.Write_image(board_number, slm_flat.ctypes.data_as(ct.POINTER(ct.c_ubyte)), height.value*width.value, 
                    wait_For_Trigger, OutputPulseImageFlip, OutputPulseImageRefresh,timeout_ms)
slm_lib.ImageWriteComplete(board_number, timeout_ms)

#%% Load grey level 128 stripe diffraction pattern on slm

grey = 128

num_data_points = 256
pixels_per_stripe = 8

# allocate memory for the patterns to display on slm
pattern = np.empty([width.value*height.value], np.uint8, 'C'); 

# generate pattern
image_lib.Generate_Stripe(pattern.ctypes.data_as(ct.POINTER(ct.c_ubyte)),
                          width.value, height.value, 0, grey, pixels_per_stripe)

# add slm_flat
pattern = np.mod(pattern.astype('float64') + slm_flat.astype('float64'), 256).astype('uint8')

# display pattern on slm
slm_lib.Write_image(board_number, pattern.ctypes.data_as(ct.POINTER(ct.c_ubyte)), height.value*width.value, 
                    wait_For_Trigger, OutputPulseImageFlip, OutputPulseImageRefresh,timeout_ms)
slm_lib.ImageWriteComplete(board_number, timeout_ms)

#%% Setup camera

# initialize settings
cam.exp_time = 10e-3    # exposure time (s)
cam.n_frames = 10      # acquire cubes of n_frames images
cam.ID = 0             # ID for the data saved
roi = False

#%% Aquire image

data = acquire(cam, cam.n_frames, cam.exp_time, roi=roi, dirc = False, overwrite=True)
if data.max() > 65000:
    print("Image is saturated")

#%% Post processing

img = np.median(data, axis=0)

#%% Show post-processed image

fig, axs = plt.subplots(nrows=1,ncols=1)
im = axs.imshow(img)
axs.set_title("ORCA full frame")
plt.colorbar(im, ax=axs, fraction=0.046, pad=0.04)

print(data.max())

#%% set ROI around +1 or -1 diffraction order

roi = [853, 915, 20, 20] # roi[0] is x coordinate, i.e column number

#%% Check ROI

data = acquire(cam, cam.n_frames, cam.exp_time, roi=roi, dirc = False, overwrite=True)
if data.max() > 65000:
    print("Image is saturated")
img = np.median(data, axis=0)
fig, axs = plt.subplots(nrows=1,ncols=1)
im = axs.imshow(img)
axs.set_title("ORCA ROI")
plt.colorbar(im, ax=axs, fraction=0.046, pad=0.04)

#%% Take LUT calibration measurements

# allocate memory for the measurments
images = np.zeros([roi[2], roi[3], num_data_points])
measurements = np.zeros((num_data_points, 2)) # 1st column : grey level ; # 2nd coloumn : intensity measurement

for grey in range(num_data_points):
    
    # generate pattern
    image_lib.Generate_Stripe(pattern.ctypes.data_as(ct.POINTER(ct.c_ubyte)),
                              width.value, height.value, 0, grey, pixels_per_stripe)
    
    # add slm_flat
    pattern = np.mod(pattern.astype('float64') + slm_flat.astype('float64'), 255).astype('uint8')
    
    # display pattern on slm
    slm_lib.Write_image(board_number, pattern.ctypes.data_as(ct.POINTER(ct.c_ubyte)), height.value*width.value, 
                        wait_For_Trigger, OutputPulseImageFlip, OutputPulseImageRefresh,timeout_ms)
    slm_lib.ImageWriteComplete(board_number, timeout_ms)
    
    # take image
    
    data = acquire(cam, cam.n_frames, cam.exp_time, roi=roi, dirc = False, overwrite=True)
    if data.max() > 65000:
        print("Image is saturated")
    img = np.median(data, axis=0)
    
    measurements[grey, 0] = grey
    measurements[grey, 1] = img.max()
    
    images[:,:,grey] = img

utc_now = utc_now()

np.save(dirc_data / (utc_now + "_LUT_images.npy"), images)
np.save(dirc_data / (utc_now + "_LUT_measurements.npy"), measurements)

# To visualize the stripe patterns :
# plt.imshow(np.reshape(pattern, (height.value,width.value)))

#%% save measurements under .csv file

csv_file = pathlib.Path(dirc_data / (utc_now + "_csv") /"raw0.csv")
csv_file.parent.mkdir(exist_ok=True, parents=True)

#%%

with open(csv_file, "w") as f:
    for k in range(measurements.shape[0]):
        f.write(str(int(measurements[k,0]))+", "+str(float(measurements[k,1]))+"\n")

#%% save a .gif of the raw images

make_gif(dirc_data / (utc_now + "_LUT_images.gif"), images, interval=50)

#%% Plot raw measurements

plt.figure()
plt.plot(measurements[:,1])
plt.title("Raw intensity meadurements")
plt.xlabel("Grey Level")
plt.ylabel("Max Intensity on ORCA")

#%% delete slm sdk

# Always call Delete_SDK before exiting
slm_lib.Delete_SDK()

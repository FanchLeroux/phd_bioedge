# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 14:30:56 2025

@author: lgs
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt

import ctypes as ct

from PIL import Image

from fanch.tools.miscellaneous import get_tilt, get_circular_pupil

#%%

dirc_data = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent / "data"

#%%

def display_phase_on_slm(phase, slm_flat=np.False_, slm_shape=[1152,1920], return_command_vector=False):
    
    if slm_flat.dtype == np.dtype("bool"):
        slm_flat = np.zeros([slm_shape[0]*slm_shape[1]])
    
    else:
        slm_flat = np.reshape(slm_flat, [slm_shape[0]*slm_shape[1]])
    
    phase = np.reshape(phase, [slm_shape[0]*slm_shape[1]])
    phase = np.mod(phase+slm_flat, 256)
    phase = phase.astype(dtype=np.uint8)

    # display pattern on slm
    slm_lib.Write_image(board_number, phase.ctypes.data_as(ct.POINTER(ct.c_ubyte)), slm_shape[0]*slm_shape[1], 
                        wait_For_Trigger, OutputPulseImageFlip, OutputPulseImageRefresh,timeout_ms)
    slm_lib.ImageWriteComplete(board_number, timeout_ms)
    
    if return_command_vector:
        return phase
    else:
        return None

#%% parameters

# slm shape
slm_shape = np.array([1152,1920])

# number of phase measurements point
n_subaperture = 20

# pupil radius in SLM pixels
pupil_radius = 500 # [pixel]

# pupil center on slm
pupil_center = [576,960] # [pixel]

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
OutputPulseImageRefresh = ct.c_uint(0) #only supported on 1920x1152, FW rev 1.8.

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
    print ("Found %s SLM controller(s)" % num_boards_found.value)
    height = ct.c_uint(slm_lib.Get_image_height(board_number))
    width = ct.c_uint(slm_lib.Get_image_width(board_number))
    center_x = ct.c_uint(width.value//2)
    center_y = ct.c_uint(height.value//2)

# By default load a linear LUT and a black WFC
slm_lib.Load_LUT_file(board_number, str(dirc_data / "slm" / "LUT" / "12bit_linear.lut").encode('utf-8'))
slm_flat = np.asarray(Image.open(str(dirc_data / "slm" / "WFC" / "1920black.bmp")))
slm_flat = np.reshape(slm_flat, [width.value*height.value], 'C')
slm_lib.Write_image(board_number, slm_flat.ctypes.data_as(ct.POINTER(ct.c_ubyte)), height.value*width.value, 
                    wait_For_Trigger, OutputPulseImageFlip, OutputPulseImageRefresh,timeout_ms)
slm_lib.ImageWriteComplete(board_number, timeout_ms)

#%% Load LUT

# slm_lib.Load_LUT_file(board_number, str(dirc_data / "LUT" / "12bit_linear.lut").encode('utf-8'))
# slm_lib.Load_LUT_file(board_number, str(dirc_data / "LUT" / "slm5758_at675.lut").encode('utf-8'))
slm_lib.Load_LUT_file(board_number, str(dirc_data / "slm" / "LUT" / "utc_2025-06-27_11-37-41_slm0_at675.lut").encode('utf-8'))

#%% Load WFC

slm_flat = np.asarray(Image.open(str(dirc_data / "slm" / "WFC" / "slm5758_at675.bmp")), dtype=np.float64)
slm_flat = slm_flat / slm_flat.max() * 255.0
slm_flat = slm_flat.astype(dtype=np.uint8)
display_phase_on_slm(slm_flat)

#%% Update WFC with a small tilt at 45° to get out of 0th order

tilt_amplitude = 10.0*np.pi # [rad]
tilt_angle = 45.0 # [deg]
tilt = get_tilt([1152,1920], theta=np.deg2rad(tilt_angle), amplitude = tilt_amplitude)/(2*np.pi) * 255.0

slm_flat = np.mod(tilt+slm_flat, 256)
slm_flat = slm_flat.astype(dtype=np.uint8)

np.save(dirc_data / "slm" / "WFC" /
           ("slm5758_at675_tilt_amplitude"+
            str(int(tilt_amplitude/np.pi))+
            "pi_tilt_angle_"+str(int(tilt_angle))+
            "degree.npy"), slm_flat)

#%% Load new WFC

display_phase_on_slm(slm_flat)

#%% Find pupil footprit on SLM

tilt_amplitude = -10.0*np.pi # [rad]
tilt_angle = -45.0 # [deg]
tilt = get_tilt([1152,1920], theta=np.deg2rad(tilt_angle), amplitude = tilt_amplitude)/(2*np.pi) * 255.0

# apply tilt on full slm
# command = display_phase_on_slm(tilt, slm_flat, slm_shape=[1152,1920], return_command_vector=True)
# plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

# generate SLM phase screen
pupil = get_circular_pupil(2*pupil_radius)
tilt_in_pupil = np.zeros([1152,1920])
tilt_in_pupil[pupil_center[0]-pupil_radius:pupil_center[0]+pupil_radius,
              pupil_center[1]-pupil_radius:pupil_center[1]+pupil_radius] = pupil
tilt_in_pupil = tilt_in_pupil * tilt

command = display_phase_on_slm(tilt_in_pupil, slm_flat, slm_shape=[1152,1920], return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

#%% Load a linear LUT and a black WFC

slm_lib.Load_LUT_file(board_number, str(dirc_data / "slm" / "LUT" / "12bit_linear.lut").encode('utf-8'))
display_phase_on_slm(np.zeros(slm_shape))

#%% delete slm sdk

# Always call Delete_SDK before exiting
slm_lib.Delete_SDK()

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 16:33:43 2025

@author: fleroux
"""

import datetime
import pathlib
import math
import tqdm

import numpy as np
import matplotlib.pyplot as plt

import ctypes as ct

from pylablib.devices import DCAM

from astropy.io import fits

from fanch.tools.miscellaneous import get_tilt, get_circular_pupil
from fanch.plots import make_gif

# %% Function declaration

# utc datetime now


def get_utc_now():
    return datetime.datetime.utcnow().strftime("utc_%Y-%m-%d_%H-%M-%S")

# Get Frames with ORCA


def acquire(cam, n_frames, exp_time, roi=False, dirc=False, overwrite=False):

    # roi = [xcenter, ycenter, xwidth, ywidth]

    cam.set_exposure(exp_time)
    t = str(datetime.datetime.now())
    image = np.double(cam.grab(n_frames))

    if roi != False:
        image = image[:, roi[1]-roi[3]//2:roi[1]+roi[3]//2,
                      roi[0]-roi[2]//2:roi[0]+roi[2]//2]

    if dirc != False:

        hdu = fits.PrimaryHDU(data=image)
        hdr = hdu.header
        tmp = cam.get_acquisition_parameters()
        hdr['NFRAMES'] = (tmp['nframes'], 'Size of the data cube')

        tmp = cam.get_all_attribute_values()
        hdr['EXP_TIME'] = (tmp['exposure_time'], 'Exposure time in s')
        hdr['FPS'] = (tmp['internal_frame_rate'], 'Frame rate in Hz')
        hdr['INTERVAL'] = (tmp['internal_frame_interval'],
                           'Delay between two successive acquisitions in s')
        hdr['HPOS'] = (tmp['subarray_hpos'], 'X-position of the ROI')
        hdr['YPOS'] = (tmp['subarray_vpos'], 'Y-position of the ROI')
        hdr['TIME'] = (t, 'Local Time of Acquisition')

        file_name = dirc / pathlib.Path(str(cam.ID)+"_exp_" +
                                        str(np.round(cam.get_exposure() *
                                                     1000, 3)) + '_nframes_' +
                                        str(n_frames) + '.fits')

        hdu.writeto(file_name, overwrite=overwrite)

    return image


def display_phase_on_slm(phase, slm_flat=np.False_, slm_shape=[1152, 1920],
                         return_command_vector=False):

    if slm_flat.dtype == np.dtype("bool"):
        slm_flat = np.zeros([slm_shape[0]*slm_shape[1]])

    else:
        slm_flat = np.reshape(slm_flat, [slm_shape[0]*slm_shape[1]])

    phase = np.reshape(phase, [slm_shape[0]*slm_shape[1]])
    phase = np.mod(phase+slm_flat, 256)
    phase = phase.astype(dtype=np.uint8)

    # display pattern on slm
    slm_lib.Write_image(board_number,
                        phase.ctypes.data_as(ct.POINTER(ct.c_ubyte)),
                        slm_shape[0]*slm_shape[1],
                        wait_For_Trigger, OutputPulseImageFlip,
                        OutputPulseImageRefresh, timeout_ms)
    slm_lib.ImageWriteComplete(board_number, timeout_ms)

    if return_command_vector:
        return phase
    else:
        del phase
        return None

# %% parameters


# directory
dirc_data = pathlib.Path(
    __file__).parent.parent.parent.parent.parent / "data"

# slm shape
slm_shape = np.array([1152, 1920])

# number of phase measurements point
# n_subaperture = 20

# define pupil

# pupil radius in SLM pixels
pupil_radius = 300  # [pixel]

# pupil center on slm
pupil_center = (slm_shape/2).astype(int)  # [pixel]

# orcas exposure time
exposure_time = 200e-3    # exposure time (s)

# %% Link slm MEADOWLARK

# load slm library
ct.cdll.LoadLibrary(
    "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\Blink_C_wrapper")
slm_lib = ct.CDLL("Blink_C_wrapper")

# load image generation library
ct.cdll.LoadLibrary(
    "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\ImageGen")
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
OutputPulseImageRefresh = ct.c_uint(0)  # only supported on 1920x1152

# Call the Create_SDK constructor

# Returns a handle that's passed to subsequent SDK calls
slm_lib.Create_SDK(bit_depth, ct.byref(num_boards_found),
                   ct.byref(constructed_okay),
                   is_nematic_type, RAM_write_enable,
                   use_GPU, max_transients, 0)

if constructed_okay.value == 0:
    print("Blink SDK did not construct successfully")
    # Python ctypes assumes the return value is always int
    # We need to tell it the return type by setting restype
    slm_lib.Get_last_error_message.restype = ct.c_char_p
    print(slm_lib.Get_last_error_message())

if num_boards_found.value == 1:
    print("Blink SDK was successfully constructed")
    print("Found %s SLM controller(s)" % num_boards_found.value)
    height = ct.c_uint(slm_lib.Get_image_height(board_number))
    width = ct.c_uint(slm_lib.Get_image_width(board_number))
    center_x = ct.c_uint(width.value//2)
    center_y = ct.c_uint(height.value//2)

# %% By default load a linear LUT and a black WFC

slm_lib.Load_LUT_file(
    board_number,
    str(dirc_data / "phd_bioedge" / "manip" / "slm_lut" /
        "12bit_linear.lut").encode('utf-8'))
command = display_phase_on_slm(np.zeros(slm_shape), return_command_vector=True)
plt.figure()
plt.imshow(np.reshape(command, slm_shape))
plt.title("Command")

# %% Load LUT

slm_lib.Load_LUT_file(
    board_number,
    str(dirc_data / "phd_bioedge" / "manip" / "slm_lut" /
        "slm5758_at675.lut").encode('utf-8'))

# %% Load WFC

slm_flat = np.zeros(slm_shape)
# slm_flat = np.load(dirc_data / "phd_bioedge" / "manip" / "slm_wfc" /
#                    "slm5758_at675.npy")
command = display_phase_on_slm(slm_flat, return_command_vector=True)
plt.figure()
plt.imshow(np.reshape(command, slm_shape))
plt.title("Command")

# %% Link WFS cameras (ORCA)

DCAM.get_cameras_number()

# %% connect

cam1 = DCAM.DCAMCamera(idx=0)
cam2 = DCAM.DCAMCamera(idx=1)

if cam1.get_device_info().serial_number == 'S/N: 002369':
    orca_inline = cam1
    del cam1
    orca_folded = cam2
    del cam2

else:
    orca_inline = cam2
    del cam2
    orca_folded = cam1
    del cam1

# %% Setup cameras

# initialize settings
orca_inline.exp_time = exposure_time
orca_inline.n_frames = 3  # acquire cubes of n_frames images
orca_inline.ID = 0  # ID for the data saved

# initialize settings
orca_folded.exp_time = orca_inline.exp_time  # exposure time (s)
orca_folded.n_frames = orca_inline.n_frames  # acquire cubes of n_frames images
orca_folded.ID = 0  # ID for the data saved

# %% Create folder to save results

utc_now = get_utc_now()

dirc_closed_loop = dirc_data / "phd_bioedge" / \
    "manip" / "closed_loop" / utc_now

pathlib.Path(dirc_interaction_matrix).mkdir(parents=True, exist_ok=True)

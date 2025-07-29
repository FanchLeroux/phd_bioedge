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

from pylablib.devices import DCAM

from astropy.io import fits

from fanch.tools.miscellaneous import get_tilt, get_circular_pupil
from fanch.plots import make_gif

from OOPAO.tools.displayTools import displayMap

import math
import time

#%% Function declaration

# utc datetime now

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
def live_view(cams, roi=None, interval=0.005):
    """
    Live view for one or multiple cameras, displaying serial numbers.

    Parameters:
        cams: camera object or list of camera objects
        roi: region of interest (not currently used in this version)
        interval: pause time between frame updates
    """
    plt.ion()

    # Ensure cams is a list
    if not isinstance(cams, (list, tuple)):
        cams = [cams]

    for cam in cams:
        cam.set_exposure(cam.exp_time)

    # Get initial frames and serial numbers
    frames = [cam.grab(1)[0] for cam in cams]
    serials = [cam.get_device_info().serial_number for cam in cams]

    # Set up subplot layout
    n = len(cams)
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    ims, titles = [], []
    for i, (frame, ax) in enumerate(zip(frames, axes)):
        im = ax.imshow(frame, cmap='viridis')
        plt.colorbar(im, ax=ax)
        serial = serials[i]
        titles.append(ax.set_title(f"Serial: {serial} - \nMax: {np.max(frame):.2f}\nMean: {np.mean(frame):.2f}"))
        ims.append(im)

    # Hide unused subplots
    for ax in axes[n:]:
        ax.axis('off')

    # Live update loop
    while plt.fignum_exists(fig.number):
        for i, cam in enumerate(cams):
            frame = cam.grab(1)[0]
            ims[i].set_data(frame)
            ims[i].set_clim(vmin=np.min(frame), vmax=np.max(frame))
            serial = serials[i]  # <- Fix here
            titles[i].set_text(f"Serial: {serial} - \nMax: {np.max(frame):.2f}\nMean: {np.mean(frame):.2f}")
        plt.pause(interval)


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
        del phase
        return None

# def display_phase_on_slm(phase, slm_flat=None, slm_shape=(1152, 1920), return_command_vector=False):
#     slm_size = slm_shape[0] * slm_shape[1]

#     if slm_flat is None:
#         slm_flat = np.zeros(slm_size, dtype=np.uint8)
#     else:
#         slm_flat = np.asarray(slm_flat, dtype=np.uint8).reshape(slm_size)

#     phase = np.asarray(phase, dtype=np.uint8).reshape(slm_size)
#     phase = (phase + slm_flat) % 256

#     # display pattern on SLM
#     slm_lib.Write_image(
#         board_number,
#         phase.ctypes.data_as(ct.POINTER(ct.c_ubyte)),
#         slm_size,
#         wait_For_Trigger,
#         OutputPulseImageFlip,
#         OutputPulseImageRefresh,
#         timeout_ms
#     )
#     slm_lib.ImageWriteComplete(board_number, timeout_ms)

#     return phase if return_command_vector else None


def measure_interaction_matrix(path_to_slm_phase_screens, n_modes_calib, cam, n_frames, exp_time, 
                               slm_flat=np.False_, slm_shape=(1152, 1920), pupil_center=[565,1010],
                               roi=False, dirc = False, overwrite=False, display=True):
    
    # load modes (carefully)
    slm_phase_screens = np.load(path_to_slm_phase_screens, mmap_mode='r')
    
    # get one image to infer dimensions
    img = acquire(cam, 1, exp_time, roi=roi)
    
    interaction_matrix = np.zeros((img.shape[1], img.shape[2], slm_phase_screens.shape[2]), dtype=np.float64)
    
    if display:
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(nrows=1, ncols=2)
        im1 = ax[0].imshow(np.zeros((slm_phase_screens.shape[0],slm_phase_screens.shape[1])), cmap='viridis')
        im2 = ax[1].imshow(interaction_matrix[:,:,0], cmap='viridis')
        ax[0].set_title("SLM Command")
        ax[1].set_title("Detector Irradiance")
        plt.tight_layout()
        plt.show()
        
    
    for n_phase_screen in range(n_modes_calib):
        a =time.time()
        mode_full_slm = np.zeros((slm_shape[0], slm_shape[1]))
        mode_full_slm[pupil_center[0]-slm_phase_screens.shape[0]//2:
              pupil_center[0]+slm_phase_screens.shape[0]//2,
              pupil_center[1]-slm_phase_screens.shape[1]//2:
              pupil_center[1]+slm_phase_screens.shape[1]//2] = slm_phase_screens[:,:,n_phase_screen]
        display_phase_on_slm(mode_full_slm, slm_flat=slm_flat)
        b =time.time()
        print('display:'+str(b-a))
        interaction_matrix[:,:,n_phase_screen] = np.mean(acquire(cam, n_frames, exp_time, roi=roi), axis=0)
        c =time.time()
        print('orca:'+str(c-b))
        
        
        
        if display:
            im1.set_data(slm_phase_screens[:,:,n_phase_screen])
            im1.set_clim(vmin=np.min(slm_phase_screens[:,:,n_phase_screen]), vmax=np.max(slm_phase_screens[:,:,n_phase_screen]))  # Adjust color scale
            im2.set_data(interaction_matrix[:,:,n_phase_screen])
            im2.set_clim(vmin=np.min(interaction_matrix[:,:,n_phase_screen]), vmax=np.max(interaction_matrix[:,:,n_phase_screen]))  # Adjust color scale
            plt.pause(0.005)
        del mode_full_slm
    #display_phase_on_slm(slm_flat)
    
    if dirc != False:
        np.save(dirc, interaction_matrix)
    
    return interaction_matrix
        
#%% parameters

# directories
dirc_data = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent / "data"

# slm shape
slm_shape = np.array([1152,1920])

# number of phase measurements point
n_subaperture = 20

# pupil radius in SLM pixels
n_pixels_in_slm_pupil = 1100

# pupil center on slm
pupil_center = [565,1010] # [pixel]

amplitude_calibration = 1 # (std) [rad]

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

#%% By default load a linear LUT and a black WFC

slm_lib.Load_LUT_file(board_number, str(dirc_data / "slm" / "LUT" / "12bit_linear.lut").encode('utf-8'))
command = display_phase_on_slm(np.zeros(slm_shape), return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

#%% Load LUT

slm_lib.Load_LUT_file(board_number, str(dirc_data / "slm" / "LUT" / "slm5758_at675.lut").encode('utf-8'))
# slm_lib.Load_LUT_file(board_number, str(dirc_data / "slm" / "LUT" / "utc_2025-06-27_11-37-41_slm0_at675.lut").encode('utf-8'))

#%% Load WFC

# slm_flat = np.zeros(slm_shape)
slm_flat = np.load(dirc_data / "slm" / "WFC" / "slm5758_at675.npy")
command = display_phase_on_slm(slm_flat, return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

#%% Modify WFC to simulate pupil

# define pupil

# pupil radius in SLM pixels
pupil_radius = 300 # [pixel]
# pupil center on slm
pupil_center = [565,1010] # [pixel]

# get large tilt on slm shaped support

tilt_amplitude = 1000.0*np.pi # [rad]
tilt_angle = -135.0 # [deg]
tilt = get_tilt([1152,1920], theta=np.deg2rad(tilt_angle), amplitude = tilt_amplitude)/(2*np.pi) * 255.0

# replace pupil location by zeros

pupil = np.abs(get_circular_pupil(2*pupil_radius)-1)
tilt_out_of_pupil = np.ones([1152,1920])
tilt_out_of_pupil[pupil_center[0]-pupil_radius:pupil_center[0]+pupil_radius,
              pupil_center[1]-pupil_radius:pupil_center[1]+pupil_radius] = pupil
tilt_out_of_pupil = tilt_out_of_pupil * tilt

slm_flat = np.load(dirc_data / "slm" / "WFC" / "slm5758_at675.npy")
slm_flat = slm_flat + tilt_out_of_pupil

# Load new WFC

command = display_phase_on_slm(slm_flat, return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

#%% Link WFS cameras (ORCA)

DCAM.get_cameras_number()

#%% connect

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

#%% Setup cameras

# initialize settings
orca_inline.exp_time = 200e-3    # exposure time (s)
orca_inline.n_frames = 3         # acquire cubes of n_frames images
orca_inline.ID = 0               # ID for the data saved

# initialize settings
orca_folded.exp_time = orca_inline.exp_time    # exposure time (s)
orca_folded.n_frames = orca_inline.n_frames    # acquire cubes of n_frames images
orca_folded.ID = 0                             # ID for the data saved

#%% live view

roi=False
live_view([orca_inline, orca_folded])

#%% Link focal plane camera (Thorlabs)



#%% Load zernike modes

zernike_modes = np.load(dirc_data / "slm" / "modal_basis" / "zernike_modes" / 
                        ("zernike_modes_" + str(n_pixels_in_slm_pupil) +
                        "_pixels_in_slm_pupil_" + str(n_subaperture) +
                        "_subapertures.npy"))

zernike_modes_full_slm = np.zeros((slm_shape[0], slm_shape[1], zernike_modes.shape[2]))
zernike_modes_full_slm[pupil_center[0]-zernike_modes.shape[0]//2:
              pupil_center[0]+zernike_modes.shape[0]//2,
              pupil_center[1]-zernike_modes.shape[1]//2:
              pupil_center[1]+zernike_modes.shape[1]//2, :] = zernike_modes


#%% Load KL modes

KL_modes = np.load(dirc_data / "slm" / "modal_basis" / "KL_modes" / 
                        "KL_modes_600_pixels_in_slm_pupil_20_subapertures.npy")

KL_modes_full_slm = np.zeros((slm_shape[0], slm_shape[1], KL_modes.shape[2]))
KL_modes_full_slm[pupil_center[0]-KL_modes.shape[0]//2:
              pupil_center[0]+KL_modes.shape[0]//2,
              pupil_center[1]-KL_modes.shape[1]//2:
              pupil_center[1]+KL_modes.shape[1]//2, :] = KL_modes

#%% Load fourier modes

fourier_modes = np.load(dirc_data / "slm" / "modal_basis" / "fourier_modes" / 
                        "fourier_modes_1152_pixels_in_slm_pupil_20_subapertures.npy")

fourier_modes_full_slm = np.zeros((slm_shape[0], slm_shape[1], fourier_modes.shape[2]))
fourier_modes_full_slm[slm_shape[0]//2-fourier_modes.shape[0]//2:
              slm_shape[0]//2+fourier_modes.shape[0]//2,
              pupil_center[1]-fourier_modes.shape[1]//2:
              pupil_center[1]+fourier_modes.shape[1]//2, :] = fourier_modes

#%% Load horizontal fourier modes

horizontal_fourier_modes = np.load(dirc_data / "slm" / "modal_basis" / "fourier_modes" / 
                        "horizontal_fourier_modes_1152_pixels_in_slm_pupil_20_subapertures.npy")

horizontal_fourier_modes_full_slm = np.zeros((slm_shape[0], slm_shape[1], horizontal_fourier_modes.shape[2]))
horizontal_fourier_modes_full_slm[slm_shape[0]//2-horizontal_fourier_modes.shape[0]//2:
              slm_shape[0]//2+horizontal_fourier_modes.shape[0]//2,
              pupil_center[1]-horizontal_fourier_modes.shape[1]//2:
              pupil_center[1]+horizontal_fourier_modes.shape[1]//2, :] = horizontal_fourier_modes
    
#%% Load  vertical fourier modes

vertical_fourier_modes = np.load(dirc_data / "slm" / "modal_basis" / "fourier_modes" / 
                        "vertical_fourier_modes_1152_pixels_in_slm_pupil_20_subapertures.npy")

vertical_fourier_modes_full_slm = np.zeros((slm_shape[0], slm_shape[1], vertical_fourier_modes.shape[2]))
vertical_fourier_modes_full_slm[slm_shape[0]//2-vertical_fourier_modes.shape[0]//2:
              slm_shape[0]//2+vertical_fourier_modes.shape[0]//2,
              pupil_center[1]-vertical_fourier_modes.shape[1]//2:
              pupil_center[1]+vertical_fourier_modes.shape[1]//2, :] = vertical_fourier_modes
    
#%% Load diagonal fourier modes

diagonal_fourier_modes = np.load(dirc_data / "slm" / "modal_basis" / "fourier_modes" / 
                        "diagonal_fourier_modes_1152_pixels_in_slm_pupil_20_subapertures.npy")

diagonal_fourier_modes_full_slm = np.zeros((slm_shape[0], slm_shape[1], diagonal_fourier_modes.shape[2]))
diagonal_fourier_modes_full_slm[slm_shape[0]//2-diagonal_fourier_modes.shape[0]//2:
              slm_shape[0]//2+diagonal_fourier_modes.shape[0]//2,
              pupil_center[1]-diagonal_fourier_modes.shape[1]//2:
              pupil_center[1]+diagonal_fourier_modes.shape[1]//2, :] = diagonal_fourier_modes
    
#%% display zernike mode on slm

command = display_phase_on_slm(0.5*zernike_modes_full_slm[:,:,50], slm_flat, slm_shape=[1152,1920], return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

#%% display KL mode on slm

command = display_phase_on_slm(0.2*KL_modes_full_slm[:,:,100], slm_flat, slm_shape=[1152,1920], return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

#%% display fourier mode on slm

command = display_phase_on_slm(0.5*fourier_modes_full_slm[:,:,-1], slm_flat, slm_shape=[1152,1920], return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

#%% display horizontal fourier mode on slm

command = display_phase_on_slm(0.5*horizontal_fourier_modes_full_slm[:,:,9], slm_flat, slm_shape=[1152,1920], return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

#%% display vertical fourier mode on slm

command = display_phase_on_slm(0.1*vertical_fourier_modes_full_slm[:,:,-1], slm_flat, slm_shape=[1152,1920], return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

#%% display diagonal fourier mode on slm

command = display_phase_on_slm(0.2*diagonal_fourier_modes_full_slm[:,:,-1], slm_flat, slm_shape=[1152,1920], return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

#%% Load flat on SLM

command = display_phase_on_slm(slm_flat, return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

#%% Load zeros on SLM

command = display_phase_on_slm(np.zeros(slm_shape), return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

#%% Live view

roi=False
live_view([orca_inline, orca_folded], roi)

#%% Select valid pixels

n_frames = 100
threshold = 0.1

pupils_raw_orca_inline = np.median(acquire(orca_inline, n_frames=10, exp_time=orca_inline.exp_time, roi=roi), axis=0)
pupils_orca_inline = pupils_raw_orca_inline/pupils_raw_orca_inline.max()
pupils_orca_inline[pupils_orca_inline>threshold] = 1.0
pupils_orca_inline[pupils_orca_inline!=1.0] = 0.0

pupils_raw_orca_folded = np.median(acquire(orca_folded, n_frames=10, exp_time=orca_folded.exp_time, roi=roi), axis=0)
pupils_orca_folded = pupils_raw_orca_folded/pupils_raw_orca_folded.max()
pupils_orca_folded[pupils_orca_folded>threshold] = 1.0
pupils_orca_folded[pupils_orca_folded!=1.0] = 0.0

#%%

fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].imshow(pupils_orca_inline)
axs[1].imshow(pupils_orca_folded)

#%% Get rid of slm edges - orca_inline

x1, x2, x3, x4 = 450, 820, 1120, 1490
y1, y2, y3, y4 = 400, 800, 1320, 1690

pupils_orca_inline[:y1,:], pupils_orca_inline[y2:y3, :], pupils_orca_inline[y4:, :] = 0, 0, 0
pupils_orca_inline[:, :x1], pupils_orca_inline[:, x2:x3], pupils_orca_inline[:, x4:] = 0, 0, 0

#%%

plt.figure()
plt.imshow(pupils_orca_inline)

#%% Measure reference intensities

command = display_phase_on_slm(slm_flat, return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

reference_intensities_orca_inline = np.mean(acquire(orca_inline, n_frames=10, exp_time=orca_inline.exp_time), axis=0)

#%%

plt.figure()
plt.imshow(reference_intensities_orca_inline)

#%% Measure interaction matrix - orca_inline

# Load KL modes

slm_phase_screens = np.load(dirc_data / "slm" / "modal_basis" / "KL_modes" / 
                        "KL_modes_600_pixels_in_slm_pupil_20_subapertures.npy", mmap_mode='r')

#%%

n_phase_screens_calib = slm_phase_screens.shape[2]

amplitude_calibration = 0.1

display=True

# get one image to infer dimensions
img = acquire(orca_inline, 1, orca_inline.exp_time, roi=roi)

interaction_matrix = np.zeros((img.shape[1], img.shape[2], n_phase_screens_calib), dtype=np.float32)

if display:
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(nrows=1, ncols=2)
    im1 = ax[0].imshow(np.zeros((slm_phase_screens.shape[0],slm_phase_screens.shape[1])), cmap='viridis')
    im2 = ax[1].imshow(interaction_matrix[:,:,0], cmap='viridis')
    ax[0].set_title("SLM Command")
    ax[1].set_title("Detector Irradiance")
    plt.tight_layout()
    plt.show()

for n_phase_screen in range(n_phase_screens_calib):
    
    KL_mode_full_slm = np.zeros((slm_shape[0], slm_shape[1]))
    KL_mode_full_slm[pupil_center[0]-slm_phase_screens.shape[0]//2:
                  pupil_center[0]+slm_phase_screens.shape[0]//2,
                  pupil_center[1]-slm_phase_screens.shape[1]//2:
                  pupil_center[1]+slm_phase_screens.shape[1]//2] = slm_phase_screens[:,:,n_phase_screen]
    
    command = display_phase_on_slm(amplitude_calibration*KL_mode_full_slm, slm_flat, slm_shape=[1152,1920], return_command_vector=True)
    
    interaction_matrix[:,:,n_phase_screen] = np.mean(acquire(orca_inline, 3, orca_inline.exp_time, roi=roi), axis=0)
    
    print(str(n_phase_screen))
    
    if display:
        im1.set_data(slm_phase_screens[:,:,n_phase_screen])
        im1.set_clim(vmin=np.min(slm_phase_screens[:,:,n_phase_screen]), vmax=np.max(slm_phase_screens[:,:,n_phase_screen]))  # Adjust color scale
        im2.set_data(interaction_matrix[:,:,n_phase_screen])
        im2.set_clim(vmin=np.min(interaction_matrix[:,:,n_phase_screen]), vmax=np.max(interaction_matrix[:,:,n_phase_screen]))  # Adjust color scale
        plt.pause(0.005)
    
#%%

KL_modes_full_slm = np.zeros((slm_shape[0], slm_shape[1], KL_modes.shape[2]))
KL_modes_full_slm[pupil_center[0]-KL_modes.shape[0]//2:
              pupil_center[0]+KL_modes.shape[0]//2,
              pupil_center[1]-KL_modes.shape[1]//2:
              pupil_center[1]+KL_modes.shape[1]//2, :] = KL_modes

#%% try

n_modes_calib = 4

amplitude_calibration = 0.1

interaction_matrix_KL_modes = measure_interaction_matrix(dirc_data / "slm" / "modal_basis" / "KL_modes" / "KL_modes_600_pixels_in_slm_pupil_20_subapertures.npy",
                                                         n_modes_calib,
                                                         orca_inline, 3, exp_time=orca_inline.exp_time, slm_flat=slm_flat,
                                                         roi=False,
                                                         dirc = dirc_data / "orca" / "interaction_matrix" / "raw_interaction_matrix_fourier_modes.npy", 
                                                         overwrite=True,
                                                         display=False)

#%%
    
n_modes_calib = -1

amplitude_calibration = 0.1

modes = amplitude_calibration*KL_modes_full_slm[:,:,:n_modes_calib]

interaction_matrix_KL_modes = measure_interaction_matrix(modes,
                                                              orca_inline, 3, exp_time=orca_inline.exp_time, slm_flat=slm_flat,
                                                              roi=False,
                                                              dirc = dirc_data / "orca" / "interaction_matrix" / 
                                                              "raw_interaction_matrix_fourier_modes.npy", 
                                                              overwrite=True,
                                                              display=False)

#%%

# keep only valid pixels
interaction_matrix_KL_modes = interaction_matrix_KL_modes * pupils_orca_inline[:,:,np.newaxis]

#%%

interaction_matrix_KL_modes_substracted = interaction_matrix_KL_modes -\
    (reference_intensities_orca_inline * pupils_orca_inline)[:,:,np.newaxis]

#%%

interaction_matrix_KL_modes_normalized = interaction_matrix_KL_modes_substracted /\
    interaction_matrix_KL_modes_substracted.sum(axis=(0,1), keepdims=True)

#%%

interaction_matrix_KL_modes_normalized_reshaped = np.reshape(interaction_matrix_KL_modes_normalized[pupils_orca_inline == 1.0],
                                                             (int(pupils_orca_inline.sum()), 
                                                              interaction_matrix_KL_modes_normalized.shape[-1]))

#%% SVD

U, S, Vt = np.linalg.svd(interaction_matrix_KL_modes_normalized_reshaped, full_matrices=False)

plt.figure()
plt.plot(S/S.max())
plt.yscale('log')
#%%
support = np.zeros((2048,2048))

support[np.where(pupils_orca_inline==1)] = U[:,1]
plt.figure()
plt.imshow(support)


#%% Inversion

command_matrix = np.linalg.pinv(interaction_matrix_KL_modes_normalized_reshaped)

plt.figure()
plt.imshow(command_matrix @ interaction_matrix_KL_modes_normalized_reshaped)

#%% Measure test matrix - orca_inline

n_modes_calib =100

amplitude_calibration = 0.3

modes = amplitude_calibration*KL_modes_full_slm[:,:,:n_modes_calib]

#%%

test_matrix_KL_modes = measure_interaction_matrix(modes,
                                                              orca_inline, 3, exp_time=orca_inline.exp_time, slm_flat=slm_flat,
                                                              roi=False,
                                                              dirc = dirc_data / "orca" / "interaction_matrix" / 
                                                              "raw_test_matrix_fourier_modes.npy", 
                                                              overwrite=True,
                                                              display=True)

#%%

# keep only valid pixels
test_matrix_KL_modes = test_matrix_KL_modes * pupils_orca_inline[:,:,np.newaxis]

#%%

test_matrix_KL_modes_substracted = test_matrix_KL_modes - (reference_intensities_orca_inline * pupils_orca_inline)[:,:,np.newaxis]

#%%

test_matrix_KL_modes_normalized = test_matrix_KL_modes_substracted / test_matrix_KL_modes_substracted.sum(axis=(0,1), 
                                                                                                          keepdims=True)

#%%

test_matrix_KL_modes_normalized_reshaped = np.reshape(test_matrix_KL_modes_normalized[pupils_orca_inline == 1.0],
                                                             (int(pupils_orca_inline.sum()), 
                                                              test_matrix_KL_modes_normalized.shape[-1]))

#%%

plt.figure()
plt.imshow(command_matrix @ test_matrix_KL_modes_normalized_reshaped)

#%% End connection with SLM

# Load a linear LUT and a black WFC

slm_lib.Load_LUT_file(board_number, str(dirc_data / "slm" / "LUT" / "12bit_linear.lut").encode('utf-8'))
display_phase_on_slm(np.zeros(slm_shape))

#%% delete slm sdk

# Always call Delete_SDK before exiting
slm_lib.Delete_SDK()

#%% End conection with cameras

orca_inline.close()
orca_folded.close()

#%% Bonus - Make gif

make_gif(dirc_data / "interaction_matrix_measeurements.gif", interaction_matrix_KL_modes)
make_gif(dirc_data / "test_matrix_measeurements.gif", test_matrix_KL_modes)


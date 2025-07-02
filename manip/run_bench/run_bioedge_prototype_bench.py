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

from fanch.tools.miscellaneous import get_tilt, get_circular_pupil, zeros_padding

from OOPAO.Telescope import Telescope
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Source import Source
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C

from OOPAO.Zernike import Zernike

from scipy.ndimage import zoom

#%%

dirc_data = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent / "data"

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

def live_view(get_frame_func, cam, roi, dirc = False, overwrite=True, interval=0.005):
    
    cam.n_frames = 1
    
    plt.ion()  # Turn on interactive mode

    # First frame for setup
    frame = get_frame_func(cam, cam.n_frames, cam.exp_time, roi=roi, dirc = False, overwrite=True)[0,:,:,]
    is_color = frame.ndim == 3 and frame.shape[2] == 3

    fig, ax = plt.subplots()
    im = ax.imshow(frame, cmap='viridis' if not is_color else None)
    cbar = plt.colorbar(im, ax=ax)
    title = fig.suptitle(f"Max value: {np.max(frame):.2f}", fontsize=24)

    while plt.fignum_exists(fig.number):  # Loop while window is open
        frame = get_frame_func(cam, cam.n_frames, cam.exp_time, roi=roi, dirc = False, overwrite=True)[0,:,:,]
        im.set_data(frame)
        im.set_clim(vmin=np.min(frame), vmax=np.max(frame))  # Adjust color scale
        title.set_text(f"Max value: {np.max(frame):.2f}")  # Update title
        plt.pause(interval)

    plt.ioff()

def get_slm_pupil_tilt(pupil_radius, pupil_center, delta_phi, theta=0, slm_shape = [1152,1920]):
    
    phase_map = np.zeros(slm_shape)
    pupil = get_circular_pupil(2*pupil_radius)
    tilt = get_tilt(pupil.shape, amplitude=1., theta=theta)
    pupil = tilt * pupil
    pupil = delta_phi*(pupil - pupil.min())/(pupil.max() - pupil.min())
    phase_map[pupil_center[1]-pupil_radius:pupil_center[1]+pupil_radius,
              pupil_center[0]-pupil_radius:pupil_center[0]+pupil_radius] = pupil
    
    phase_map = (phase_map/(2*np.pi) * 255).astype(np.uint8)
    
    phase_map = np.reshape(phase_map, slm_shape[0]*slm_shape[1], 'C')
    
    return phase_map

#%% parameters

# pupil radius in SLM pixels
pupil_radius = 550

# pupil center on slm
pupil_center = [860,575]

amplitude_calibration = 1 # (std) [rad]

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
slm_flat = np.reshape(slm_flat, [width.value*height.value], 'C')
slm_lib.Write_image(board_number, slm_flat.ctypes.data_as(ct.POINTER(ct.c_ubyte)), height.value*width.value, 
                    wait_For_Trigger, OutputPulseImageFlip, OutputPulseImageRefresh,timeout_ms)
slm_lib.ImageWriteComplete(board_number, timeout_ms)

#%% apply tilt on slm

tilt_amplitude = -50.0*np.pi # [rad]
tilt_angle = -90.0 # [deg]
tilt = get_tilt([1920, 1152], theta=np.deg2rad(tilt_angle), amplitude = tilt_amplitude)/(2*np.pi) * 255.0
tilt = np.reshape(tilt, [1152*1920])
tilt = np.mod(tilt+slm_flat, 256)
tilt = tilt.astype(dtype=np.uint8)

plt.figure(); plt.imshow(np.reshape(tilt, [1152,1920]))

# display pattern on slm
slm_lib.Write_image(board_number, tilt.ctypes.data_as(ct.POINTER(ct.c_ubyte)), height.value*width.value, 
                    wait_For_Trigger, OutputPulseImageFlip, OutputPulseImageRefresh,timeout_ms)
slm_lib.ImageWriteComplete(board_number, timeout_ms)

#%% Find pupil footprit on SLM

# generate SLM phase screen
pupil = get_circular_pupil(2*pupil_radius)
tilt_in_pupil = np.zeros([1152,1920])
tilt_in_pupil[pupil_center[1]-pupil_radius:pupil_center[1]+pupil_radius,
              pupil_center[0]-pupil_radius:pupil_center[0]+pupil_radius] = pupil

tilt = get_tilt([1920, 1152], theta=np.deg2rad(tilt_angle), amplitude = tilt_amplitude)/(2*np.pi) * 255.0
tilt_in_pupil = tilt_in_pupil * tilt
tilt_in_pupil = np.reshape(tilt_in_pupil, [1152*1920])
tilt_in_pupil = np.mod(tilt_in_pupil+slm_flat, 256)
tilt_in_pupil = tilt_in_pupil.astype(dtype=np.uint8)

# display pattern on slm
slm_lib.Write_image(board_number, tilt_in_pupil.ctypes.data_as(ct.POINTER(ct.c_ubyte)), height.value*width.value, 
                    wait_For_Trigger, OutputPulseImageFlip, OutputPulseImageRefresh,timeout_ms)
slm_lib.ImageWriteComplete(board_number, timeout_ms)

plt.figure(); plt.imshow(np.reshape(tilt_in_pupil, [1152,1920]))

#%% Compute KL basis

# get KL modes from OOPAO simulation
KL_modes_from_simulation = np.load(dirc_data / "slm" / "slm_phase_screens" 
                                 / "KL_basis" / "modal_basis_OPD_in_tel_pupil.npy")

# Keep only the first modes to speed up the computations
KL_modes_from_simulation = KL_modes_from_simulation[:,:,:10]

# normalize in std and consider amplitude_calibration
KL_modes_from_simulation = amplitude_calibration * (KL_modes_from_simulation\
                                                    /np.std(KL_modes_from_simulation, axis=(0,1)))

# scale for 2*np.pi <=> 255
KL_modes_from_simulation = KL_modes_from_simulation * 255/(2*np.pi)

# offset at half the dynamic
KL_modes_from_simulation[KL_modes_from_simulation!=0] = KL_modes_from_simulation\
    [KL_modes_from_simulation!=0] + 128.0

# adapt to slm shape
KL_modes_slm_shape = zoom(KL_modes_from_simulation, (2*pupil_radius/KL_modes_from_simulation.shape[0],
                                                    2*pupil_radius/KL_modes_from_simulation.shape[0],
                                                    1))

# multiply by pupil
KL_modes_slm_shape = KL_modes_slm_shape * get_circular_pupil(2*pupil_radius)[:,:,np.newaxis]


np.save(dirc_data / "slm" / "slm_phase_screens" 
                                 / "KL_basis" / "KL_modes_slm_shape.npy", KL_modes_slm_shape)

#%%

KL_modes_full_slm = np.zeros([1152,1920,KL_modes_slm_shape.shape[2]])

KL_modes_full_slm[pupil_center[1]-pupil_radius:pupil_center[1]+pupil_radius,
              pupil_center[0]-pupil_radius:pupil_center[0]+pupil_radius,:] = KL_modes_slm_shape[:,:,:]
# add WFC
# KL_modes_full_slm = np.reshape(KL_modes_full_slm, [1152*1920, KL_modes_full_slm.shape[2]])
# KL_modes_full_slm = KL_modes_full_slm+np.reshape(slm_flat, (slm_flat.shape[0], 1))

KL_modes_slm = np.empty([KL_modes_full_slm.shape[0]*KL_modes_full_slm.shape[1], KL_modes_full_slm.shape[2]])

for k in range(KL_modes_full_slm.shape[2]):
    KL_modes_slm[:,k] = np.reshape(KL_modes_full_slm[:,:,k], 
                                    [KL_modes_full_slm.shape[0]*KL_modes_full_slm.shape[1]])
    KL_modes_slm[:,k] = np.mod(KL_modes_slm[:,k] + slm_flat, 256)

KL_modes_slm = KL_modes_slm.astype(dtype=np.uint8)

#%%

plt.figure(); plt.imshow(np.reshape(KL_modes_slm[:,0], [1152,1920]))

#%%

slm_lib.Write_image(board_number, KL_modes_slm[:,0].ctypes.data_as(ct.POINTER(ct.c_ubyte)), height.value*width.value, 
                    wait_For_Trigger, OutputPulseImageFlip, OutputPulseImageRefresh,timeout_ms)
slm_lib.ImageWriteComplete(board_number, timeout_ms)

#%% Compute Zernike basis

tel = Telescope(2*pupil_radius, diameter=8)
Z = Zernike(tel,10) # create a Zernike object considering 300 polynomials
Z.computeZernike(tel) # compute the Zernike
zernike_modes = Z.modesFullRes

# std normalization and set amplitude calibration
zernike_modes = amplitude_calibration * (zernike_modes\
                                                    /np.std(zernike_modes, axis=(0,1)))
    
# scale phase such that 0-2pi <=> 0-255
zernike_modes = zernike_modes * 255/(2*np.pi)
    
# offset modes around half of the slm dynamic
zernike_modes = zernike_modes + 128.0
zernike_modes = zernike_modes * tel.pupil[:,:,np.newaxis]

# create SLM phase maps

zernike_modes_full_slm = np.zeros([1152,1920,zernike_modes.shape[2]])

zernike_modes_full_slm[pupil_center[1]-pupil_radius:pupil_center[1]+pupil_radius,
              pupil_center[0]-pupil_radius:pupil_center[0]+pupil_radius,:] = zernike_modes

#%% apply zernike on slm

zernike_mode = zernike_modes_full_slm[:,:,6]
zernike_mode = np.reshape(zernike_mode, [1152*1920])
zernike_mode = np.mod(zernike_mode+slm_flat, 256)
zernike_mode = zernike_mode.astype(dtype=np.uint8)

# display pattern on slm
slm_lib.Write_image(board_number, zernike_mode.ctypes.data_as(ct.POINTER(ct.c_ubyte)), height.value*width.value, 
                    wait_For_Trigger, OutputPulseImageFlip, OutputPulseImageRefresh,timeout_ms)
slm_lib.ImageWriteComplete(board_number, timeout_ms)

plt.figure(); plt.imshow(np.reshape(zernike_mode, [1152,1920]))
plt.figure(); plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j * zernike_modes_full_slm[:,:,6] * 2*np.pi / 255))))**2)

#%% Load WFC

slm_flat = np.asarray(Image.open(str(dirc_data / "slm" / "WFC" / "slm5758_at675.bmp")), dtype=np.float64)
slm_flat = slm_flat / slm_flat.max() * 255.0
slm_flat = slm_flat.astype(dtype=np.uint8)
slm_flat = np.reshape(slm_flat, [width.value*height.value], 'C')
slm_lib.Write_image(board_number, slm_flat.ctypes.data_as(ct.POINTER(ct.c_ubyte)), height.value*width.value, 
                    wait_For_Trigger, OutputPulseImageFlip, OutputPulseImageRefresh,timeout_ms)
slm_lib.ImageWriteComplete(board_number, timeout_ms)

#%% Setup camera

# initialize settings
cam.exp_time = 5e-3    # exposure time (s)
cam.n_frames = 10      # acquire cubes of n_frames images
cam.ID = 0             # ID for the data saved
roi = False

#%% Live view

live_view(acquire, cam, roi)

#%% Load a linear LUT and a black WFC

slm_lib.Load_LUT_file(board_number, str(dirc_data / "slm" / "LUT" / "12bit_linear.lut").encode('utf-8'))
slm_flat = np.asarray(Image.open(str(dirc_data / "slm" / "WFC" / "1920black.bmp")))
slm_flat = np.reshape(slm_flat, [width.value*height.value], 'C')
slm_lib.Write_image(board_number, slm_flat.ctypes.data_as(ct.POINTER(ct.c_ubyte)), height.value*width.value, 
                    wait_For_Trigger, OutputPulseImageFlip, OutputPulseImageRefresh,timeout_ms)
slm_lib.ImageWriteComplete(board_number, timeout_ms)

#%% delete slm sdk

# Always call Delete_SDK before exiting
slm_lib.Delete_SDK()

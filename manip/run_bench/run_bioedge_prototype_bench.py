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

from pylablib.devices import DCAM

from astropy.io import fits

from fanch.plots import make_gif

from fanch.tools.miscellaneous import zeros_padding

from OOPAO.tools.displayTools import displayMap

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

def measure_interaction_matrix(slm_phase_screens, cam, n_frames, exp_time, slm_flat=np.False_,
                               roi=False, dirc = False, overwrite=False, display=True):
    
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
        
    for n_mode in range(slm_phase_screens.shape[2]):
        display_phase_on_slm(slm_phase_screens[:,:,n_mode], slm_flat=slm_flat)
        interaction_matrix[:,:,n_mode] = np.mean(acquire(cam, n_frames, exp_time, roi=roi), axis=0)
        
        if display:
            im1.set_data(slm_phase_screens[:,:,n_mode])
            im1.set_clim(vmin=np.min(slm_phase_screens[:,:,n_mode]), vmax=np.max(slm_phase_screens[:,:,n_mode]))  # Adjust color scale
            im2.set_data(interaction_matrix[:,:,n_mode])
            im2.set_clim(vmin=np.min(interaction_matrix[:,:,n_mode]), vmax=np.max(interaction_matrix[:,:,n_mode]))  # Adjust color scale
            plt.pause(0.005)
        
    display_phase_on_slm(slm_flat)
    
    return interaction_matrix
        
#%% parameters

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
slm_flat = np.asarray(Image.open(str(dirc_data / "slm" / "WFC" / "1920black.bmp")))
slm_flat = np.reshape(slm_flat, [width.value*height.value], 'C')
slm_lib.Write_image(board_number, slm_flat.ctypes.data_as(ct.POINTER(ct.c_ubyte)), height.value*width.value, 
                    wait_For_Trigger, OutputPulseImageFlip, OutputPulseImageRefresh,timeout_ms)
slm_lib.ImageWriteComplete(board_number, timeout_ms)

#%% Load LUT

slm_lib.Load_LUT_file(board_number, str(dirc_data / "slm" / "LUT" / "slm5758_at675.lut").encode('utf-8'))
slm_lib.Load_LUT_file(board_number, str(dirc_data / "slm" / "LUT" / "utc_2025-06-27_11-37-41_slm0_at675.lut").encode('utf-8'))

#%% Load WFC

slm_flat = np.zeros(slm_shape)
slm_flat = np.load(dirc_data / "slm" / "WFC" / "slm5758_at675.npy")
#slm_flat = np.load(dirc_data / "slm" / "WFC" / "slm5758_at675_tilt_amplitude6pi_tilt_angle_45degree.npy")
command = display_phase_on_slm(slm_flat, return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

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

#%% display zernike mode on slm

command = display_phase_on_slm(0.1*zernike_modes_full_slm[:,:,-50], slm_flat, slm_shape=[1152,1920], return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

#%% Load KL modes

KL_modes = np.load(dirc_data / "slm" / "modal_basis" / "KL_modes" / 
                        "KL_modes_600_pixels_in_slm_pupil_20_subapertures.npy")

KL_modes_full_slm = np.zeros((slm_shape[0], slm_shape[1], KL_modes.shape[2]))
KL_modes_full_slm[pupil_center[0]-KL_modes.shape[0]//2:
              pupil_center[0]+KL_modes.shape[0]//2,
              pupil_center[1]-KL_modes.shape[1]//2:
              pupil_center[1]+KL_modes.shape[1]//2, :] = KL_modes

#%% display KL mode on slm

command = display_phase_on_slm(KL_modes_full_slm[:,:,2], slm_flat, slm_shape=[1152,1920], return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

#%% Load fourier modes

fourier_modes = np.load(dirc_data / "slm" / "modal_basis" / "fourier_modes" / 
                        "fourier_modes_1152_pixels_in_slm_pupil_20_subapertures.npy")

fourier_modes_full_slm = np.zeros((slm_shape[0], slm_shape[1], fourier_modes.shape[2]))
fourier_modes_full_slm[slm_shape[0]//2-fourier_modes.shape[0]//2:
              slm_shape[0]//2+fourier_modes.shape[0]//2,
              pupil_center[1]-fourier_modes.shape[1]//2:
              pupil_center[1]+fourier_modes.shape[1]//2, :] = fourier_modes

#%% display fourier mode on slm

n_mode = -24

fourier_mode = fourier_modes_full_slm[:,:,n_mode] - 128.0
fourier_mode[fourier_modes_full_slm[:,:,n_mode] == 0] = 0
fourier_mode = 128*np.ones(slm_shape) + 0.2*fourier_mode

slm_flat = np.zeros(slm_shape)
# command = display_phase_on_slm(1*fourier_modes_full_slm[:,:,-24], slm_flat, slm_shape=[1152,1920], return_command_vector=True)
# slm_flat = np.load(dirc_data / "slm" / "WFC" / "slm_no_WFC_tilt_amplitude_120pi_tilt_angle_135degree.npy")
# slm_flat = np.load(dirc_data / "slm" / "WFC" / "slm5758_at675.npy")
command = display_phase_on_slm(fourier_mode, slm_flat, slm_shape=[1152,1920], return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

#%% Load mode without flat on SLM

command = display_phase_on_slm(fourier_mode,slm_shape=[1152,1920], return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

#%% Load flat on SLM

command = display_phase_on_slm(slm_flat, return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

#%% Load zeros on SLM

command = display_phase_on_slm(np.zeros(slm_shape), return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

#%% Link camera ORCA

cam = DCAM.DCAMCamera()

#%% Setup camera

# initialize settings
cam.exp_time = 5e-3    # exposure time (s)
cam.n_frames = 10      # acquire cubes of n_frames images
cam.ID = 0             # ID for the data saved
roi = False

#%% Live view

live_view(acquire, cam, roi)

#%% Enter roi

roi = [900, 830, 500, 500] # roi[0] is x coordinate, i.e column number

#%% Check roi

live_view(acquire, cam, roi)

#%% Acquire image

data_orca = acquire(cam, n_frames=10, exp_time=5e-3, roi=roi,
                    dirc = dirc_data / "orca", overwrite=True)



#%% Load flat on SLM

command = display_phase_on_slm(slm_flat, return_command_vector=True)
plt.figure(); plt.imshow(np.reshape(command, slm_shape)); plt.title("Command")

#%% Make interaction matrix

interaction_matrix_fourier_modes = measure_interaction_matrix(fourier_modes_full_slm[:,:,50:60], 
                                                              cam, 10, exp_time=10e-3, slm_flat=slm_flat,
                                                              roi=roi, dirc = False, overwrite=False)

#%%

interaction_matrix_zernike_modes = measure_interaction_matrix(zernike_modes_full_slm[:,:,0:20], 
                                                              cam, 10, exp_time=10e-3, slm_flat=slm_flat,
                                                              roi=roi, dirc = False, overwrite=False)

#%%

interaction_matrix_KL_modes = measure_interaction_matrix(KL_modes_full_slm[:,:,0:20], 
                                                         cam, 10, exp_time=10e-3, slm_flat=slm_flat,
                                                         roi=roi, dirc = False, overwrite=False,
                                                         display=True)

#%% Plot interaction matrix

displayMap(interaction_matrix_KL_modes)

#%% save as gif

make_gif(dirc_data / "orca" / "fourier_modes_with_slm.gif", interaction_matrix_fourier_modes)

#%% Load a linear LUT and a black WFC

slm_lib.Load_LUT_file(board_number, str(dirc_data / "slm" / "LUT" / "12bit_linear.lut").encode('utf-8'))
display_phase_on_slm(np.zeros(slm_shape))

#%% delete slm sdk

# Always call Delete_SDK before exiting
slm_lib.Delete_SDK()

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 17:02:10 2025

@author: lgs
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
        titles.append(ax.set_title(
            f"Serial: {serial} - \nMax: {np.max(frame):.2f}\nMean: "
            f"{np.mean(frame):.2f}"))
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
            titles[i].set_text(
                f"Serial: {serial} - \nMax: {np.max(frame):.2f}\nMean: "
                f"{np.mean(frame):.2f}")
        plt.pause(interval)


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

# get large tilt on slm shaped support
tilt_amplitude = 1000.0*np.pi  # [rad]
tilt_angle = -135.0  # [deg]

stroke = 1.  # (std) [SLM units if std normalized modes are used]
stroke_test_matrix = 10.  # (std) [rad]

# orcas exposure time
exposure_time = 200e-3    # exposure time (s)

# valid pixel selection
n_frames = 100
threshold = 0.1

# calibration

# Load KL modes
filename = ("KL_modes_slm_units_600_pixels_in_slm_pupil_20_subapertures_"
            "no_wrapping_required.npy")
KL_modes = np.load(
    dirc_data / "phd_bioedge" / "manip" / "slm_screens" / "modal_basis" /
    "KL_modes" / filename,
    mmap_mode='r')
# chose how many modes are used to calibrate
n_calibration_modes = KL_modes.shape[0]

# do gif
do_gif = False

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

# %% live view - align Bi-O-Edge mask

roi = False
live_view([orca_inline, orca_folded])

# %% Create folder to save results

utc_now = get_utc_now()

dirc_interaction_matrix = dirc_data / "phd_bioedge" / \
    "manip" / "interaction_matrix" / utc_now

pathlib.Path(dirc_interaction_matrix).mkdir(parents=True, exist_ok=True)

# %% Compute calibration modes

calibration_modes = np.mod(stroke * KL_modes, 256).astype(np.uint8)

# %% Save calibration modes

filename = (utc_now + "_calibration_modes_slm_units.npy")
np.save(dirc_interaction_matrix / filename,
        calibration_modes)

# %% Measure reference intensities

command = display_phase_on_slm(slm_flat, return_command_vector=True)

reference_intensities_orca_inline = np.mean(
    acquire(orca_inline, n_frames=10,
            exp_time=orca_inline.exp_time), axis=0)

plt.figure()
plt.imshow(reference_intensities_orca_inline)

# %% Save reference intensities

filename = (utc_now + "_reference_intensities_orca_inline.npy")
np.save(dirc_interaction_matrix / filename,
        reference_intensities_orca_inline)

# %% Measure interaction matrix - orca_inline

display = True

# get one image to infer dimensions
img = acquire(orca_inline, 1, orca_inline.exp_time, roi=roi)

interaction_matrix = np.zeros((img.shape[1], img.shape[2],
                               n_calibration_modes), dtype=np.float32)

if display:

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(nrows=1, ncols=2)
    im1 = ax[0].imshow(np.zeros((calibration_modes.shape[1],
                                 calibration_modes.shape[2])), cmap='viridis')
    im2 = ax[1].imshow(interaction_matrix[:, :, 0], cmap='viridis')
    ax[0].set_title("SLM Command")
    ax[1].set_title("Detector Irradiance")
    plt.tight_layout()
    plt.show()

for n_phase_screen in tqdm.tqdm(range(n_calibration_modes)):

    calibration_modes_full_slm = np.zeros((slm_shape[0], slm_shape[1]))
    calibration_modes_full_slm[pupil_center[0]-calibration_modes.shape[1]//2:
                               pupil_center[0]+calibration_modes.shape[1]//2,
                               pupil_center[1]-calibration_modes.shape[2]//2:
                               pupil_center[1] +
                                   calibration_modes.shape[2]//2] =\
        calibration_modes[n_phase_screen, :, :]

    command = display_phase_on_slm(
        stroke*calibration_modes_full_slm,
        slm_flat, slm_shape=[1152, 1920], return_command_vector=True)

    push = np.mean(acquire(orca_inline, 3, orca_inline.exp_time, roi=roi),
                   axis=0)

    command = display_phase_on_slm(
        -stroke*calibration_modes_full_slm, slm_shape=[1152, 1920],
        return_command_vector=True)

    pull = np.mean(acquire(orca_inline, 3, orca_inline.exp_time, roi=roi),
                   axis=0)

    interaction_matrix[:, :, n_phase_screen] = push/push.sum()\
        - pull/pull.sum()

    del push, pull

    if display:
        im1.set_data(calibration_modes[n_phase_screen, :, :])
        im1.set_clim(vmin=np.min(calibration_modes[n_phase_screen, :, :]),
                     vmax=np.max(calibration_modes[n_phase_screen, :, :]))
        im2.set_data(interaction_matrix[:, :, n_phase_screen])
        im2.set_clim(vmin=np.min(interaction_matrix[:, :, n_phase_screen]),
                     vmax=np.max(interaction_matrix[:, :, n_phase_screen]))
        plt.pause(0.01)

# %% Save interaction matrix

np.save(dirc_interaction_matrix /
        (utc_now + "_interaction_matrix_push_pull_orca_inline.npy"),
        interaction_matrix)

# %% Measure test matrix - orca_inline

display = True

# get one image to infer dimensions
img = acquire(orca_inline, 1, orca_inline.exp_time, roi=roi)

test_matrix = np.zeros((img.shape[1], img.shape[2], n_calibration_modes),
                       dtype=np.float32)

if display:

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(nrows=1, ncols=2)
    im1 = ax[0].imshow(np.zeros((calibration_modes.shape[0],
                                 calibration_modes.shape[1])), cmap='viridis')
    im2 = ax[1].imshow(test_matrix[:, :, 0], cmap='viridis')
    ax[0].set_title("SLM Command")
    ax[1].set_title("Detector Irradiance")
    plt.tight_layout()
    plt.show()

for n_phase_screen in range(n_calibration_modes):

    calibration_modes_full_slm = np.zeros((slm_shape[0], slm_shape[1]))
    calibration_modes_full_slm[pupil_center[0]-calibration_modes.shape[0]//2:
                               pupil_center[0]+calibration_modes.shape[0]//2,
                               pupil_center[1]-calibration_modes.shape[1]//2:
                               pupil_center[1]
                                   + calibration_modes.shape[1]//2] =\
        calibration_modes[n_phase_screen, :, :]

    command = display_phase_on_slm(
        stroke_test_matrix*calibration_modes_full_slm, slm_flat,
        slm_shape=[1152, 1920], return_command_vector=True)

    test_matrix[:, :, n_phase_screen] = np.mean(
        acquire(orca_inline, 3,
                orca_inline.exp_time, roi=roi), axis=0)

    print(str(n_phase_screen))

    if display:
        im1.set_data(calibration_modes[n_phase_screen, :, :])
        im1.set_clim(vmin=np.min(calibration_modes[n_phase_screen, :, :]),
                     vmax=np.max(calibration_modes[n_phase_screen, :, :]))
        im2.set_data(test_matrix[:, :, n_phase_screen])
        im2.set_clim(vmin=np.min(test_matrix[:, :, n_phase_screen]),
                     vmax=np.max(test_matrix[:, :, n_phase_screen]))
        plt.pause(0.005)

# %% Save test matrix

np.save(dirc_interaction_matrix / (utc_now + "_test_matrix.npy"), test_matrix)

# %% End connection with SLM

# Load a linear LUT and a black WFC

slm_lib.Load_LUT_file(
    board_number,
    str(dirc_data / "slm" / "LUT" / "12bit_linear.lut").encode('utf-8'))
display_phase_on_slm(np.zeros(slm_shape))

# Always call Delete_SDK before exiting
slm_lib.Delete_SDK()

# %% End conection with cameras

orca_inline.close()
orca_folded.close()

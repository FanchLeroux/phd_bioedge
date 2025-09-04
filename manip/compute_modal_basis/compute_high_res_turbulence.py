# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:01:52 2024

@author: fleroux
"""

# %% Imports

import pathlib

import numpy as np
import matplotlib.pyplot as plt
import tqdm

from OOPAO.Telescope import Telescope
from OOPAO.Source import Source
from OOPAO.Atmosphere import Atmosphere
from OOPAO.tools.interpolateGeometricalTransformation import interpolate_cube

# %% File path

dirc_data = (
    pathlib.Path(__file__).parent.parent.parent.parent.parent
    / "data" / "data_banc_proto_bioedge" / "turbulence"
)

# %% Parameters

WAVELENGTH = 675e-9  # [m]

n_subaperture = 20
n_pixels_in_slm_pupil = 1152

r0 = 0.15  # Fried Parameter [m]
L0 = 25  # Outer Scale [m]
fractionalR0 = [0.45, 0.1, 0.1, 0.25, 0.1]  # Cn2 Profile
windSpeed = [10, 12, 11, 15, 20]  # Wind Speed [m/s]
windDirection = [0, 72, 144, 216, 288]  # Wind Direction [deg]
altitude = [0, 1000, 5000, 10000, 12000]  # Altitude Layers [m]

n_opd_screens = 10
seed = 0

# %% Telescope

tel = Telescope(
    resolution=8 * n_subaperture,
    diameter=8,
    samplingTime=1 / 1000,
    centralObstruction=0.0,
    display_optical_path=False,
    fov=10
)

tel_hr = Telescope(
    resolution=n_pixels_in_slm_pupil,
    diameter=8,
    samplingTime=1 / 1000,
    centralObstruction=0.0,
    display_optical_path=False,
    fov=10
)

# %% Natural Guide Star

ngs = Source(
    optBand='I',
    magnitude=8,
    coordinates=[0, 0]
)

ngs * tel
ngs * tel_hr

# %% Atmosphere

atm = Atmosphere(
    telescope=tel,
    r0=r0,
    L0=L0,
    fractionalR0=fractionalR0,
    windSpeed=windSpeed,
    windDirection=windDirection,
    altitude=altitude
)

atm.initializeAtmosphere(tel)
tel + atm

# %% Compute atmosphere opd screens

atmosphere_opd_screens = np.full(
    (n_opd_screens,) + atm.OPD_no_pupil.shape,
    np.nan,
    dtype=float
)

for k in tqdm.tqdm(range(n_opd_screens)):
    atmosphere_opd_screens[k, :, :] = atm.OPD_no_pupil
    atm.update()

# %% Interpolation to high resolution

atmosphere_opd_screens_hr = interpolate_cube(
    atmosphere_opd_screens,
    tel.pixelSize,
    tel_hr.pixelSize,
    tel_hr.resolution
)

# %% Convert to SLM phase map

# get a version with no wraping required
atmosphere_opd_screens_hr_no_wraping_required =\
    atmosphere_opd_screens_hr - \
    atmosphere_opd_screens_hr.mean(axis=(1, 2), keepdims=True)  # set zero mean
atmosphere_opd_screens_hr_no_wraping_required = 127.5 *\
    atmosphere_opd_screens_hr_no_wraping_required /\
    np.array([atmosphere_opd_screens_hr_no_wraping_required.max(),
              -atmosphere_opd_screens_hr_no_wraping_required.min()]).max() +\
    127.5

# set piston at half the wavelength (slm dynamic range in meter)
atmosphere_opd_screens_hr_piston_corrected = atmosphere_opd_screens_hr - \
    atmosphere_opd_screens_hr.mean(axis=(1, 2), keepdims=True) + WAVELENGTH/2.

# scale between 0 and 255
atmosphere_slm_screens_hr_slm_units =\
    atmosphere_opd_screens_hr_piston_corrected * 255./WAVELENGTH

# wraping
atmosphere_slm_screens_hr_wrapped = np.mod(
    atmosphere_slm_screens_hr_slm_units, 256)

# encode on 8-bit
atmosphere_slm_screens_hr_8bit = atmosphere_slm_screens_hr_wrapped.astype(
    np.uint8)

# %% Save results

#
filename = (
    f"{n_opd_screens}_atmosphere_screens_piston_1275e-1_slm_units"
    f"_unwrapped_float64_{n_pixels_in_slm_pupil}"
    f"_pixels_{int(r0 * 100)}cm_r0_seed_{seed}.npy"
)
np.save(dirc_data / filename, atmosphere_slm_screens_hr_slm_units)

filename = (
    f"{n_opd_screens}_atmosphere_screens_piston_1275e-1_slm_units"
    f"_no_wraping_required_float64_{n_pixels_in_slm_pupil}"
    f"_pixels_{int(r0 * 100)}cm_r0_seed_{seed}.npy"
)
np.save(dirc_data / filename, atmosphere_opd_screens_hr_no_wraping_required)

# %% Plots

"""
plt.figure()
plt.plot(atmosphere_slm_screens_hr_slm_units.mean(axis=(1, 2)))

deltas = (
    atmosphere_opd_screens_hr.max(axis=(1, 2))
    - atmosphere_opd_screens_hr.min(axis=(1, 2))
)

plt.figure()
plt.plot(atmosphere_opd_screens_hr_piston_corrected.mean(axis=(1, 2))*1e9)
plt.title("Mean of OPD\nturbulent phase screens without piston")
plt.xlabel("# OPD phase screen")
plt.ylabel("mean [m]")
# plt.ylim(atmosphere_opd_screens_hr_piston_corrected.mean(axis=(1, 2)).min(
# ), atmosphere_opd_screens_hr_piston_corrected.mean(axis=(1, 2)).max()*1e9)

plt.figure()
plt.plot(deltas, label="deltas")
plt.plot(WAVELENGTH * np.ones(deltas.shape), color="r", label="wavelength")
plt.xlabel("# opd screen")
plt.ylabel("OPD [m]")
plt.title("To wrap or not to wrap?")
plt.legend()

plt.figure()
plt.imshow(atmosphere_opd_screens_hr_piston_corrected[0, :, :])
plt.title("raw")

plt.figure()
plt.imshow(atmosphere_slm_screens_hr_8bit[0, :, :])
plt.title("slm shaped")
"""

# %% Debug phase encoding

"""
test_phase_2pi = np.arange(256, dtype=float) * 2*np.pi/255.
test_phase_4pi = np.arange(256, dtype=float) * 4*np.pi/255.

plt.figure()
plt.plot(test_phase_2pi, label="test_phase_2pi")
plt.plot(test_phase_4pi, label="test_phase_4pi")
plt.xlabel("spatial coordinate")
plt.ylabel("phase")
plt.title("test phases")
plt.legend()

test_phase_2pi_mod_255 = np.mod(test_phase_2pi*255./(2*np.pi), 255)
test_phase_2pi_mod_256 = np.mod(test_phase_2pi*255./(2*np.pi), 256)

test_phase_4pi_mod_255 = np.mod(test_phase_4pi*255./(2*np.pi), 255)
test_phase_4pi_mod_256 = np.mod(test_phase_4pi*255./(2*np.pi), 256)

plt.figure()
plt.plot(test_phase_2pi_mod_255, "+", label="test_phase_2pi_mod_255")
plt.plot(test_phase_2pi_mod_256, label="test_phase_2pi_mod_256")
plt.plot(test_phase_4pi_mod_255, "+", label="test_phase_4pi_mod_255")
plt.plot(test_phase_4pi_mod_256, label="test_phase_4pi_mod_256")
plt.title("test phases modulo ?")
plt.legend()

test_phase_2pi_mod_255_8bit = test_phase_2pi_mod_255.astype(np.uint8)
test_phase_2pi_mod_256_8bit = test_phase_2pi_mod_256.astype(np.uint8)

test_phase_4pi_mod_255_8bit = test_phase_4pi_mod_255.astype(np.uint8)
test_phase_4pi_mod_256_8bit = test_phase_4pi_mod_256.astype(np.uint8)

plt.figure()
plt.plot(test_phase_2pi_mod_255_8bit, "+", label="test_phase_2pi_mod_255_8bit")
plt.plot(test_phase_2pi_mod_256_8bit, "*", label="test_phase_2pi_mod_256_8bit")
plt.plot(test_phase_4pi_mod_255_8bit, "+", label="test_phase_4pi_mod_255_8bit")
plt.plot(test_phase_4pi_mod_256_8bit, "*", label="test_phase_4pi_mod_256_8bit")
plt.title("test phases 8 bit")
plt.legend()
"""

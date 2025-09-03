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

# %% Plot OPD deltas vs wavelength

deltas = (
    atmosphere_opd_screens_hr.max(axis=(1, 2))
    - atmosphere_opd_screens_hr.min(axis=(1, 2))
)

plt.figure()
plt.plot(deltas, label="deltas")
plt.plot(WAVELENGTH * np.ones(deltas.shape), color="r", label="wavelength")
plt.xlabel("# opd screen")
plt.ylabel("OPD [m]")
plt.title("To wrap or not to wrap?")
plt.legend()

# %% Save results

filename = (
    f"{n_opd_screens}_atmosphere_opd_screens_{n_pixels_in_slm_pupil}"
    f"_pixels_{int(r0 * 100)}_r0_seed_{seed}.npy"
)

np.save(dirc_data / filename, atmosphere_opd_screens_hr)

# %% Convert to SLM phase map

# set piston at half the wavelength (slm dynamic range in meter)

atmosphere_opd_screens_hr_piston_corrected = atmosphere_opd_screens_hr - \
    atmosphere_opd_screens_hr.mean(axis=(1, 2), keepdims=True) + WAVELENGTH/2.

plt.figure()
plt.plot(atmosphere_opd_screens_hr_piston_corrected.mean(axis=(1, 2)))
plt.title("Mean of OPD\nturbulent phase screens without piston")
plt.xlabel("# OPD phase screen")
plt.ylabel("mean [m]")
plt.ylim(atmosphere_opd_screens_hr_piston_corrected.mean(axis=(1, 2)).min(
), atmosphere_opd_screens_hr_piston_corrected.mean(axis=(1, 2)).max())

# scale between 0 and 255

atmosphere_slm_screens_hr_wrapped =\
    atmosphere_opd_screens_hr_piston_corrected *\
    255./WAVELENGTH

atmosphere_slm_screens_hr_wrapped = np.mod(
    atmosphere_slm_screens_hr_wrapped, 256.)

# encode on 8-bit

atmosphere_slm_screens_hr_8bit = atmosphere_slm_screens_hr_wrapped.astype(
    np.uint8)

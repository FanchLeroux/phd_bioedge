# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:01:52 2024

@author: fleroux
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt

import tqdm

from OOPAO.Telescope import Telescope
from OOPAO.Source import Source
from OOPAO.Atmosphere import Atmosphere

from OOPAO.tools.interpolateGeometricalTransformation import interpolate_cube

# %%

dirc_data = pathlib.Path(__file__).parent.parent.parent.parent.parent /\
    "data" / "data_banc_proto_bioedge" / "turbulence"

# %% Parameters

n_subaperture = 20
n_pixels_in_slm_pupil = 1152

r0 = 0.15  # Fried Parameter [m]
L0 = 25  # Outer Scale [m]
fractionalR0 = [0.45, 0.1, 0.1, 0.25, 0.1]  # Cn2 Profile
windSpeed = [10, 12, 11, 15, 20]  # Wind Speed in [m]
# Wind Direction in [degrees]
windDirection = [0, 72, 144, 216, 288]
# Altitude Layers in [m]
altitude = [0, 1000, 5000, 10000, 12000]

n_phase_screens = 10

seed = 0

# %% Telescope

# create the Telescope object
tel = Telescope(  # resolution of the telescope in [pix]
    resolution=8*n_subaperture,
    # diameter in [m]
    diameter=8,
    # Sampling time in [s] of the AO loop
    samplingTime=1/1000,
    # Central obstruction in [%] of a diameter
    centralObstruction=0.,
    # Flag to display optical path
    display_optical_path=False,
    # field of view in [arcsec]. If set to 0 (default) this speeds up the
    # computation of the phase screens but is uncompatible with
    # off-axis targets
    fov=10)


tel_hr = Telescope(  # resolution of the telescope in [pix]
    resolution=n_pixels_in_slm_pupil,
    # diameter in [m]
    diameter=8,
    # Sampling time in [s] of the AO loop
    samplingTime=1/1000,
    # Central obstruction in [%] of a diameter
    centralObstruction=0.,
    # Flag to display optical path
    display_optical_path=False,
    # field of view in [arcsec]. If set to 0 (default) this speeds up the
    # computation of the phase screens but is uncompatible with
    # off-axis targets
    fov=10)

# %% Natural Guide Star

# create the Natural Guide Star object
ngs = Source(optBand='I',  # Optical band (see photometry.py)
             magnitude=8,  # Source Magnitude
             coordinates=[0, 0])  # Source coordinated [arcsec,deg]

# combine the NGS to the telescope using '*'
ngs*tel
ngs*tel_hr

# %% Atmosphere

# create the Atmosphere object
atm = Atmosphere(telescope=tel,  # Telescope
                 r0=r0,  # Fried Parameter [m]
                 L0=L0,  # Outer Scale [m]
                 fractionalR0=fractionalR0,  # Cn2 Profile
                 windSpeed=windSpeed,  # Wind Speed in [m]
                 # Wind Direction in [degrees]
                 windDirection=windDirection,
                 # Altitude Layers in [m]
                 altitude=altitude)

atm.initializeAtmosphere(tel)
tel+atm

# %% Compute atmosphere phase screens

atmosphere_phase_screens = np.zeros(
    (n_phase_screens,) + atm.OPD_no_pupil.shape, dtype=float)
atmosphere_phase_screens.fill(np.nan)

for k in tqdm.tqdm(range(n_phase_screens)):

    atmosphere_phase_screens[k, :, :] = atm.OPD_no_pupil
    atm.update()

# %%

atmosphere_phase_screens_hr = interpolate_cube(
    atmosphere_phase_screens, tel.pixelSize, tel_hr.pixelSize,
    tel_hr.resolution)

# %%

deltas = atmosphere_phase_screens_hr.max(
    axis=(1, 2)) - atmosphere_phase_screens_hr.min(axis=(1, 2))

wavelength = 675e-9 * np.ones(deltas.shape)

plt.figure()
plt.plot(deltas, label="deltas")
plt.plot(wavelength, color="r", label="wavelength")
plt.xlabel("# phase screen")
plt.ylabel("OPD [m]")
plt.title("to wrap or not to wrap ?")
plt.legend()

# %% Save results

filename = (
    f"{n_phase_screens}_atmosphere_phase_screens_{n_pixels_in_slm_pupil}"
    f"_pixels_{int(r0 * 100)}_r0_seed_{seed}.npy"
)
np.save(dirc_data / filename, atmosphere_phase_screens_hr)

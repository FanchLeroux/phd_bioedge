# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:01:52 2024

@author: fleroux
"""

import pathlib

import numpy as np

from OOPAO.Telescope import Telescope
from OOPAO.Source import Source
from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis

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


tel_HR = Telescope(  # resolution of the telescope in [pix]
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

# %% Deformable Mirror

# specifying a given number of actuators along the diameter:
nAct = n_subaperture+1

dm = DeformableMirror(telescope=tel,  # Telescope
                      # number of subaperture of the system considered
                      # (by default the DM has n_subaperture + 1 actuators to
                      # be in a Fried Geometry)
                      nSubap=nAct-1,
                      # Mechanical Coupling for the influence functions
                      mechCoupling=0.35,
                      # coordinates in [m].
                      # Should be input as an array of size [n_actuators, 2]
                      coordinates=None,
                      # inter actuator distance. Only used to compute the
                      # influence function coupling. The default is based on
                      # the n_subaperture value.
                      pitch=tel.D/nAct, floating_precision=32)

dm_HR = DeformableMirror(telescope=tel_HR,  # Telescope
                         # number of subaperture of the system considered
                         # (by default the DM has n_subaperture + 1 actuators
                         # to be in a Fried Geometry)
                         nSubap=nAct-1,
                         # Mechanical Coupling for the influence functions
                         mechCoupling=0.35,
                         # coordinates in [m]. Should be input as an array of
                         # size [n_actuators, 2]
                         coordinates=None,
                         # inter actuator distance. Only used to compute the
                         # influence function coupling. The default is based on
                         # the n_subaperture value.
                         pitch=tel.D/nAct, floating_precision=32)

# %%  Modal Basis

# use the default definition of the KL modes with forced Tip and Tilt. For more
# complex KL modes, consider the use of the compute_KL_basis function.
# matrix to apply modes on the DM
M2C_KL = compute_KL_basis(tel, atm, dm, lim=1e-5)

# %% Compute high res KL modes

# dm_HR.coefs = M2C_KL

# # propagate through the DM
# ngs*tel_HR*dm_HR
# KL_modes = tel_HR.OPD

# # std normalization
# KL_modes = KL_modes / np.std(KL_modes, axis=(0, 1))

# # set the mean value around pi
# KL_modes = (KL_modes + np.pi)*tel_HR.pupil[:, :, np.newaxis]

# # scale from 0 to 255 for a 2pi phase shift
# KL_modes = KL_modes * 255/(2*np.pi)

# # convert to 8-bit integers
# KL_modes = KL_modes.astype(np.uint8)

# %%

atmosphere = np.zeros(tel_HR.pupil.shape)

np.save(dirc_data / "atmosphere.npy", atmosphere)

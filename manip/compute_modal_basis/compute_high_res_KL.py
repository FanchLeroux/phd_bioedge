# %% Imports

import pathlib

import numpy as np
import matplotlib.pyplot as plt

from OOPAO.Telescope import Telescope
from OOPAO.Source import Source
from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
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
tel+atm

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

# %%  Modal Basis

# use the default definition of the KL modes with forced Tip and Tilt. For more
# complex KL modes, consider the use of the compute_KL_basis function.
# matrix to apply modes on the DM
M2C_KL = compute_KL_basis(tel, atm, dm, lim=1e-3)

# %% Compute KL modes

tel.resetOPD()

dm.coefs = M2C_KL

# propagate through the DM
ngs*tel*dm
KL_modes = tel.OPD

# %% Interpolation to high resolution

KL_modes_hr = interpolate_cube(
    KL_modes,
    tel.pixelSize,
    tel_hr.pixelSize,
    tel_hr.resolution
)

# %%


# %%

np.save(dirc_data / "compute_high_res_KL_output.npy", KL_modes)

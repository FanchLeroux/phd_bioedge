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

n_opd = 10
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

# reshape data cube as [n_images, n_pixels, n_pixels]
KL_modes = np.transpose(KL_modes, (2, 0, 1))

# %% Interpolation to high resolution

KL_modes_hr = interpolate_cube(
    KL_modes,
    tel.pixelSize,
    tel_hr.pixelSize,
    tel_hr.resolution
)

# %% Convert to SLM phase map

# scale to slm units
KL_modes_slm_units = KL_modes_hr * 255./WAVELENGTH

# set 0 mean
KL_modes_slm_units = KL_modes_slm_units - \
    KL_modes_slm_units.mean(axis=(1, 2), keepdims=True)

# set unitary variance
KL_modes_slm_units = KL_modes_slm_units / \
    KL_modes_slm_units.std(axis=(1, 2), keepdims=True)

# get a version with no wraping required
KL_modes_slm_units_no_wraping_required = 127.5 *\
    KL_modes_slm_units /\
    np.array([KL_modes_slm_units.max(),
              -KL_modes_slm_units.min()]).max()\
    + 127.5

# encode on 8-bit
KL_modes_slm_units_no_wraping_required_8bit = np.round(
    KL_modes_slm_units_no_wraping_required)
KL_modes_slm_units_no_wraping_required_8bit =\
    KL_modes_slm_units_no_wraping_required_8bit.astype(np.uint8)

# set piston at half the wavelength (slm dynamic range in meter)
KL_modes_slm_units_piston_corrected = KL_modes_slm_units + 127.5

# wraping
KL_modes_slm_units_wrapped = np.mod(
    KL_modes_slm_units_piston_corrected, 256)

# encode on 8-bit
KL_modes_slm_units_8bit = np.round(KL_modes_slm_units_wrapped)
KL_modes_slm_units_8bit =\
    KL_modes_slm_units_wrapped.astype(np.uint8)

# %% Save results

filename = "KL_modes_slm_units_no_wraping_required.npy"
np.save(dirc_data / filename, KL_modes_slm_units_no_wraping_required)

filename = "KL_modes_slm_units_piston_corrected.npy"
np.save(dirc_data / filename, KL_modes_slm_units_piston_corrected)

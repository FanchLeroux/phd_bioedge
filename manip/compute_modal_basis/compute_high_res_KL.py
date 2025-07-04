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

from OOPAO.tools.displayTools import displayMap

#%%

dirc_data = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent / "data"

#%%

n_subaperture = 20
n_pixels_in_slm_pupil = 600

#%% -----------------------     TELESCOPE   ----------------------------------


# create the Telescope object
tel = Telescope(resolution           = 8*n_subaperture,                          # resolution of the telescope in [pix]
                diameter             = 8,                                        # diameter in [m]        
                samplingTime         = 1/1000,                                   # Sampling time in [s] of the AO loop
                centralObstruction   = 0.,                                      # Central obstruction in [%] of a diameter 
                display_optical_path = False,                                    # Flag to display optical path
                fov                  = 10 )                                     # field of view in [arcsec]. If set to 0 (default) this speeds up the computation of the phase screens but is uncompatible with off-axis targets


tel_HR = Telescope(resolution        = n_pixels_in_slm_pupil,                         # resolution of the telescope in [pix]
                diameter             = 8,                                        # diameter in [m]        
                samplingTime         = 1/1000,                                   # Sampling time in [s] of the AO loop
                centralObstruction   = 0.,                                      # Central obstruction in [%] of a diameter 
                display_optical_path = False,                                    # Flag to display optical path
                fov                  = 10 )                                      # field of view in [arcsec]. If set to 0 (default) this speeds up the computation of the phase screens but is uncompatible with off-axis targets



#%% -----------------------     NGS   ----------------------------------

# create the Natural Guide Star object
ngs = Source(optBand     = 'I',           # Optical band (see photometry.py)
             magnitude   = 8,             # Source Magnitude
             coordinates = [0,0])         # Source coordinated [arcsec,deg]

# combine the NGS to the telescope using '*'
ngs*tel


#%% -----------------------     ATMOSPHERE   ----------------------------------

           
# create the Atmosphere object
atm = Atmosphere(telescope     = tel,                               # Telescope                              
                 r0            = 0.15,                              # Fried Parameter [m]
                 L0            = 25,                                # Outer Scale [m]
                 fractionalR0  = [0.45 ,0.1  ,0.1  ,0.25  ,0.1   ], # Cn2 Profile
                 windSpeed     = [10   ,12   ,11   ,15    ,20    ], # Wind Speed in [m]
                 windDirection = [0    ,72   ,144  ,216   ,288   ], # Wind Direction in [degrees]
                 altitude      = [0    ,1000 ,5000 ,10000 ,12000 ]) # Altitude Layers in [m]




#%% -----------------------     DEFORMABLE MIRROR   ----------------------------------

# specifying a given number of actuators along the diameter: 
nAct = n_subaperture+1
    
dm = DeformableMirror(telescope  = tel,                        # Telescope
                    nSubap       = nAct-1,                     # number of subaperture of the system considered (by default the DM has n_subaperture + 1 actuators to be in a Fried Geometry)
                    mechCoupling = 0.35,                       # Mechanical Coupling for the influence functions
                    coordinates  = None,                       # coordinates in [m]. Should be input as an array of size [n_actuators, 2] 
                    pitch        = tel.D/nAct,floating_precision=32)                 # inter actuator distance. Only used to compute the influence function coupling. The default is based on the n_subaperture value. 
    
dm_HR = DeformableMirror(telescope  = tel_HR,                        # Telescope
                    nSubap       = nAct-1,                     # number of subaperture of the system considered (by default the DM has n_subaperture + 1 actuators to be in a Fried Geometry)
                    mechCoupling = 0.35,                       # Mechanical Coupling for the influence functions
                    coordinates  = None,                       # coordinates in [m]. Should be input as an array of size [n_actuators, 2] 
                    pitch        = tel.D/nAct,floating_precision=32)                 # inter actuator distance. Only used to compute the influence function coupling. The default is based on the n_subaperture value. 
    


#%% -----------------------     Modal Basis - KL Basis  ----------------------------------

# use the default definition of the KL modes with forced Tip and Tilt. For more complex KL modes, consider the use of the compute_KL_basis function. 
M2C_KL = compute_KL_basis(tel, atm, dm, lim = 1e-3) # matrix to apply modes on the DM

#%% compute high res KL modes

dm_HR.coefs = M2C_KL

# propagate through the DM
ngs*tel_HR*dm_HR
KL_modes = tel_HR.OPD

# std normalization
KL_modes = KL_modes / np.std(KL_modes, axis=(0,1))

# set the mean value around pi
KL_modes = KL_modes + np.pi

# scale from 0 to 255 for a 2pi phase shift
KL_modes = KL_modes * 255/(2*np.pi)

# convert to 8-bit integers
KL_modes = KL_modes.astype(np.uint8)

#%%

np.save(dirc_data / "slm" / "modal_basis" / "KL_modes" / ("KL_modes_" + 
                                                               str(n_pixels_in_slm_pupil) + 
                                                               "_pixels_in_slm_pupil_" +
                                                               str(n_subaperture) +
                                                               "_subapertures.npy"), KL_modes)

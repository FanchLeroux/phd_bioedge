# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:10:53 2024

@author: fleroux
"""

import numpy as np

wavelength_lam = 675e-9 # [m]
wavelength_ghost = 770e-9 # [m]

mfd_lam = [3.6e-6, 5.3e-6] # [m]
mfd_ghost = 5e-6 # [m]

half_divergence_angle_lam = [wavelength_lam/(np.pi*(mfd_lam[0]/2)), wavelength_lam/(np.pi*(mfd_lam[1]/2))] # [rad]
half_divergence_angle_ghost = wavelength_ghost/(np.pi*(mfd_ghost/2)) # [rad]

half_divergence_angle_lam_deg = np.rad2deg(np.array(half_divergence_angle_lam)) # [°]
half_divergence_angle_ghost_deg = np.rad2deg(half_divergence_angle_ghost) # [°]


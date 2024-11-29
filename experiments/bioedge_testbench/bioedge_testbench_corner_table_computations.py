# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:10:53 2024

@author: fleroux
"""

import numpy as np

diameter_L1 = 25e-3 # [m]
focal_length_L1 = 100e-3 # [m]

numerical_aperture_sLED = 0.13 # sin of the half aperture angle

alpha_LED = np.arcsin(numerical_aperture_sLED)
alpha_L1 = np.arctan(diameter_L1/(2*focal_length_L1))

collimated_beam_diameter = 2*numerical_aperture_sLED*focal_length_L1

eta_L1 = (1.-np.cos(alpha_L1))/(1.-np.cos(alpha_LED))
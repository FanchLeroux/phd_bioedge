# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 17:53:13 2025

@author: fleroux
"""

wavelength = 675e-9 # [m]

focal_length_L5 = 200e-3

pupil_diameter = focal_length_L5 / 62.0 # [m]

n_subapertures = 64.0

pupil_opd = 360e-6 # [m]

talbot_length = 8.0 * pupil_diameter**2 / (wavelength * n_subapertures**2)

print(talbot_length / pupil_opd)

print(4*pupil_diameter*1e3)
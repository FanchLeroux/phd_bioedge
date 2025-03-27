# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 11:47:26 2025

@author: fleroux
"""

# source : https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=1500&pn=S1FC675
# tutorial laser divergence : https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=14204

import numpy as np

wavelength = 675e-9 # [m] wavelength

mfd = np.array([3.6e-6, 5.3e-6])  # [m] mode field diameter

zr = np.pi/wavelength * (mfd/2)**2 # [m] Rayleigh range

propagation_distance = 75e-3 # [m] collimating lens focal length

collimated_beam_diameter = mfd * (1 + (propagation_distance/zr)**2)**0.5

print(collimated_beam_diameter*1e3)
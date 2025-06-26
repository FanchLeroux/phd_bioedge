# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 16:32:45 2025

@author: lgs
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt

dirc_lut = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent / "data" / "slm" / "LUT"

lut_linear = np.loadtxt(dirc_lut / "12bit_linear.lut")

lut_ref = np.loadtxt(dirc_lut / "slm5758_at675.LUT")

lut_new = np.loadtxt(dirc_lut / "utc_2026-06-26_12-12-56_slm0_at675.LUT")

plt.figure()
plt.plot(lut_linear[:,1], label='lut_linear')
plt.plot(lut_ref[:,1], label='lut_ref')
plt.plot(lut_new[:,1], label='lut_new')
plt.legend()
plt.xlabel("Phase level")
plt.ylabel("LUT value")
plt.title("SLM LUT comparison")




# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 11:30:16 2025

@author: fleroux
"""

import pathlib

import tifffile as tif

import matplotlib.pyplot as plt

path_data = pathlib.Path(__file__).parent.parent.parent / "phd_bioedge_data" / "thorcam"

filename = "20250613_gbioedge_001.tif"

data = tif.imread(path_data / filename).astype(float)

plt.figure()
plt.imshow(data, cmap="gist_gray")
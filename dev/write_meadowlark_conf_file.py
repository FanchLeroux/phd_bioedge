# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 11:54:41 2025

@author: fleroux
"""

#%%

import pathlib

#%%

dirc = pathlib.Path(__file__).parent

filename = "plico_dm_server.conf"

with open(dirc / filename, 'w') as f:
    f.write("[PaulsdeviceMaedowlarkSLM]\n\
name= Meadowlark SLM 1920\n\
model= meadowlarkSLM\n\
blink_dir_root= C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\n\
lut_filename= C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\LUT Files\slm5758_at_589_september_2021_ok.lut\n\
wfc_filename= C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\WFC Files\WFC_flat_zero.bmp\n\
wl_calibration= 589e-9\n\
default_flat_tag=zero_slm\n\n\
[server1]\n\
name= Deformable Mirror Server 1\n\
log_level= info\n\
mirror= PaulsdeviceMaedowlarkSLM\n\
host= localhost\n\
port= 7000")

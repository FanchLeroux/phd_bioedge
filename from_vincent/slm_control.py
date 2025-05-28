# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 16:20:54 2021@author: UCSCL
"""

import os
import numpy
from ctypes import *
from scipy import misc
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from hcipy import *
import zmq
import matplotlib.pyplot as plt
import time

try:
    slm_lib.Delete_SDK()
except:
    pass

cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\Blink_C_wrapper")
slm_lib = CDLL("Blink_C_wrapper")

# Open the image generation library
cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\ImageGen")
image_lib = CDLL("ImageGen")

# Basic parameters for calling Create_SDK
bit_depth = c_uint(12)
num_boards_found = c_uint(0)
constructed_okay = c_bool(0)
is_nematic_type = c_bool(1)
RAM_write_enable = c_bool(1)
use_GPU = c_bool(1)
max_transients = c_uint(20) #20 was good
board_number = c_uint(1)
wait_For_Trigger = c_uint(0)
output_Pulse = c_uint(0)
timeout_ms = c_uint(5000)# Call the Create_SDK constructor
# Returns a handle that"s passed to subsequent SDK calls
slm_lib.Create_SDK(bit_depth, byref(num_boards_found), byref(constructed_okay), is_nematic_type, RAM_write_enable, use_GPU, max_transients, 0)
if constructed_okay == -1:
    print ("Blink SDK was not successfully constructed");
    # Python ctypes assumes the return value is always int
    # We need to tell it the return type by setting restype
    slm_lib.Get_last_error_message.restype = c_char_p
    print (slm_lib.Get_last_error_message());    # Always call Delete_SDK before exiting
    slm_lib.Delete_SDK()
else:
    time.sleep(2)
    print ("Blink SDK was successfully constructed");
    print ("Found %s SLM controller(s)" % num_boards_found.value);
    height = c_uint(slm_lib.Get_image_height(board_number));
    width = c_uint(slm_lib.Get_image_width(board_number));
    center_x = c_uint(width.value//2);
    center_y = c_uint(height.value//2);    
    slm_lib.Load_LUT_file(board_number, "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6164_at633_012025");    
    print ("Found %s SLM controller(s), %dx%d" % (num_boards_found.value,
           height.value, width.value));    #USE THEIR LIBRARY TO MAKE SHAPES SO AS TO NOT MESS UP THE TYPES AT FIRST
    #GENERATE SOLID with 0 as the values to get a flat shape
    #make Image One holding zeros
    image2 = numpy.empty([width.value*height.value], numpy.uint8, "C");
    image_lib.Generate_Solid(image2.ctypes.data_as(POINTER(c_ubyte)), width.value, height.value, 0)    # Loop between our phase gradient images
    WFC=image2
    slm_lib.Write_image(board_number, WFC.ctypes.data_as(POINTER(c_ubyte)), height.value*width.value, wait_For_Trigger, output_Pulse, timeout_ms)
    slm_lib.ImageWriteComplete(board_number, timeout_ms)
    
    def applyPhase(phase,slm_lib=slm_lib):
        #LETS WRITE ALL ZEROS TO THE BOARD
        #phase_slm=phase.reshape((1,width.value*height.value),order="C").copy()
        ImageOne = numpy.empty([width.value*height.value], numpy.uint8, "C");
        # turb=np.load("D:\Program Files\Meadowlark Optics\Blink OverDrive Plus\SDK\SPIE data\Turb\AO_turb_r0_0.022831_L0_4.56621_unmod_1khz_offcenter.npy") #reads in as radians; needs to be edited to get correct folder
        # turb=np.mod(turb,6*np.pi)
        # convert_rad=256/(6*np.pi)
        # phase=turb[2,:,:]*convert_rad
        ImageOne=np.mod(phase,256).astype(np.uint8).flatten("F")    
        print(ImageOne.dtype)
        print(ImageOne.shape)
        slm_lib.Write_image(board_number, ImageOne.ctypes.data_as(POINTER(c_ubyte)), height.value*width.value, wait_For_Trigger, output_Pulse, timeout_ms)
        slm_lib.ImageWriteComplete(board_number, timeout_ms);
    
    def disconnect(socket):
        socket.close()
    
    if __name__ == "__main__":
        
        import ctypes as ct
        import sys 
        
        try:
            port = "5558"
            context = zmq.Context()
            socket = context.socket(zmq.REP)
            socket.bind("tcp://*:%s" % port)
            print("Started...")
            while True:
                data = socket.recv()
                phase = np.frombuffer(data, dtype=np.float64)
                phase=phase.reshape(height.value, width.value).transpose()
                print(phase.shape)
                print("Data received...")
                applyPhase(phase)
              #   applyPhase(np.zeros((width.value, height.value)))            data = "success"
                socket.send_string(data)
        except Exception as e:
            print(e)
        finally:
            print("Sending zeros")
            #send zeros
            #applyPhase(phase*0)
            print("cleaning up")
            slm_lib.Delete_SDK()
            socket.close()
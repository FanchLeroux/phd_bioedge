# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:50:22 2023

@author: cheritier
"""

import pylablib as pll
from   pylablib.devices import DCAM
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import datetime
import os
from maoppy.psffit import psffit
from maoppy.instrument import papyrus
from maoppy.psfmodel import Turbulent

from numpy.fft import fft2, ifft2, ifftshift, fftshift

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from OOPAO.tools.tools import emptyClass

cam = DCAM.DCAMCamera()
cam.ID = 518
# pll.par['devices/dlls/dcamapi']='path/to/dll/'

#%%

# ---------------------------------------- DEFINITION OF FUNCTIONS -----------------------------------------
def makeSquareAxes(ax):
    """Make an axes square in screen units.

    Should be called after plotting.
    """
    ax.set_aspect(1 / ax.get_data_ratio())

def orca_gui(list_fig,plt_obj= None, type_fig = None,fig_number = 20,\
             n_subplot = None,list_ratio = None, list_title = None,\
                 list_lim = None,list_label = None, list_display_axis = None,\
                     list_buffer =  None,s=16,cam = None,Pmodel= None):
    
    n_im = len(list_fig)
    if n_subplot is None:
        n_sp = int(np.ceil(np.sqrt(n_im)))
        n_sp_y = n_sp + 1
    else:
        n_sp   = n_subplot[0]
        n_sp_y = n_subplot[1] +3
    if plt_obj is None:
        if list_ratio is None:
            gs = gridspec.GridSpec(n_sp_y,n_sp, height_ratios=np.ones(n_sp_y), width_ratios=np.ones(n_sp), hspace=0.25, wspace=0.25)
        else:
            gs = gridspec.GridSpec(n_sp_y,n_sp, height_ratios=list_ratio[0], width_ratios=list_ratio[1], hspace=0.25, wspace=0.25)
        
        plt_obj = emptyClass()
        plt_obj.isOnSky = True
        plt_obj.cam = cam
        setattr(plt_obj,'gs',gs)
        plt_obj.list_label = list_label
        plt_obj.list_buffer = list_buffer        
        plt_obj.list_lim = list_lim

        plt_obj.keep_going = True
        f = plt.figure(fig_number,figsize = [n_sp*4,n_sp_y*2],facecolor=[0,0.1,0.25], edgecolor = None)
        plt_obj.filename = plt.figtext(0.2,0.2,'')

        COLOR = 'white'
        mpl.rcParams['text.color'] = COLOR
        mpl.rcParams['axes.labelcolor'] = COLOR
        mpl.rcParams['xtick.color'] = COLOR
        mpl.rcParams['ytick.color'] = COLOR
        line_comm = ['ACQUIRE',
                     'Data Exp. Time: '+
                                          str(np.round(1000*cam.exp_time,2)) +
                                          ' ms -- '+str(cam.n_frames)+' Frames'
                                          '\n Live Exp. Time: '+
                                          str(np.round(1000*cam.get_exposure(),2)),
                     'ESTIMATE SEEING/SR']
        col_comm = ['g','k','g']
        
        for i in range(3):
            setattr(plt_obj,'ax_0_'+str(i+1), plt.subplot(gs[n_sp_y-2,i]))            
            sp_tmp =getattr(plt_obj,'ax_0_'+str(i+1)) 
            
            annot = sp_tmp.annotate(line_comm[i],color ='w',  xy=(0.5,0.5), ha='center',fontsize=20,bbox=dict(boxstyle="round",edgecolor='w',pad=0.4, fc=col_comm[i]),annotation_clip=False)
            setattr(plt_obj,'annotation_'+str(i+1), annot)
        
            plt.axis('off')
            
        line_comm = ['RECENTER','SET ROI','SET EXPOSURE']
        col_comm = ['b','b','b']
        
        for i in range(3):
            setattr(plt_obj,'ax_cam_'+str(i+1), plt.subplot(gs[n_sp_y-4,i]))            
            sp_tmp =getattr(plt_obj,'ax_cam_'+str(i+1)) 
            
            annot = sp_tmp.annotate(line_comm[i],color ='w', fontsize=20,  xy=(0.5,0.5), ha='center',bbox=dict(boxstyle="round",pad=0.4, fc=col_comm[i]),annotation_clip=False)
            setattr(plt_obj,'annot_'+str(i+1), annot)
        
            plt.axis('off')   
            
            
        COLOR = 'white'
        mpl.rcParams['text.color'] = COLOR
        mpl.rcParams['axes.labelcolor'] = COLOR
        mpl.rcParams['xtick.color'] = COLOR
        mpl.rcParams['ytick.color'] = COLOR
        line_comm = ['Next ID:'+str(cam.ID+1),'STOP','SR']
        col_comm = ['k','r','k']

        for i in range(3):
            setattr(plt_obj,'ax_00_'+str(i+1), plt.subplot(gs[n_sp_y-1,i]))            
            sp_tmp =getattr(plt_obj,'ax_00_'+str(i+1)) 
            
            annot = sp_tmp.annotate(line_comm[i],color ='w', fontsize=20, xy=(0.5,0.5),va='bottom' ,ha='center',bbox=dict(boxstyle="round", pad=0.4,fc=col_comm[i]))
            setattr(plt_obj,'last_'+str(i+1), annot)
        
            plt.axis('off')
        

        count = -1
        for i in range(n_sp):
            for j in range(n_sp):
                if count < n_im-1:
                    count+=1

                    # print(count)
                    setattr(plt_obj,'ax_'+str(count), plt.subplot(gs[i+1,j]))
                    sp_tmp =getattr(plt_obj,'ax_'+str(count))            

                    setattr(plt_obj,'type_fig_'+str(count),type_fig[count])
           # IMSHOW
                    if type_fig[count] == 'imshow':
                        data_tmp = list_fig[count]
                        if len(data_tmp)==3:
                            setattr(plt_obj,'im_'+str(count),sp_tmp.imshow(data_tmp[2],extent = [data_tmp[0][0],data_tmp[0][1],data_tmp[1][0],data_tmp[1][1]],cmap='hot'))        
                        else:
                            setattr(plt_obj,'im_'+str(count),sp_tmp.imshow(data_tmp))                                    
                        im_tmp =getattr(plt_obj,'im_'+str(count))
                        plt.colorbar(im_tmp)
      
           # PLOT     
                    if type_fig[count] == 'plot':
                        data_tmp = list_fig[count]
                        if len(data_tmp)==2:
                            line_tmp, = sp_tmp.plot(data_tmp[0],data_tmp[1],'-')      
                        else:
                            line_tmp, = sp_tmp.plot(data_tmp,'-o')         
                        setattr(plt_obj,'im_'+str(count),line_tmp)
                        
                            
           # SCATTER
                    if type_fig[count] == 'scatter':
                        data_tmp = list_fig[count]
                        n = mpl.colors.Normalize(vmin = min(data_tmp[2]), vmax = max(data_tmp[2]))
                        m = mpl.cm.ScalarMappable(norm=n)
                        scatter_tmp = sp_tmp.scatter(data_tmp[0],data_tmp[1],c=m.to_rgba(data_tmp[2]),marker = 'o', s =s)
                        setattr(plt_obj,'im_'+str(count),scatter_tmp)  
                        sp_tmp.set_facecolor([0,0.1,0.25])
                        for spine in sp_tmp.spines.values():
                            spine.set_edgecolor([0,0.1,0.25])
                        makeSquareAxes(plt.gca())
                        plt.colorbar(scatter_tmp)
                    if list_title is not None:
                        plt.title(list_title[count])
                    if list_display_axis is not None:
                        if list_display_axis[count] is None:
                            sp_tmp.set_xticks([])                                
                            sp_tmp.set_yticks([])                                                                
                            sp_tmp.set_xticklabels([])                                
                            sp_tmp.set_yticklabels([])                                
                            
                    if plt_obj.list_label is not None:
                        if plt_obj.list_label[count] is not None:
                            plt.xlabel(plt_obj.list_label[count][0])
                            plt.ylabel(plt_obj.list_label[count][1]) 
        COLOR = 'black'
        mpl.rcParams['text.color'] = COLOR
        mpl.rcParams['axes.labelcolor'] = COLOR
        mpl.rcParams['xtick.color'] = COLOR
        mpl.rcParams['ytick.color'] = COLOR

        def hover(event):
            if event.inaxes == plt_obj.ax_00_2:
                cont, ind = f.contains(event)        
                if cont:
                    plt_obj.keep_going = False
                    plt_obj.last_2.set_text('STOPPED')
                    
            if event.inaxes == plt_obj.ax_cam_2:
                cont, ind = f.contains(event)        
                if cont:
                    if event.button == 1:
                        hs,hf,vs,vf,_,_ = cam.get_roi()   
                        cam.set_roi(hstart = hs+25,\
                                    hend   = hf-25,\
                                    vstart = vs+25,\
                                    vend   = vf-25)
                    if event.button == 3:
                        hs,hf,vs,vf,_,_ = cam.get_roi()   
                        cam.set_roi(hstart = hs-25,\
                                    hend   = hf+25,\
                                    vstart = vs-25,\
                                    vend   = vf+25)
            if event.inaxes == plt_obj.ax_cam_3:
                cont, ind = f.contains(event)        
                if cont:
                    if event.button == 1:
                        cam.set_exposure(cam.get_exposure()+0.01)

                    if event.button == 3:
                        cam.set_exposure(cam.get_exposure()-0.01)
                sp_tmp =getattr(plt_obj,'annotation_'+str(2))
                sp_tmp.set_text('Data Exp. Time: '+
                                                          str(np.round(1000*cam.exp_time,2)) +
                                          ' ms -- '+str(cam.n_frames)+' Frames'
                                                          '\n Live Exp. Time: '+
                                                          str(np.round(1000*cam.get_exposure(),2)))

                    
            if event.inaxes == plt_obj.ax_cam_1:
                cont, ind = f.contains(event)        
                if cont:
                    
                    hs,hf,vs,vf,_,_ = cam.get_roi()             
                    
                    sx = hf-hs
                    sy=  vf-vs
                    
                    hsize = cam.get_detector_size()[0]
                    vsize = cam.get_detector_size()[1]
                    
                    
                    cam.set_roi(hstart = 0,\
                                    hend   = hsize,\
                                    vstart = 0,\
                                    vend   = vsize)

                    im = np.squeeze(cam.grab(1))
                    # Find indices where we have mass
                    mass_x, mass_y = np.where(im>(0.1*im.max()))
                    # mass_x and mass_y are the list of x indices and y indices of mass pixels

                    cent_y = np.average(mass_x)
                    cent_x = np.average(mass_y)
                    
                    # setattr(plt_obj, 'centroid', centroid_im)
                    print(cent_x)
                    print(cent_y)                    
                    hs      = int(cent_x) -sx//2
                    hf      = int(cent_x) +sx//2
                    vs      = int(cent_y) -sy//2
                    vf      = int(cent_y) +sy//2

                    cam.set_roi(hstart = hs,\
                                hend   = hf,\
                                vstart = vs,\
                                vend   = vf)
                    cam.grab(1)
                    
                    
            if event.inaxes == plt_obj.ax_0_1:
                cont, ind = f.contains(event)        
                if cont:
                    plt_obj.annotation_1.set_backgroundcolor('g')
                    plt_obj.annotation_1.set_text('ACQUIRING...')
                    plt_obj.annotation_1.set_fontweight('bold')

                    plt.pause(0.0001)
                    print('Acquiring data for ID'+str(cam.ID+1)+'...')
                    image = np.mean(acquire(cam, cam.n_frames,cam.exp_time),axis=0)
                    # plt_obj.filename.set_text(fname)
                    plt.draw()
                    plt.pause(0.0001)
                    if image.max()>65500:
                        print('SATURATION! ')

                    print('Done!')
                    # plt_obj.annotation_1.set_backgroundcolor('b')
                    
                    plt_obj.annotation_1.set_text('ACQUIRE')
                    plt_obj.annotation_1.set_fontweight('normal')

            if event.inaxes == plt_obj.ax_0_3:
                cont, ind = f.contains(event)        
                if cont:
                    plt_obj.annotation_3.set_backgroundcolor('g')
                    plt_obj.annotation_3.set_text('ESTIMATING...')
                    plt_obj.annotation_3.set_fontweight('bold')

                    
                    plt.pause(0.0001)
                    
                    im_tmp = getattr(plt_obj,'im_'+str(1))
                    image = (im_tmp.get_array()).data
                    
                    npix = image.shape[0] # pixel size of PSF                    
                    if plt_obj.isOnSky:
                        samp = 2.55
                    else:
                        samp = 2.85
                        
                    Pmodel = Turbulent((npix,npix), system=papyrus, samp=samp)
                    occ = 0.3
                    psf_diff = psf_diffraction(npix, samp=samp, occ=occ)
                    out = psffit(image, Pmodel, [0.05,20], fixed=[False,True])


                    flx,bck = out.flux_bck
                    out_im = flx*out.psf+bck
                    psf__bc = (image - bck)/flx

                    SR = np.max(psf__bc)/np.max(psf_diff)
                    
                    plt_obj.last_3.set_text('SR:'+str(np.round(100*SR,2)))


                    cx = int(image.shape[0]//2+out.dxdy[0])
                    cy = int(image.shape[0]//2+out.dxdy[1])
                    
                    data = [np.arange(len(image[cx,:])),image[cx,:]] 
                    setattr(plt_obj,'ax_'+str(2), plt.subplot(gs[1,2]))
                    
                    # plot
                    plt_obj.ax_2.cla()
                    plt_obj.ax_2.plot(image[cx,:],label='X-cut (Data)')
                    plt_obj.ax_2.plot(out_im[cx,:],label='X-cut (Model)')
                    plt_obj.ax_2.plot(image[:,cy],label='Y-cut (Data)')
                    plt_obj.ax_2.plot(out_im[:,cy],label='Y-cut (Model)')
                    plt.legend(labelcolor='k')
                    plt.title('Seeing @500nm Estimated: '+str(np.round(206265*500e-9/(out.x[0]*(635/500)**(-6/5)),2))+'"',fontsize=20)
                    plt_obj.ax_2.set_xlim([np.min(data[0])-0.1*np.abs(np.min(data[0])),np.max(data[0])+0.1*np.abs(np.max(data[0]))])
                    # plt_obj.annotation_3.set_backgroundcolor('r')
                    # plt_obj.annotation_3.set_text('ERROR!')
                    # plt.pause(1)
                    
                    plt_obj.annotation_3.set_backgroundcolor('g')
                    plt_obj.annotation_3.set_text('ESTIMATE SEEING')
                    plt_obj.annotation_3.set_fontweight('normal')

                    
                    
        def shift(event,n_pix=50):
            hs,hf,vs,vf,_,_ = cam.get_roi()
            
            if event.key =='right':
                hs -= n_pix
                hf -= n_pix
            if event.key =='left':
                hs += n_pix
                hf += n_pix
            if event.key =='down':
                vs -= n_pix
                vf -= n_pix
            if event.key =='up':
                vs += n_pix
                vf += n_pix
            cam.set_roi(hstart = hs,\
                        hend   = hf,\
                        vstart = vs,\
                        vend   = vf)
                    
                    
        f.canvas.mpl_connect('button_press_event', hover)   
        f.canvas.mpl_connect('key_press_event', shift)   

        return plt_obj
    
    if plt_obj is not None:
        count = 0
        for i in range(n_sp):
            for j in range(n_sp):
                if count < n_im:
                    data = list_fig[count]
                    if getattr(plt_obj,'type_fig_'+str(count)) == 'imshow':
                        im_tmp =getattr(plt_obj,'im_'+str(count))
                        im_tmp.set_data(data)
                        if plt_obj.list_lim[count] is None:
                            im_tmp.set_clim(vmin=data.min(),vmax=data.max())
                        else:
                            im_tmp.set_clim(vmin=plt_obj.list_lim[count][0],vmax=plt_obj.list_lim[count][1])
                    if getattr(plt_obj,'type_fig_'+str(count)) == 'plot':
                        
                        if len(data)==2:
                            im_tmp =getattr(plt_obj,'im_'+str(count))
                            im_tmp.set_xdata(data[0])
                            im_tmp.set_ydata(data[1])
                            im_tmp.axes.set_xlim([np.min(data[0])-0.1*np.abs(np.min(data[0])),np.max(data[0])+0.1*np.abs(np.max(data[0]))])

                            if plt_obj.list_lim[count] is None:
                                im_tmp.axes.set_ylim([np.min(data[1])-0.1*np.abs(np.min(data[1])),np.max(data[1])+0.1*np.abs(np.max(data[1]))])
                            else:
                                im_tmp.axes.set_ylim([plt_obj.list_lim[count][0],plt_obj.list_lim[count][1]])                            

                        else:
                            im_tmp =getattr(plt_obj,'im_'+str(count))
                            im_tmp.set_ydata(data[0])
                            if plt_obj.list_lim[count] is None:
                                im_tmp.axes.set_ylim([np.min(data[0])-0.1*np.abs(np.min(data[0])),np.max(data[0])+0.1*np.abs(np.max(data[0]))])
                            else:
                                im_tmp.axes.set_ylim([plt_obj.list_lim[count][0],plt_obj.list_lim[count][1]])
                                
                                                
                    if getattr(plt_obj,'type_fig_'+str(count)) == 'scatter':
                        n = mpl.colors.Normalize(vmin = min(data), vmax = max(data))
                        m = mpl.cm.ScalarMappable(norm=n)

                        im_tmp = getattr(plt_obj,'im_'+str(count))

                        im_tmp.set_facecolors(m.to_rgba(data))
                        im_tmp.set_clim(vmin=min(data), vmax=max(data))

                        im_tmp.colorbar.update_normal(m)    
                count+=1
    plt.draw()

def acquire(cam, n_frames,exp_time,save = True) :
    cam.set_exposure(exp_time)
    t = str(datetime.datetime.now())
    image = np.double(cam.grab(n_frames))
    if save:
        hdu = fits.PrimaryHDU(data = image)
        hdr=hdu.header
        tmp = cam.get_acquisition_parameters()
        hdr['NFRAMES']      = (tmp['nframes'],'Size of the data cube')
                
        tmp = cam.get_all_attribute_values()
        hdr['EXP_TIME']     = (tmp['exposure_time'],'Exposure time in s')
        hdr['FPS']          = (tmp['internal_frame_rate'],'Frame rate in Hz')
        hdr['INTERVAL']     = (tmp['internal_frame_interval'],'Delay between two successive acquisitions in s')
        hdr['HPOS']         = (tmp['subarray_hpos'],'X-position of the ROI')
        hdr['YPOS']         = (tmp['subarray_vpos'],'Y-position of the ROI')
        hdr['TIME']         = (t,'Local Time of Acquisition')    
        
        
        cam.ID = cam.ID+1
    
        file_name = 'C:/Users/cheritier/Documents/PAPYRUS/data_july_2023/'+'ORCA_' + str(cam.ID)+"_exp_" + str(np.round(cam.get_exposure()*1000, 3)) + '_nframes_' + str(n_frames) + '.fits'
        # file_name = 'ORCA_' + t[:-7].replace(' ', '_').replace(':', '-') + "_exp" + str(np.round(cam.get_exposure(), 2)) + '_nframes' + str(n_frames) + '.fits'
        while os.path.exists(file_name):
            cam.ID = cam.ID +1
            file_name = 'C:/Users/cheritier/Documents/PAPYRUS/data_july_2023/'+'ORCA_' + str(cam.ID)+"_exp_" + str(np.round(cam.get_exposure()*1000, 3)) + '_nframes_' + str(n_frames) + '.fits'
        hdu.writeto(file_name)
    
    return image


#% FUNCTIONS
def aperture(npix, samp=1, occ=0, center=None):
    if center is None:
        center = npix//2
    xx,yy = (np.mgrid[0:npix,0:npix] - center) # ogrid or mgrid is faster???
    rho = np.sqrt(xx**2 + yy**2)
    aper = rho <= (npix/2/samp)
    if occ>0:
        aper *= rho >= (npix/2/samp*occ)
    return aper

def psf_diffraction(npix, samp=2, occ=0):
    aper = aperture(npix, samp=samp, occ=occ)
    otf = ifftshift(np.abs(ifft2(np.abs(fft2(aper))**2))) / np.sum(aper)
    return fftshift(np.real(ifft2(fftshift(otf))))



# ---------------------------------------- DEFINITION OF FUNCTIONS -----------------------------------------

#%% ================================== START HERE ============================================

plt.close('all')
###### PARAMETERS ######
exposure = 0.2 #0.0033325714157860335 is the minimum

full_frame = True # 

n_frames = 100

hsize = cam.get_detector_size()[0]
vsize = cam.get_detector_size()[1]
cam.set_exposure(0.00000001)

# ROI
n = 250
hstart = hsize//2 - 100
hend   = hsize//2 + 100 
vstart = vsize//2 - 100
vend   = vsize//2 + 100

############

cam.exp_time = exposure
cam.n_frames = n_frames

if full_frame:
    cam.set_roi(hstart = 0,\
                    hend   = hsize,\
                    vstart = 0,\
                    vend   = vsize)
else:
    cam.set_roi(hstart = hstart,\
                hend   = hend,\
                vstart = vstart,\
                vend   = vend)

# create the first image to inialize
image = np.mean(np.asarray(cam.grab(1)),axis=0)

# define the gui object
plt_obj  = orca_gui(list_fig          = [image,image,[[0,0],[0,0]]],\
                   type_fig          = ['imshow','imshow','plot'],\
                   list_title        = ['ORCA - SHORT EXPOSURE','ORCA - LONG EXPOSURE',None],\
                   list_lim          = [None,None,None],\
                   list_label        = [None,None,None],\
                   n_subplot         = [3,1],\
                   list_display_axis = [True,True,True],\
                   list_ratio        = [[0.5,1,1,0.5],[1,1,1]],cam=cam)
    
    
# initialize settings, variables and flags
cam.exp_time = 1e-3     #
cam.n_frames = 100  # acquire cubes of 100 images
cam.ID = 30         # ID for the data saved
show_cam = True     # flag to run the gui
delta = 0 
count = 0

print('=======  Acquisition Started ======')
PSF = []
max_psf = []
image_norma = image.copy()*0

# number of frames for the buffer of the live display
n_buffer = 10
im_buffer = np.zeros((image.shape[0],image.shape[1],n_buffer))

while show_cam is True:
        image_in = np.mean(np.asarray(cam.grab(1)),axis=0)
        # image_in = (np.asarray(im[count,:,:]))

        # image_in = np.mean(np.asarray(im),axis=0)
        if image_in.shape == image_norma.shape:
            im_buffer[:,:,count%n_buffer] = image_in
        else:
            im_buffer = np.zeros((image_in.shape[0],image_in.shape[1],n_buffer))

            
            
        if count>n_buffer:
            image_norma = np.mean(np.asarray(im_buffer),axis=2)
        else:
            image_norma = image.copy()*0



        # if count>0:
        #     image_norma /=count
        
        orca_gui(list_fig   = [(image_in),np.log10(np.abs(image_norma))],
                                plt_obj = plt_obj)
        count+=1
        print('Live...'+str(count))
        if plt_obj.keep_going is False:
            print('=======  Acquisition Stopped ======')
            break
        if image_in.max()>65530:
            if count%2 ==0:
                plt_obj.ax_0.title.set_text('ORCA FRAME - SATURATING')
                plt_obj.ax_0.title.set_fontsize('20')
                plt_obj.ax_0.title.set_color('r')
            else:
                plt_obj.ax_0.title.set_text('ORCA - SHORT EXPOSURE')
                plt_obj.ax_0.title.set_fontsize('15')
                plt_obj.ax_0.title.set_color('w')
        else:
            plt_obj.ax_0.title.set_text('ORCA - SHORT EXPOSURE')
            plt_obj.ax_0.title.set_fontsize('15')
            plt_obj.ax_0.title.set_color('w')
                
        plt.pause(0.001)

                
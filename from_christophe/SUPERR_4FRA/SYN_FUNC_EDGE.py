import numpy as np
import importlib
import ao_cockpit_syn as aou
importlib.reload(aou)
import pdb


def CREATE_SCREENS_FROM_DSP(PSD,df,nscr,nsplit):
    SZ = PSD.shape[0]
    Es = (np.random.uniform(low=0,high=1.0,size=[SZ,SZ,nscr])-0.5)*2.*np.pi
    iEs = 1j*Es
    expiEs = ne.evaluate('exp(iEs)')
    #PSDs = np.fft.fftshift(PSD)
    PSDr = np.reshape(np.repeat(PSD,nscr),[SZ,SZ,nscr])
    expiEsXdsp = expiEs * np.sqrt(PSDr)*df *np.sqrt(2.0)
    SCRs = aou.DO_FT_MULTI_C(expiEsXdsp,nsplit)
    return SCRs.real


def build_spatial_freq(S_Z,df):
    fx=(np.array(range(S_Z))-S_Z//2)*df
    ffx=np.zeros([S_Z,S_Z])
    for k in range(0,S_Z): ffx[k,:] = fx
    ffy=np.transpose(ffx)
    ffr = np.sqrt(ffx**2+ffy**2)
    return ffx,ffy,ffr


def sum_ARR2Dto0D(arr2d,axe=None):
    sum0=np.sum(arr2d,axis=(0,1))
    return sum0

def sum_ARR2Dto1D(arr2d,axe=None):
    if axe is None:
        axe=1
    sum1=np.sum(arr2d,axis=axe)
    return sum1

def conv(A,B):
    SZ=A.shape[0]
    As=np.fft.fftshift(A)
    Bs=np.fft.fftshift(B)
    ftAs=np.fft.fft2(As)
    ftBs=np.fft.fft2(Bs)
    prod = ftAs * ftBs
    Fprod=np.fft.fftshift(np.fft.ifft2(prod))
    return Fprod

def corre(A,B):
    SZ=A.shape[0]
    As=np.fft.fftshift(A)
    Bs=np.fft.fftshift(B)
    ftAs=np.fft.fft2(As)
    ftBs=np.fft.fft2(Bs)
    prod = ftAs * np.conjugate(ftBs)
    Fprod=np.fft.fftshift(np.fft.ifft2(prod))
    return Fprod.real


def conv2(A,B):
    SZ=A.shape[0]
    As=np.fft.fftshift(A)
    Bs=np.fft.fftshift(B)
    ftAs=np.fft.fft2(As)
    ftBs=np.fft.fft2(Bs)
    #prod = ftAs * ftBs
    prod = As*Bs
    Fprod=np.fft.fftshift(np.fft.ifft2(prod))
    return Fprod


def build_roofx2_syn_c1(modulation_angle,SZ,sz,pupil=None,PSF=None,convolvPSF=False,TYPE=None):
    Mvp=np.zeros([SZ,SZ])
    Mvm=np.zeros([SZ,SZ])
    Mhp=np.zeros([SZ,SZ])
    Mhm=np.zeros([SZ,SZ])
    Mvp[:,0:SZ//2] = 1
    Mvp[:,SZ//2] = np.sqrt(0.5)
    Mvm[:,SZ//2:] = 1
    Mvm[:,SZ//2] = np.sqrt(0.5)
    Mhp[0:SZ//2,:] = 1
    Mhp[SZ//2,:] = np.sqrt(0.5)
    Mhm[SZ//2:,:] = 1
    Mhm[SZ//2,:] = np.sqrt(0.5)
    MASK = np.moveaxis(np.asarray( [Mvp,Mvm,Mhp,Mhm] ),0, -1   )
    radius_pix = modulation_angle*SZ/sz
    MODU_PATH = np.zeros([SZ,SZ])
    ep=1
    dim_x=SZ
    dim_y=SZ
    xc    = dim_x/2. #(dim_x-1)/2.
    yc    = dim_y/2. #(dim_y-1)/2.
    xx = np.resize(np.array(range(dim_x)), (dim_x, dim_y)) - xc
    yy = np.transpose(xx)
    rr = np.sqrt(xx**2+yy**2)
    idxin = np.where(rr > radius_pix-ep/2.)
    idxout = np.where(rr > radius_pix+ep/2.)
    MODU_PATH[idxin]=1.0
    MODU_PATH[idxout]=0.
    MODU_PATH = MODU_PATH / np.sum(MODU_PATH)
    Omega =  MODU_PATH.copy()
    if modulation_angle == 0.:
        Omega = Omega*0.
        Omega[SZ//2,SZ//2]=1.0
    ## CONVOLVE WITH PSF
    if convolvPSF==True:
        if PSF is None:
            pupilL = np.zeros([SZ,SZ])
            pupilL[SZ//2-sz//2:SZ//2+sz//2,SZ//2-sz//2:SZ//2+sz//2] = pupil
            PSF = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pupilL))))**2
        PSF=PSF/np.sum(PSF)
        Omega0=Omega.copy()
        Omega = (conv(Omega0,PSF)).real
        Omega=Omega/np.sum(Omega)
    ## COMPUTE SENSOR FILTERS
    MvpO = Mvp*Omega
    MvmO = Mvm*Omega
    MhpO = Mhp*Omega
    MhmO = Mhm*Omega
    TFx = -2j * (conv(Mvm,MvpO) - conv(Mvp,MvmO))
    TFy = -2j * (conv(Mhm,MhpO) - conv(Mhp,MhmO))
    return TFx,TFy,Mvp,Mvm,Mhp,Mhm, Omega, PSF, Omega0


def build_bio_blunt_syn_c1(modu_eff_blunt,modulation_angle,SZ,sz,pupil=None,PSF=None,convolvPSF=False,TYPE=None):
    r_modG=modu_eff_blunt*np.sqrt(2.0)
    RG=SZ//sz
    x = (np.asarray(range(SZ))-SZ//2)/RG
    # Mvp=np.zeros([SZ,SZ])
#     Mvm=np.zeros([SZ,SZ])
#     Mhp=np.zeros([SZ,SZ])
#     Mhm=np.zeros([SZ,SZ])
#     Mvp[:,0:SZ//2] = 1
#     Mvp[:,SZ//2] = np.sqrt(0.5)
#     Mvm[:,SZ//2:] = 1
#     Mvm[:,SZ//2] = np.sqrt(0.5)
#     Mhp[0:SZ//2,:] = 1
#     Mhp[SZ//2,:] = np.sqrt(0.5)
#     Mhm[SZ//2:,:] = 1
#     Mhm[SZ//2,:] = np.sqrt(0.5)
    BW = np.zeros([SZ])
    BW[0:SZ//2] = 1.
    BW2d = np.zeros([SZ,SZ])

    for k in range(0,SZ):
            BW2d[k,:] = BW

    shiftmod =  np.int64(np.round(r_modG*RG))
    BW2d_mod = BW2d.copy()

    for j in range(0,SZ):
            BW2d_mod[j,SZ//2-shiftmod:SZ//2+shiftmod] = -1.*x[SZ//2-shiftmod:SZ//2+shiftmod]*1/(2.*r_modG)+0.5
    #pdb.set_trace()        
    BW2d_mod[np.where(BW2d_mod>1.0)]=1.0
    BW2d_mod[np.where(BW2d_mod<0.)]=0.0
    Mvp = np.sqrt(BW2d_mod.copy())
    Mvm = np.sqrt(1.-Mvp**2)
    Mhp = np.copy(Mvp.T)
    Mhm = np.copy(Mvm.T)
    MASK = np.moveaxis(np.asarray( [Mvp,Mvm,Mhp,Mhm] ),0, -1   )
    radius_pix = modulation_angle*SZ/sz
    MODU_PATH = np.zeros([SZ,SZ])
    ep=1
    dim_x=SZ
    dim_y=SZ
    xc    = dim_x/2. #(dim_x-1)/2.
    yc    = dim_y/2. #(dim_y-1)/2.
    xx = np.resize(np.array(range(dim_x)), (dim_x, dim_y)) - xc
    yy = np.transpose(xx)
    rr = np.sqrt(xx**2+yy**2)
    idxin = np.where(rr > radius_pix-ep/2.)
    idxout = np.where(rr > radius_pix+ep/2.)
    MODU_PATH[idxin]=1.0
    MODU_PATH[idxout]=0.
    MODU_PATH = MODU_PATH / np.sum(MODU_PATH)
    Omega =  MODU_PATH.copy()
    if modulation_angle == 0.:
        Omega = Omega*0.
        Omega[SZ//2,SZ//2]=1.0
    ## CONVOLVE WITH PSF
    if convolvPSF==True:
        if PSF is None:
            pupilL = np.zeros([SZ,SZ])
            pupilL[SZ//2-sz//2:SZ//2+sz//2,SZ//2-sz//2:SZ//2+sz//2] = pupil
            PSF = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pupilL))))**2
        PSF=PSF/np.sum(PSF)
        Omega0=Omega.copy()
        Omega = (conv(Omega0,PSF)).real
        Omega=Omega/np.sum(Omega)
    ## COMPUTE SENSOR FILTERS
    MvpO = Mvp*Omega
    MvmO = Mvm*Omega
    MhpO = Mhp*Omega
    MhmO = Mhm*Omega
    TFx = -2j * (conv(Mvm,MvpO) - conv(Mvp,MvmO))
    TFy = -2j * (conv(Mhm,MhpO) - conv(Mhp,MhmO))
    return TFx,TFy,Mvp,Mvm,Mhp,Mhm, Omega, PSF, Omega0




def build_sensor_syn_c1(modulation_angle,SZ,sz,pupil=None,PSF=None,convolvPSF=False,TYPE=None):
    ## BUILD PYRAMID MASKS
    M1=np.zeros([SZ,SZ])
    M2=np.zeros([SZ,SZ])
    M3=np.zeros([SZ,SZ])
    M4=np.zeros([SZ,SZ])
    ## M1
    M1[SZ//2:,0:SZ//2] = 1
    M1[SZ//2,0:SZ//2] = np.sqrt(0.5)
    M1[SZ//2:,SZ//2] = np.sqrt(0.5)
    M1[SZ//2,SZ//2] = np.sqrt(0.25)
    ## M2
    M2[SZ//2:,SZ//2:] = 1
    M2[SZ//2,SZ//2:]= np.sqrt(0.5)
    M2[SZ//2:,SZ//2]= np.sqrt(0.5)
    M2[SZ//2,SZ//2] = np.sqrt(0.25)
    # M3
    M3[0:SZ//2,0:SZ//2] = 1
    M3[SZ//2,0:SZ//2] = np.sqrt(0.5)
    M3[0:SZ//2,SZ//2] = np.sqrt(0.5)
    M3[SZ//2,SZ//2] = np.sqrt(0.25)
    #M4
    M4[0:SZ//2,SZ//2:] = 1
    M4[SZ//2,SZ//2:] = np.sqrt(0.5)
    M4[0:SZ//2,SZ//2] = np.sqrt(0.5)
    M4[SZ//2,SZ//2]  = np.sqrt(0.25)
    MASK = np.moveaxis(np.asarray( [M1,M2,M3,M4] ),0, -1   )
    radius_pix = modulation_angle*SZ/sz
    MODU_PATH = np.zeros([SZ,SZ])
    ep=1
    dim_x=SZ
    dim_y=SZ
    xc    = dim_x/2. #(dim_x-1)/2.
    yc    = dim_y/2. #(dim_y-1)/2.
    xx = np.resize(np.array(range(dim_x)), (dim_x, dim_y)) - xc
    yy = np.transpose(xx)
    rr = np.sqrt(xx**2+yy**2)
    idxin = np.where(rr > radius_pix-ep/2.)
    idxout = np.where(rr > radius_pix+ep/2.)
    MODU_PATH[idxin]=1.0
    MODU_PATH[idxout]=0.
    MODU_PATH = MODU_PATH / np.sum(MODU_PATH)
    Omega =  MODU_PATH.copy()
    if modulation_angle == 0.:
        Omega = Omega*0.
        Omega[SZ//2,SZ//2]=1.0
    ## CONVOLVE WITH PSF
    if convolvPSF==True: 
        if PSF is None:
            pupilL = np.zeros([SZ,SZ])
            pupilL[SZ//2-sz//2:SZ//2+sz//2,SZ//2-sz//2:SZ//2+sz//2] = pupil
            PSF = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pupilL))))**2
        PSF=PSF/np.sum(PSF)
        Omega0=Omega.copy()
        Omega = (conv(Omega0,PSF)).real
        Omega=Omega/np.sum(Omega)
    ## COMPUTE SENSOR FILTERS
    if convolvPSF == False:
        Omega0=Omega.copy()
    M1O = M1*Omega
    M2O = M2*Omega
    M3O = M3*Omega
    M4O = M4*Omega
    TFx = 2j * (conv(M3,M2O) -conv(M2,M3O) + conv(M1,M4O) - conv(M4,M1O))
    TFy = 2j *(conv(M3,M2O) -conv(M2,M3O) - conv(M1,M4O) + conv(M4,M1O))
    return TFx,TFy,M1O,M2O,M3O,M4O, M1,M2,M3,M4, Omega, PSF, Omega0





def build_sensor_syn(modulation_angle,SZ,sz,pupil=None,PSF=None,convolvPSF=False,TYPE=None):
    ## BUILD PYRAMID MASKS
    M1=np.zeros([SZ,SZ])
    M2=np.zeros([SZ,SZ])
    M3=np.zeros([SZ,SZ])
    M4=np.zeros([SZ,SZ])
    M1[SZ//2:,0:SZ//2] = 1
    M2[SZ//2:,SZ//2:] = 1
    M3[0:SZ//2,0:SZ//2]=1
    M4[0:SZ//2,SZ//2:] = 1
    ## BUILD MODULATION PATH
    radius_pix = modulation_angle*SZ/sz
    MODU_PATH = np.zeros([SZ,SZ])
    ep=1
    dim_x=SZ
    dim_y=SZ
    xc    = (dim_x-1)/2.
    yc    = (dim_y-1)/2.
    xx = np.resize(np.array(range(dim_x)), (dim_x, dim_y)) - xc
    yy = np.transpose(xx)
    rr = np.sqrt(xx**2+yy**2)
    idxin = np.where(rr > radius_pix)
    idxout = np.where(rr > radius_pix+ep)
    MODU_PATH[idxin]=1.0
    MODU_PATH[idxout]=0.
    MODU_PATH = MODU_PATH / np.sum(MODU_PATH)
    Omega =  MODU_PATH.copy()
    if modulation_angle == 0.:
        Omega = Omega*0.
        Omega[SZ//2-1:SZ//2+1,SZ//2-1:SZ//2+1]=1.0
    ## CONVOLVE WITH PSF
    if convolvPSF==True:
        if PSF is None:
            pupilL = np.zeros([SZ,SZ])
            pupilL[SZ//2-sz//2:SZ//2+sz//2,SZ//2-sz//2:SZ//2+sz//2] = pupil
            PSF = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pupilL))))**2
        PSF=PSF/np.sum(PSF)
        Omega0=Omega.copy()
        Omega = (conv(Omega0,PSF)).real
        Omega=Omega/np.sum(Omega)
    ## COMPUTE SENSOR FILTERS
    M1O = M1*Omega
    M2O = M2*Omega
    M3O = M3*Omega
    M4O = M4*Omega
    TFx = 2j * (conv(M3,M2O) -conv(M2,M3O) + conv(M1,M4O) - conv(M4,M1O))
    TFy = 2j *(conv(M3,M2O) -conv(M2,M3O) - conv(M1,M4O) + conv(M4,M1O))
    return TFx,TFy

     
def build_im_syn(TFx,TFy,dd,FT_KLs_normed,dfa,ra):
    fc = 1./(2.*dd)
    SZ=TFx.shape[0]
    fx=(np.array(range(SZ))-SZ//2+0.5)*dfa
    ffx=np.zeros([SZ,SZ])
    for k in range(0,SZ):
        ffx[k,:] = fx
    ffy=np.transpose(ffx)
    TFx_dd = TFx*np.sinc(ffx*dd) * np.sinc(ffy*dd)
    TFy_dd = TFy*np.sinc(ffx*dd) * np.sinc(ffy*dd)
    zo=np.int(fc/dfa*ra)
    if zo % 2 != 0:
        zo = zo+1
    nmoi=FT_KLs_normed.shape[2]
    IM_XF = np.zeros([(2*zo)**2,nmoi-1],dtype=np.complex128)
    IM_YF = np.zeros([(2*zo)**2,nmoi-1],dtype=np.complex128)
    for k in range(0,nmoi-1):
        print(k, ' ', end='\r', flush=True)
        IM_XF[:,k] = np.reshape( (FT_KLs_normed[:,:,k+1]*TFx_dd)[SZ//2-zo:SZ//2+zo,SZ//2-zo:SZ//2+zo],[(2*zo)**2] )
        IM_YF[:,k] = np.reshape( (FT_KLs_normed[:,:,k+1]*TFy_dd)[SZ//2-zo:SZ//2+zo,SZ//2-zo:SZ//2+zo],[(2*zo)**2] )
    IM_F = np.zeros([2*(2*zo)**2,nmoi-1],dtype=np.complex128)
    for k in range(0,nmoi-1):
        print(k, ' ', end='\r', flush=True)
        IM_F[0:(2*zo)**2,k] = IM_XF[:,k] #/np.sqrt(2.)*2*fc*sz/SZ
        IM_F[(2*zo)**2:,k] = IM_YF[:,k]  #/np.sqrt(2.)*2*fc*sz/SZ
    REC_IM_F =  np.asmatrix(IM_F).I
    RRt_F = (REC_IM_F @ REC_IM_F.H).real
    NC_IM_F = np.diag(RRt_F)
    return NC_IM_F,IM_F,TFx_dd,TFy_dd


def build_im_syn_up(TFx,TFy,dd,FT_KLs_normed,dfa,ra,computeNC=False,computeSTD=True,N_PRO=30):
    fc = 1./(2.*dd)
    SZ=TFx.shape[0]
    fx=(np.array(range(SZ))-SZ//2)*dfa
    ffx=np.zeros([SZ,SZ])
    for k in range(0,SZ):
        ffx[k,:] = fx
    ffy=np.transpose(ffx)
    TFx_dd = TFx*np.sinc(ffx*dd) * np.sinc(ffy*dd)
    TFy_dd = TFy*np.sinc(ffx*dd) * np.sinc(ffy*dd)
    zo=np.int(fc/dfa*ra)
    if zo % 2 != 0:
        zo = zo+1
    nmoi=FT_KLs_normed.shape[2]
    reshape_im_syn_global_run(FT_KLs_normed,TFx_dd,TFy_dd,SZ,zo,N_PRO)
    #if computeSTD:
    #    global_std_run(nmoi-1,dfa,N_PRO)
    if computeNC:
        REC_IM_FG =  np.asmatrix(IM_FG).I
        RRt_FG = (REC_IM_FG @ REC_IM_FG.H).real
        NC_IM_FG = np.diag(RRt_FG)
    if computeNC == False:
        return TFx_dd,TFy_dd
    if computeNC:
        return NC_IM_FG,TFx_dd,TFy_dd



def reshape_im_syn_global(FT_KL,TFx,TFy,SZ,zo,kg):
    global IM_FG
    #pdb.set_trace()
    IM_FG[0:(2*zo)**2,kg] = np.reshape( (FT_KL*TFx)[SZ//2-zo:SZ//2+zo,SZ//2-zo:SZ//2+zo],[(2*zo)**2] )
    IM_FG[(2*zo)**2:,kg]= np.reshape( (FT_KL*TFy)[SZ//2-zo:SZ//2+zo,SZ//2-zo:SZ//2+zo],[(2*zo)**2] )

def global_std(kg,dfg):
    global IM_FG,STD_IM_FG
    STD_IM_FG[kg] = np.sqrt(np.sum(np.abs(IM_FG[:,kg]/np.sqrt(2.0))**2*dfg**2))

def global_std_run(nmoi,dfu,NPRO):
    my1=[]
    my2=[]
    for k in range(0,nmoi):
        my1.append(k)
        my2.append(dfu)
    result_=Parallel(n_jobs=NPRO,prefer="threads",verbose=10)(delayed(global_std)(i,j) for i,j in zip(my1,my2) )    

def reshape_im_syn_global_run(FK,Txd,Tyd,SZ,zo,N_PRO):
    my1=[]
    my2=[]
    my3=[]
    my4=[]
    my5=[]
    my6=[]
    nmoi = FK.shape[2]
    for k in range(0,nmoi-1):
        my1.append(FK[:,:,k+1])
        my2.append(Txd)
        my3.append(Tyd)
        my4.append(SZ)
        my5.append(zo)
        my6.append(k)
    st=time.time()
    result_=Parallel(n_jobs=N_PRO,prefer="threads",verbose=10)(delayed(reshape_im_syn_global)(i,j,k,l,m,n) for i,j,k,l,m,n in zip(my1,my2,my3,my4,my5,my6) )
    et=time.time()-st
    print('ELAPSED TIME FOR GLOBAL IM SYN COMPUTATION:',et)
    



# def build_im_syn_up(TFx,TFy,dd,FT_KLs_normed,dfa,ra,N_PROCS):
#     fc = 1./(2.*dd)
#     SZ=TFx.shape[0]
#     fx=(np.array(range(SZ))-SZ//2+0.)*dfa
#     ffx=np.zeros([SZ,SZ])
#     for k in range(0,SZ):
#         ffx[k,:] = fx
#     ffy=np.transpose(ffx)
#     TFx_dd = TFx*np.sinc(ffx*dd) * np.sinc(ffy*dd)
#     TFy_dd = TFy*np.sinc(ffx*dd) * np.sinc(ffy*dd)
#     zo=np.int(fc/dfa*ra)
#     if zo % 2 != 0:
#         zo = zo+1
#     nmoi=FT_KLs_normed.shape[2]
#     IM_XF = np.zeros([(2*zo)**2,nmoi-1],dtype=np.complex128)
#     IM_YF = np.zeros([(2*zo)**2,nmoi-1],dtype=np.complex128)
#     input1=[]
#     input2=[]
#     input3=[]
#     input4=[]
#     for k in range(0,nmoi-1):
#         input1.append(FT_KLs_normed[:,:,k+1])
#         input2.append(TFx_dd)
#         input3.append(SZ)
#         input4.append(zo)
#     st=time.time()
#     result_=Parallel(n_jobs=N_PROCS,prefer="threads",verbose=10)(delayed(TEMPO_TAYLOR_)(i,j,k) for i,j,k in zip(my1,my2,my3) )
#     et=time.time()-st
#     print('ELAPSED TIME FOR TAY LOR PSD COMUTATION:',et)
#     # for k in range(0,nmoi-1):
# #         print(k, ' ', end='\r', flush=True)
# #         IM_XF[:,k] = np.reshape( (FT_KLs_normed[:,:,k+1]*TFx_dd)[SZ//2-zo:SZ//2+zo,SZ//2-zo:SZ//2+zo],[(2*zo)**2] )
# #         IM_YF[:,k] = np.reshape( (FT_KLs_normed[:,:,k+1]*TFy_dd)[SZ//2-zo:SZ//2+zo,SZ//2-zo:SZ//2+zo],[(2*zo)**2] )
# #     IM_F = np.zeros([2*(2*zo)**2,nmoi-1],dtype=np.complex128)
#     for k in range(0,nmoi-1):
#         print(k, ' ', end='\r', flush=True)
#         IM_F[0:(2*zo)**2,k] = IM_XF[:,k] #/np.sqrt(2.)*2*fc*sz/SZ
#         IM_F[(2*zo)**2:,k] = IM_YF[:,k]  #/np.sqrt(2.)*2*fc*sz/SZ
#     REC_IM_F =  np.asmatrix(IM_F).I
#     RRt_F = (REC_IM_F @ REC_IM_F.H).real
#     NC_IM_F = np.diag(RRt_F)
#     return NC_IM_F,IM_F,TFx_dd,TFy_dd


def build_im_syn_al(TFx,TFy,dd,FT_KLs_normed,dfa,ra):
    fc = 1./(2.*dd)
    SZ=TFx.shape[0]
    fx=(np.array(range(SZ))-SZ//2+0.5)*dfa
    ffx=np.zeros([SZ,SZ])
    for k in range(0,SZ):
        ffx[k,:] = fx
    ffy=np.transpose(ffx)
    TFx_dd = TFx*np.sinc(ffx*dd) * np.sinc(ffy*dd)
    TFy_dd = TFy*np.sinc(ffx*dd) * np.sinc(ffy*dd)
    zo=np.int(fc/dfa*ra)
    if zo % 2 != 0:
        zo = zo+1
    nmoi=FT_KLs_normed.shape[2] 
    IM_XF = np.zeros([(2*zo)**2,nmoi-1],dtype=np.complex128)
    IM_YF = np.zeros([(2*zo)**2,nmoi-1],dtype=np.complex128)
    px_1sd =  1./dd/dfa
    nshiftx = np.int(SZ/px_1sd)
    nshifty = np.int(SZ/px_1sd)
    for k in range(0,nmoi-1):
        print(k, ' ', end='\r', flush=True)
        IMxF0 = FT_KLs_normed[:,:,k+1]*TFx_dd
        IMyF0 = FT_KLs_normed[:,:,k+1]*TFy_dd
        IMxF0s = IMxF0*0.
        IMyF0s = IMyF0*0.
        for m in range(-nshiftx//2+1 , nshiftx//2 ):
            for n in range(-nshiftx//2 , nshiftx//2 ):
                IMxF0s = IMxF0s + aou.myshift2D(IMxF0,m*px_1sd,n*px_1sd)
                IMyF0s = IMyF0s + aou.myshift2D(IMyF0,m*px_1sd,n*px_1sd)
        IM_XF[:,k] = np.reshape( (IMxF0s)[SZ//2-zo:SZ//2+zo,SZ//2-zo:SZ//2+zo],[(2*zo)**2] )
        IM_YF[:,k] = np.reshape( (IMyF0s)[SZ//2-zo:SZ//2+zo,SZ//2-zo:SZ//2+zo],[(2*zo)**2] )
    IM_F = np.zeros([2*(2*zo)**2,nmoi-1],dtype=np.complex128)
    for k in range(0,nmoi-1):
        print(k, ' ', end='\r', flush=True)
        IM_F[0:(2*zo)**2,k] = IM_XF[:,k] #/np.sqrt(2.)*2*fc*sz/SZ
        IM_F[(2*zo)**2:,k] = IM_YF[:,k]  #/np.sqrt(2.)*2*fc*sz/SZ
    REC_IM_F =  np.asmatrix(IM_F).I
    RRt_F = (REC_IM_F @ REC_IM_F.H).real
    NC_IM_F = np.diag(RRt_F)
    return NC_IM_F,IM_F,TFx_dd,TFy_dd




def build_im_syn_nodd(TFx,TFy,dd,FT_KLs_normed,dfa,ra):
    fc = 1./(2.*dd)
    SZ=TFx.shape[0]
    fx=(np.array(range(SZ))-SZ//2+0.5)*dfa
    ffx=np.zeros([SZ,SZ])
    for k in range(0,SZ):
        ffx[k,:] = fx
    ffy=np.transpose(ffx)
    TFx_dd = TFx*np.sinc(ffx*dd) * np.sinc(ffy*dd)
    TFy_dd = TFy*np.sinc(ffx*dd) * np.sinc(ffy*dd)
    zo=np.int(fc/dfa*ra)
    if zo % 2 != 0:
        zo = zo+1
    nmoi=FT_KLs_normed.shape[2]
    IM_XF = np.zeros([(2*zo)**2,nmoi-1],dtype=np.complex128)
    IM_YF = np.zeros([(2*zo)**2,nmoi-1],dtype=np.complex128)
    for k in range(0,nmoi-1):
        print(k, ' ', end='\r', flush=True)
        IM_XF[:,k] = np.reshape( (FT_KLs_normed[:,:,k+1]*TFx)[SZ//2-zo:SZ//2+zo,SZ//2-zo:SZ//2+zo],[(2*zo)**2] )
        IM_YF[:,k] = np.reshape( (FT_KLs_normed[:,:,k+1]*TFy)[SZ//2-zo:SZ//2+zo,SZ//2-zo:SZ//2+zo],[(2*zo)**2] )
    IM_F = np.zeros([2*(2*zo)**2,nmoi-1],dtype=np.complex128)
    for k in range(0,nmoi-1):
        print(k, ' ', end='\r', flush=True)
        IM_F[0:(2*zo)**2,k] = IM_XF[:,k] #/np.sqrt(2.)*2*fc*sz/SZ
        IM_F[(2*zo)**2:,k] = IM_YF[:,k]  #/np.sqrt(2.)*2*fc*sz/SZ
    REC_IM_F =  np.asmatrix(IM_F).I
    RRt_F = (REC_IM_F @ REC_IM_F.H).real
    NC_IM_F = np.diag(RRt_F)
    return NC_IM_F,IM_F,TFx_dd,TFy_dd
 
def build_modal_syn_arrays(kls_2d,dxo,PSDa=None,dfa=None,BLOCK_SIZE=None,N_PROCS=None):
    nmo=kls_2d.shape[2]
    sz=kls_2d.shape[0]
    pupil=np.zeros([sz,sz])
    pupil[np.where(kls_2d[:,:,0] != 0.)]=1.0      
    tpup=np.sum(pupil)
    if PSDa is None:
        diam_max=sz*dxo
        r0=1.0
        L0=30.0
        PSDa , dfa, pterma = VK_DSP_up(diam_max,r0,L0,SZ,sz,0,pupil)
        PSDa[SZ//2,SZ//2]=0
        #PSD_atmt=PSD_atmt #*(0.5e-6/(2.*np.pi))**2
    
    SZ=PSDa.shape[0]
    
    ## 1. FT_KL
    if BLOCK_SIZE is None:
        BLOCK_SIZE=10
    st=time.time()
    KLs_2D=np.zeros([SZ,SZ,nmo],dtype=np.float64)
    KLs_2D[SZ//2-sz//2:SZ//2+sz//2,SZ//2-sz//2:SZ//2+sz//2,:] = kls_2d[:,:,0:nmo]
    nsplit=nmo//BLOCK_SIZE
    norma_KL = 1/SZ/np.sqrt(tpup)/dfa
    FT_KLs = aou.DO_FT_MULTI(KLs_2D,nsplit)
    #FT_KLs_normed = FT_KLs/SZ/np.sqrt(tpup)/dfa
    et=time.time()-st
    print('ELAPSED TIME FOR FT of KLs:',et)
    
    ## 2. ABS(FT_KL)**2*PSDa and RMSn2
    ## COMPUTE ATM SPATIAL PSD (if not provied)
    if N_PROCS is None:
        N_PROCS=30
    
    ## MP PROCESSING
    input1=[]
    input2=[]
    ## BUILD THE LIST OF INPUTS FOR MULTITHREADING
    for k in range(0,nmo):
        input1.append(FT_KLs[:,:,k])
        input2.append(PSDa)
    my1=tuple(input1)
    my2=tuple(input2)
    ## EXECUTION
    st=time.time()
    result_=Parallel(n_jobs=N_PROCS,prefer="threads",verbose=10)(delayed(aou.PROD_FT_DSP_)(i,j) for i,j in zip(my1,my2) )
    et=time.time()-st
    FKL2xPSDa=np.zeros([SZ,SZ,nmo],dtype=np.float64)
    for k in range(0,nmo):
        FKL2xPSDa[:,:,k] =  result_[k]
    print('ELAPSED TIME FOR PSD(KL)*DSPa:',et)
    RMSn2 = np.sum(FKL2xPSDa*dfa**2,axis=(0,1))
    
    ## 3.FKL2xPSD_i1d
    axeu=1
    input1=[]
    input2=[]
    for k in range(0,nmo//nsplit):
        input1.append(FKL2xPSDa[:,:,k*nsplit:(k+1)*nsplit])
        input2.append(axeu)
    my1=tuple(input1)
    my2=tuple(input2)
    res_sum1 = Parallel(n_jobs=N_PROCS,prefer="threads",verbose=10)(delayed(sum_ARR2Dto1D)(i,j) for i,j in zip(my1,my2) )
    
    FKL2xPSD_i1d =  np.zeros([SZ,nmo])
    for k in range(0,nmo//nsplit):
        FKL2xPSD_i1d [:,k*nsplit:(k+1)*nsplit] = res_sum1[k]
    FT_KLs *= norma_KL
    return FT_KLs, norma_KL, RMSn2, FKL2xPSD_i1d

    
def build_modal_arrays_syn(kls_2d,dxo,PSDa=None,dfa=None,BLOCK_SIZE=None,N_PROCS=None):
    nmo=kls_2d.shape[2]
    sz=kls_2d.shape[0]
    pupil=np.zeros([sz,sz])
    pupil[np.where(kls_2d[:,:,0] != 0.)]=1.0      
    tpup=np.sum(pupil)
    if PSDa is None:
        diam_max=sz*dxo
        r0=1.0
        L0=30.0
        PSDa , dfa, pterma = VK_DSP_up(diam_max,r0,L0,SZ,sz,0,pupil)
        PSDa[SZ//2,SZ//2]=0
        #PSD_atmt=PSD_atmt #*(0.5e-6/(2.*np.pi))**2
    
    SZ=PSDa.shape[0]
    
    ## 1. FT_KL
    if BLOCK_SIZE is None:
        BLOCK_SIZE=10
    st=time.time()
    KLs_2D=np.zeros([SZ,SZ,nmo],dtype=np.float64)
    KLs_2D[SZ//2-sz//2:SZ//2+sz//2,SZ//2-sz//2:SZ//2+sz//2,:] = kls_2d[:,:,0:nmo]
    nsplit=nmo//BLOCK_SIZE
    norma_KL = 1/SZ/np.sqrt(tpup)/dfa
    FT_KLs = aou.DO_FT_MULTI(KLs_2D,nsplit)
    #FT_KLs_normed = FT_KLs/SZ/np.sqrt(tpup)/dfa
    et=time.time()-st
    print('ELAPSED TIME FOR FT of KLs:',et)
    
    ## 2. ABS(FT_KL)**2*PSDa and RMSn2
    ## COMPUTE ATM SPATIAL PSD (if not provied)
    if N_PROCS is None:
        N_PROCS=30
    
    ## MP PROCESSING
    input1=[]
    input2=[]
    ## BUILD THE LIST OF INPUTS FOR MULTITHREADING
    for k in range(0,nmo):
        input1.append(FT_KLs[:,:,k])
        input2.append(PSDa)
    my1=tuple(input1)
    my2=tuple(input2)
    ## EXECUTION
    st=time.time()
    result_=Parallel(n_jobs=N_PROCS,prefer="threads",verbose=10)(delayed(aou.PROD_FT_DSP_)(i,j) for i,j in zip(my1,my2) )
    et=time.time()-st
    FKL2xPSDa=np.zeros([SZ,SZ,nmo],dtype=np.float64)
    for k in range(0,nmo):
        FKL2xPSDa[:,:,k] =  result_[k]
    print('ELAPSED TIME FOR PSD(KL)*DSPa:',et)
    RMSn2 = np.sum(FKL2xPSDa*dfa**2,axis=(0,1))
    
    ## 3.FKL2xPSD_i1d
    axeu=1
    input1=[]
    input2=[]
    for k in range(0,nmo//nsplit):
        input1.append(FKL2xPSDa[:,:,k*nsplit:(k+1)*nsplit])
        input2.append(axeu)
    my1=tuple(input1)
    my2=tuple(input2)
    res_sum1 = Parallel(n_jobs=N_PROCS,prefer="threads",verbose=10)(delayed(sum_ARR2Dto1D)(i,j) for i,j in zip(my1,my2) )
    
    FKL2xPSD_i1d =  np.zeros([SZ,nmo])
    for k in range(0,nmo//nsplit):
        FKL2xPSD_i1d [:,k*nsplit:(k+1)*nsplit] = res_sum1[k]
    FT_KLs *= norma_KL
    return FT_KLs, norma_KL, FKL2xPSDa, RMSn2, FKL2xPSD_i1d


##### ERROR TERMS
def fitting_e_syn(PSDa,dfa,RMSn2,tpup,r0a,nmo):
    #piston needs to be included
    norm_r0a_meter=r0a**(-5./3.)/(tpup**2)*(0.5e-6/(2.*np.pi))**2
    RMSpsd = np.sqrt(np.sum(r0a**(-5./3.)*PSDa*dfa**2))*0.5e-6/(2.*np.pi)
    RMSu=np.sqrt(RMSn2*norm_r0a_meter)
    RMSu_tot0=np.sqrt(np.sum(RMSu[0:nmo]**2))
    fitting_u=np.sqrt(RMSpsd**2-RMSu_tot0**2)
    return fitting_u


def fitting_dsp_syn(PSDa,dfa,fitting_f,tpup,r0a):
    SZ=PSDa.shape[0]
    fx=(np.array(range(SZ))-SZ//2+0.)*dfa
    ffx=np.zeros([SZ,SZ])
    for k in range(0,SZ): ffx[k,:] = fx
    ffy=np.transpose(ffx)
    ffr = np.sqrt(ffx**2+ffy**2)
    ddmin = 0.1
    ddmax=1.0
    nfc=1000
    dds=np.asarray(range(nfc))*(ddmax)/(nfc)+ddmin+0.001
    fcs=1./(2.*dds)
    rmfs = np.zeros(nfc)
    for k in range(0,nfc):
        idxLO = np.where(ffr < fcs[k])
        PSDa_fitting = PSDa.copy()
        PSDa_fitting[idxLO] = 0.
        rmfs[k] = np.sqrt(np.sum(PSDa_fitting*dfa**2))  
    ks = np.min(np.where(rmfs > fitting_f ))
    fce = fcs[ks]
    idxLO = np.where(ffr < fce)
    PSDa_fitting_m = PSDa.copy()
    PSDa_fitting_m[idxLO] = 0.
    RMSfids = np.sqrt(np.sum(PSDa_fitting_m*dfa**2))
    print('Fitting error is: ',RMSfids)
    return RMSfids, PSDa_fitting_m


def temporal_e_syn(FKL2xPSD_i1d,dfu,tpup,r0a,nmo,Vx,Ti=None,TauC=None,Tdm=None,loop_gain=0.5,SIMU=True):
    # THE PISTON IS REMOVED AT THE END, SO THE INPUT FKL2xPSD_i1d CONTAINS THE PISTON. MAY BE REMOVED LATER ?
    # if type(loop_gain) is np.float64:
#         gains = np.zeros(nmo)
#         gains[:] = loop_gain
#     if  type(loop_gain) is np.ndarray:
#         gains=loop_gain.copy()
    SZ = FKL2xPSD_i1d.shape[0]
    fxu=np.array(range(SZ//2))*dfu
    fy=fxu
    dfy=dfu
    Fv_max = Vx*np.max(fxu)
    Fi_max = 1./Ti/2.
    normG=0.5e-6/(2.*np.pi)/(tpup/np.sqrt(2.))
    norm1D_P = normG**2*dfy
    wg = FKL2xPSD_i1d[SZ//2:,1:nmo]*norm1D_P
    RMSi,RMSo,DSPi,DSPo,fp,dfp,Her_s  = aou.AO_CONTROL_TEMPORAL_ERROR(wg,fxu,r0a,Vx,Ti=Ti,TauC=TauC,Tdm=Tdm,loop_gain=loop_gain,which='FMAX',SIMU=SIMU)
    RMS_tempo_tot = np.sqrt(np.sum(RMSo**2))
    print('Temporal error on ',nmo, ' modes is : ' ,  RMS_tempo_tot )
    return RMS_tempo_tot,RMSo,Her_s,fp,dfp
    #return RMS_tempo_tot,RMSo,RMSi,Her_s,fp,dfp



def temporal_e_syn_og(FKL2xPSD_i1d,dfu,tpup,r0a,nmo,Vx,Ti=None,TauC=None,Tdm=None,OG=None,loop_gain=0.5,SIMU=True):
    # THE PISTON IS REMOVED AT THE END, SO THE INPUT FKL2xPSD_i1d CONTAINS THE PISTON. MAY BE REMOVED LATER ?
    # if type(loop_gain) is np.float64:
#         gains = np.zeros(nmo)
#         gains[:] = loop_gain
#     if  type(loop_gain) is np.ndarray:
#         gains=loop_gain.copy()
    SZ = FKL2xPSD_i1d.shape[0]
    fxu=np.array(range(SZ//2))*dfu
    fy=fxu
    dfy=dfu
    Fv_max = Vx*np.max(fxu)
    Fi_max = 1./Ti/2.
    normG=0.5e-6/(2.*np.pi)/(tpup/np.sqrt(2.))
    norm1D_P = normG**2*dfy
    wg = FKL2xPSD_i1d[SZ//2:,1:nmo]*norm1D_P
    #pdb.set_trace()
    RMSi,RMSo,DSPi,DSPo,fp,dfp,Her_s  = aou.AO_CONTROL_TEMPORAL_ERROR_OG(wg,fxu,r0a,Vx,Ti=Ti,TauC=TauC,Tdm=Tdm,OG=OG,loop_gain=loop_gain,which='FMAX',SIMU=SIMU)
    RMS_tempo_tot = np.sqrt(np.sum(RMSo**2))
    print('Temporal error on ',nmo, ' modes is : ' ,  RMS_tempo_tot )
    return RMS_tempo_tot,RMSo,Her_s,fp,dfp





def temporal_e_fou(PSDa_m,PSD_fi_m,dfa,r0a,Vx,Ti=None,TauC=None,Tdm=None,loop_gain=0.5,SIMU=True):
    SZ=PSD_fi_m.shape[0]
    fx=(np.array(range(SZ))-SZ//2+0.)*dfa
    ffx=np.zeros([SZ,SZ])
    for k in range(0,SZ): ffx[k,:] = fx
    ffy=np.transpose(ffx)
    ffr = np.sqrt(ffx**2+ffy**2)
    idKnz = np.where( (PSD_fi_m == 0.) & (ffx !=0))
    if type(loop_gain) is float:
        gain_tempo=np.zeros(len(idKnz[0]))
        gain_tempo[:] = loop_gain
    else:
        gain_tempo = loop_gain.copy()
    NUsV = np.abs(ffx[idKnz]*Vx)
    H_ER_NUsV,H_CL, H_OL,H_N = aou.TF_Her_Hcl_Hol_Hn_SIMU(NUsV,gain_tempo,Ti,TauC,Tdm)
    #pdb.set_trace()
    zerdx=np.where(ffx[idKnz] ==0.)
    H_ER_NUsV[zerdx]=1.0
    ERR = np.zeros([SZ,SZ])
    ERR[idKnz] = np.abs(H_ER_NUsV)**2*PSDa_m[idKnz]
    RMS_ERRtot = np.sqrt(np.sum(ERR*dfa**2))
    print('Temporal error (fourier model): ' , RMS_ERRtot  )
    return RMS_ERR, ERR



def noise_e_fou(PSD_fi_m,dfa,NPHa, ron,NC, wavelength, Hn_se=None,Ti=None,TauC=None,Tdm=None,loop_gain=0.5,SIMU=True):
    SZ=PSD_fi_m.shape[0]
    fx=(np.array(range(SZ))-SZ//2+0.)*dfa
    ffx=np.zeros([SZ,SZ])
    for k in range(0,SZ): ffx[k,:] = fx
    ffy=np.transpose(ffx)
    ffr = np.sqrt(ffx**2+ffy**2)
    idKnz = np.where( (PSD_fi_m == 0.) & (ffx !=0))
    if type(loop_gain) is float:
        gain_tempo=np.zeros(len(idKnz[0]))
        gain_tempo[:] = loop_gain
    else:
        gain_tempo = loop_gain.copy()
    NUsV = np.abs(ffx[idKnz]*Vx)
    H_ER_NUsV,H_CL, H_OL,H_N = aou.TF_Her_Hcl_Hol_Hn_SIMU(NUsV,gain_tempo,Ti,TauC,Tdm)
    #pdb.set_trace()
    zerdx=np.where(ffx[idKnz] ==0.)
    H_ER_NUsV[zerdx]=1.0
    ERR = np.zeros([SZ,SZ])
    ERR[idKnz] = np.abs(H_ER_NUsV)**2*PSDa_m[idKnz]
    RMS_ERRtot = np.sqrt(np.sum(ERR*dfa**2))
    print('Temporal error (fourier model): ' , RMS_ERRtot  )
    return RMS_ERR, ERR

       



def aliasing_e_syn(PSDa_f,dfa,dd,IM_F,TFx_dd,TFy_dd,hfc=True):
    SZ=PSDa_f.shape[0]
    ra=1.0
    fc=1./(2.*dd)
    zo=np.int(fc/dfa*ra)
    if zo % 2 != 0:
        zo = zo+1
    px_1sd =  1./dd/dfa
    nshiftx = np.int(SZ/px_1sd)
    nshifty = np.int(SZ/px_1sd)
    B2xALs = np.zeros([SZ,SZ],dtype=np.complex128) 
    B2yALs = np.zeros([SZ,SZ],dtype=np.complex128) 
    B2xyALs = np.zeros([SZ,SZ],dtype=np.complex128)
    if hfc == True:
        B2xAL0 = TFx_dd*np.conj(TFx_dd)*PSDa_f*SZ**2/sz**2
        B2yAL0 = TFy_dd*np.conj(TFy_dd)*PSDa_f*SZ**2/sz**2
        B2xyAL0 = TFx_dd*np.conj(TFy_dd)*PSDa_f*SZ**2/sz**2
    if hfc == False:
        B2xAL0 = TFx_dd*np.conj(TFx_dd)*PSDa_f*SZ**2/tpup
        B2yAL0 = TFy_dd*np.conj(TFy_dd)*PSDa_f*SZ**2/tpup
        B2xyAL0 = TFx_dd*np.conj(TFy_dd)*PSDa_f*SZ**2/tpup
    for m in range(-nshiftx//2+1 , nshiftx//2 ):
        print(m, ' ', end='\r', flush=True)
        for n in range(-nshiftx//2 , nshiftx//2 ):    
            if (m!=0) or (n!=0) :
                B2xALs = B2xALs + aou.myshift2D(B2xAL0,m*px_1sd,n*px_1sd)
                B2yALs = B2yALs + aou.myshift2D(B2yAL0,m*px_1sd,n*px_1sd)
                B2xyALs = B2xyALs + aou.myshift2D(B2xyAL0,m*px_1sd,n*px_1sd)
    #########
    nmes_sh = IM_F.shape[0]
    REC_F= np.asmatrix(IM_F).I
    Cbal = np.zeros([nmes_sh,nmes_sh],dtype=np.complex128)
    B2xAL1d = np.reshape( (B2xAL0)[SZ//2-zo:SZ//2+zo,SZ//2-zo:SZ//2+zo],[(2*zo)**2] )
    B2yAL1d = np.reshape( (B2yAL0)[SZ//2-zo:SZ//2+zo,SZ//2-zo:SZ//2+zo],[(2*zo)**2] )
    B2xyAL1d = np.reshape( (B2xyAL0)[SZ//2-zo:SZ//2+zo,SZ//2-zo:SZ//2+zo],[(2*zo)**2] )
    for i in range(0,nmes_sh//2):
        print(i, ' ', end='\r', flush=True)
        Cbal[i,i] =  B2xAL1d[i]
        Cbal[i+nmes_sh//2,i+nmes_sh//2] = B2yAL1d[i]
        Cbal[i,i+nmes_sh//2] = B2xyAL1d[i]
        Cbal[i+nmes_sh//2,i] = B2xyAL1d[i]
    CRAL = REC_F @ Cbal @ REC_F.H
    RMSal = np.sqrt(np.diag(CRAL.real))
    RMSal_tot=np.sqrt(np.sum(RMSal**2))
    print('Aliasing error is: ',RMSal_tot)
    return RMSal_tot, RMSal


def temporal_e_syn_noP(FKL2xPSD_i1d,dfu,tpup,r0a,nmo,Vx,Ti=None,TauC=None,Tdm=None,loop_gain=0.5,SIMU=True):
    # THE PISTON IS REMOVED AT THE END, SO THE INPUT FKL2xPSD_i1d CONTAINS THE PISTON. MAY BE REMOVED LATER ?
    # if type(loop_gain) is np.float64:
#         gains = np.zeros(nmo)
#         gains[:] = loop_gain
#     if  type(loop_gain) is np.ndarray:
#         gains=loop_gain.copy()
    SZ = FKL2xPSD_i1d.shape[0]
    fxu=np.array(range(SZ//2))*dfu
    fy=fxu
    dfy=dfu
    Fv_max = Vx*np.max(fxu)
    Fi_max = 1./Ti/2.
    normG=0.5e-6/(2.*np.pi)/(tpup/np.sqrt(2.))
    norm1D_P = normG**2*dfy
    wg = FKL2xPSD_i1d[SZ//2:,0:nmo]*norm1D_P
    RMSi,RMSo,DSPi,DSPo,fp,dfp,Her_s  = aou.AO_CONTROL_TEMPORAL_ERROR(wg,fxu,r0a,Vx,Ti=Ti,TauC=TauC,Tdm=Tdm,loop_gain=loop_gain,which='FMAX',SIMU=SIMU)
    RMS_tempo_tot = np.sqrt(np.sum(RMSo**2))
    print('Temporal error on ',nmo, ' modes is : ' ,  RMS_tempo_tot )
    return RMS_tempo_tot,RMSo,Her_s,fp,dfp


def noise_e_syn(NPHa, ron,NC, wavelength, Hn_se=None,loop_gain=None,Ti=None,TauC=None,Tdm=None,SIMU=True,which=None,fpe=None,dfpe=None,sze=None,f_ext=None):
    Sigma_n = np.sqrt(NPHa+4*ron**2)/NPHa
    Sigma_n_meter = Sigma_n * wavelength/(2.*np.pi)
    VAR_NOISE_PROPAG = NC * Sigma_n_meter**2
    nmo = NC.shape[0]
    RMSnoise, Hn_se, fpe, dfpe = aou.AO_CONTROL_NOISE_ERROR(VAR_NOISE_PROPAG, Hn_se=None,loop_gain=loop_gain,Ti=Ti,TauC=TauC,Tdm=Tdm,SIMU=True,which=None,fpe=fpe,dfpe=dfpe,sze=None,f_ext=None)
    RMS_noise_tot = np.sqrt(np.sum(RMSnoise**2))
    print('Noise  error on ',nmo, ' modes is : ' ,  RMS_noise_tot )
    return RMS_noise_tot,RMSnoise, Hn_se, fpe, dfpe


def noise_e_syn_og(NPHa, ron,NC, wavelength, Hn_se=None,OG=None,loop_gain=None,Ti=None,TauC=None,Tdm=None,SIMU=True,which=None,fpe=None,dfpe=None,sze=None,f_ext=None):
    Sigma_n = np.sqrt(NPHa+4*ron**2)/NPHa
    Sigma_n_meter = Sigma_n * wavelength/(2.*np.pi)
    VAR_NOISE_PROPAG = NC * Sigma_n_meter**2
    nmo = NC.shape[0]
    RMSnoise, Hn_se, fpe, dfpe = aou.AO_CONTROL_NOISE_ERROR_OG(VAR_NOISE_PROPAG, Hn_se=None,OG=OG,loop_gain=loop_gain,Ti=Ti,TauC=TauC,Tdm=Tdm,SIMU=True,which=None,fpe=fpe,dfpe=dfpe,sze=None,f_ext=None)
    RMS_noise_tot = np.sqrt(np.sum(RMSnoise**2))
    print('Noise  error on ',nmo, ' modes is : ' ,  RMS_noise_tot )
    return RMS_noise_tot,RMSnoise, Hn_se, fpe, dfpe



def analyse_gains(RMS1,RMS2,GAINS):
    nmodes = RMS1.shape[0]
    optimum_gains = np.zeros(nmodes)
    NG = RMS1.shape[1]
    for k in range(0,nmodes):
        VAR = RMS1[k,:]**2 + RMS2[k,:]**2
        idx = np.where(VAR == np.min(VAR))
        optimum_gains[k]=GAINS[idx]
    return optimum_gains

def analyse_gains_og(RMS1,RMS2,GAINS,OGos):
    nmodes = RMS1.shape[0]
    optimum_gains = np.zeros(nmodes)
    eff_gains = np.zeros(nmodes)
    NG = RMS1.shape[1]
    for k in range(0,nmodes):
        VAR = RMS1[k,:]**2 + RMS2[k,:]**2
        idx = np.where(VAR == np.min(VAR))
        optimum_gains[k]=GAINS[idx]
        eff_gains[k] = optimum_gains[k]*OGos[k]
    return optimum_gains, eff_gains




#OG_NCF = NC_B / NC_F

def BUILD_MTF(SZ,pupil,SIGN=False):
    sz=pupil.shape[0]
    tpup=np.sum(pupil)
    PUPIL = np.zeros([SZ,SZ])
    PUPIL[SZ//2-sz//2 : SZ//2+sz//2 , SZ//2-sz//2 : SZ//2+sz//2 ] = pupil
    P_c = corre(PUPIL,PUPIL)
    MTF0 = P_c/tpup
    if SIGN == True:
        idxnull = np.where(MTF0< 0.)
        MTF0[idxnull] = 0.
    return MTF0

def BUILD_STF_fromDSP(PSDa_m,df):
    RMSa_m = np.sqrt(np.sum(PSDa_m*df**2))
    FT_PSDa_m =  np.fft.fftshift(  np.fft.fft2( np.fft.fftshift (PSDa_m)  )  ).real
    COVa_m = FT_PSDa_m/np.max(FT_PSDa_m)*RMSa_m**2
    STRa_m = 2.*(np.max(COVa_m) - COVa_m)
    return STRa_m


def BUILD_DSP_fromSTF(STRt,RMSt,df):
    idd = np.where(STRt == 0.)
    COVu = RMSt**2 - 0.5*STRt
    COVu[idd] = 0.
    FT_COVu =  np.fft.fftshift(  np.fft.fft2( np.fft.fftshift (COVu)  )  ).real
    DSPu = FT_COVu/np.sum(FT_COVu)*RMSt**2/dfa**2
    return DSPu

 
   #  COV = RMS_**2 - STF_m/2.
#     FT_COV =  np.fft.fftshift(  np.fft.fft2( np.fft.fftshift (COV)  )  ).real
#     DSP = FT_COV/np.sum(FT_COV)*RMS_**2/df**2
#     return DSP



def BUILD_STF_fromRMS(RMS_B,Uii_KLs_N,nmoL ):
    SZ = Uii_KLs_N.shape[0]
    STF = np.zeros([SZ,SZ])
    for k in range(0,nmoL):
        print(k, ' ', end='\r', flush=True)
        STF = STF + RMS_B[k]**2 * Uii_KLs_N[:,:,k+1]
    return STF


def BUILD_STF_fromRMS_ups(RMS_B,Uiisa):
    SZ = np.int(np.sqrt(Uiisa.shape[0]))
    diag_mat = np.float32(np.diag(RMS_B**2))
    UiisXc = Uiisa@diag_mat
    UiisXc_summed = np.sum(UiisXc,axis=1)
    STF = np.reshape(UiisXc_summed,[SZ,SZ])
    return STF







# def BUILD_STF_fromKL_COV(Cmo_B,Uii_KLs_N,nmoL,UijL_KLs_N=UijL_KLs_N,IJ=False ):
#     SZ = Uii_KLs_N.shape[0]
#     STF = np.zeros([SZ,SZ])
#     if IJ == True:
#         nspm = UijL_KLs_N.shape[3]
#     if IJ == False:
#         for k in range(0,nmoL):
#             print(k, ' ', end='\r', flush=True)
#             STF = STF + Cmo_B[k,k] * Uii_KLs_N[:,:,k]
#     if IJ == True:
#         for k in range(0,nmoL):
#             print(k, ' ', end='\r', flush=True)
#             STF = STF + Cmo_B[k,k] * Uii_KLs_N[:,:,k]
#             for j in range(0,nspm):
#                 if k != j:
#                     STF = STF + 2.*Cmo_B[k,j] * UijL_KLs_N[:,:,k,j]
#     return STF


def BUILD_PSF_fromSTF(STF, MTF0):
    SZ=MTF0.shape[0]
    FTO = np.exp(-STF/2.)
    FTOtot = MTF0 * FTO
    PSF = np.abs(np.fft.fftshift(  np.fft.fft2( np.fft.fftshift (FTOtot/SZ**2)  )  ).real)
    PSFn=PSF/np.sum(PSF)
    return PSFn

#def BUILD_PSD_fromSTF(PSDa_m,df)


def BUILD_Uii_CORRE(KLs_2D,PUPIL,norma=True,val_norm=1.e-3):
    P_c = corre(PUPIL,PUPIL)
    iddPc = np.where(P_c > np.max(P_c)*val_norm)
    nmo=KLs_2D.shape[2]
    SZ=KLs_2D.shape[0]
    Uii_ = np.zeros([SZ,SZ,nmo])
    for k in range(0,nmo):
        print(k, ' ', end='\r', flush=True)
        Uii_[:,:,k] = corre(KLs_2D[:,:,k]**2*PUPIL,PUPIL) + corre(PUPIL,KLs_2D[:,:,k]**2*PUPIL) - 2.*corre(KLs_2D[:,:,k]*PUPIL,KLs_2D[:,:,k]*PUPIL)
    if norma == False:
        Uii_N = Uii_
    if norma == True:
       Uii_N = np.zeros([SZ,SZ,nmo])
       ratio = np.zeros([SZ,SZ])
       for k in range(0,nmo):
           ratio[iddPc] = Uii_[:,:,k][iddPc] / P_c[iddPc]
           Uii_N[:,:,k] = ratio
    return Uii_N


def BUILD_UijL(SZ,kls_2d,l_2d,pupil,norma=True,val_norm=1.e-3, IJ=True):
    nmo = kls_2d.shape[2]
    nmL = l_2d.shape[2]
    ncr = nmo*nmL
    KLs_2D = np.zeros([SZ,SZ,nmo])
    L_2D = np.zeros([SZ,SZ,nmL])
    L_2D[SZ//2-sz//2 : SZ//2+sz//2 , SZ//2-sz//2 : SZ//2+sz//2 , :] = l_2d
    PUPIL = np.zeros([SZ,SZ])
    KLs_2D[SZ//2-sz//2 : SZ//2+sz//2 , SZ//2-sz//2 : SZ//2+sz//2 , :] = kls_2d
    PUPIL[SZ//2-sz//2 : SZ//2+sz//2 , SZ//2-sz//2 : SZ//2+sz//2 ] = pupil
    P_c = corre(PUPIL,PUPIL)
    iddPc = np.where(P_c > np.max(P_c)*val_norm)
    Uii_ = np.zeros([SZ,SZ,nmo])
    UijL_ = np.zeros([SZ,SZ,nmo,nmL])
    for k in range(0,nmo):
        print(k, ' ', end='\r', flush=True)
        Uii_[:,:,k] = corre(KLs_2D[:,:,k]**2*PUPIL,PUPIL) + corre(PUPIL,KLs_2D[:,:,k]**2*PUPIL) - 2.*corre(KLs_2D[:,:,k]*PUPIL,KLs_2D[:,:,k]*PUPIL)
    if IJ == True:
        for k in range(0,nmo):
            print(k, ' ', end='\r', flush=True)
            for j in range(0,nmL):
                UijL_[:,:,k,j] = corre(KLs_2D[:,:,k]*L_2D[:,:,j]*PUPIL,PUPIL)+ corre(PUPIL,KLs_2D[:,:,k]*L_2D[:,:,j]*PUPIL) - 2.*corre(KLs_2D[:,:,k]*PUPIL,L_2D[:,:,j]*PUPIL)
    if norma == False:
        Uii_N = Uii_
        if IJ == True:
            UijL_N = UijL_
    if norma == True:
       Uii_N = np.zeros([SZ,SZ,nmo])
       if IJ == True:
           UijL_N = np.zeros([SZ,SZ,nmo,nmL])
       ratio = np.zeros([SZ,SZ])
       for k in range(0,nmo):
           ratio[iddPc] = Uii_[:,:,k][iddPc] / P_c[iddPc]
           Uii_N[:,:,k] = ratio
       if IJ == True:
           for k in range(0,nmo):
               for j in range(0,nmL):
                   ratio[iddPc] = UijL_[:,:,k,j][iddPc] / P_c[iddPc]
                   UijL_N[:,:,k,j] = ratio
    if IJ == True:
        return Uii_N, UijL_N
    if IJ == False:
        return Uii_N



def BUILD_UijL_CORRE(KLs_2D,L_2D,PUPIL,norma=True,val_norm=1.e-3):
    P_c = corre(PUPIL,PUPIL)
    iddPc = np.where(P_c > np.max(P_c)*val_norm)
    nmo = KLs_2D.shape[2]
    nmL = L_2D.shape[2]
    ncr = nmo*nmL
    SZ=KLs_2D.shape[0]
    Uii_ = np.zeros([SZ,SZ,nmo])
    UijL_ = np.zeros([SZ,SZ,nmo,nmL])
    for k in range(0,nmo):
        print(k, ' ', end='\r', flush=True)
        Uii_[:,:,k] = corre(KLs_2D[:,:,k]**2*PUPIL,PUPIL) + corre(PUPIL,KLs_2D[:,:,k]**2*PUPIL) - 2.*corre(KLs_2D[:,:,k]*PUPIL,KLs_2D[:,:,k]*PUPIL)
    for k in range(0,nmo):
        print(k, ' ', end='\r', flush=True)
        for j in range(0,nmL):
            UijL_[:,:,k,j] = corre(KLs_2D[:,:,k]*L_2D[:,:,j]*PUPIL,PUPIL)+ corre(PUPIL,KLs_2D[:,:,k]*L_2D[:,:,j]*PUPIL) - 2.*corre(KLs_2D[:,:,k]*PUPIL,L_2D[:,:,j]*PUPIL)
    if norma == False:
        Uii_N = Uii_
        UijL_N = UijL_
    if norma == True:
       Uii_N = np.zeros([SZ,SZ,nmo])
       UijL_N = np.zeros([SZ,SZ,nmo,nmL])
       ratio = np.zeros([SZ,SZ])
       for k in range(0,nmo):
           ratio[iddPc] = Uii_[:,:,k][iddPc] / P_c[iddPc]
           Uii_N[:,:,k] = ratio
           for j in range(0,nmL):
               ratio[iddPc] = UijL_[:,:,k,j][iddPc] / P_c[iddPc]
               UijL_N[:,:,k,j] = ratio
    return Uii_N, UijL_N



# AOparam={'Vx': 16.666, 'Ti': 2.e-3, 'TauC': 1.e-3, 'Tdm': 1.e-3, 'gain': np.float(0.5)}
# SENSORparam

# SENSOR='PYRAMID'

#def forecast(PSDa, dfa, r0a, Vx, TF_x, Sensor ,AOparam,):


#TURB_PARAM={}

#def AO_FORECAST(TURB_PARAM, AO_PARAM, ):
    

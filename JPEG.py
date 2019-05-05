import skvideo.io
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pywt
from scipy.ndimage.filters import gaussian_filter 
from scipy.misc import *





rgb_to_YCbCr = np.array([[0.29900, 0.58700, 0.11400],[-0.16874, -0.33126, 0.50000],[0.50000, -0.41869, -0.08131]])

YCbCr_to_rgb = np.linalg.inv(rgb_to_YCbCr)


##converts from RGB To YCbCr 
def rgb2YCbCr(U):
    width = U.shape[0]
    height = U.shape[1]

    YCbCr = np.zeros(U.shape)
    for i in range(width):
        for j in range(height):

            YCbCr[i,j] = np.dot(rgb_to_YCbCr,U[i,j]) + np.array([[0],[128],[128]])[:,0]
    return YCbCr


##converts from YCbCr To RGB
def YCbCr2rgb(U):
    s = U.shape
    Y = np.zeros(s)
    for i in range(s[0]):
        for j in range(s[1]):
            Y[i,j] = np.dot(YCbCr_to_rgb,U[i,j] -  np.array([[0],[128],[128]])[:,0]) 
    return Y


def video_RGB_to_YCbCr(rbgvideo):
    
    frames = rbgvideo.shape[0]

    YCbCr_video = np.zeros(rbgvideo.shape)
    
    
    for k in range(frames):
        YCbCr_video[k,:,:,:]  = rgb2ycbcr(rbgvideo[k,:,:,:])

            
    return YCbCr_video

def video_YCbCr_to_RGB(YCbCr_video):
    
    frames = YCbCr_video.shape[0]

    rgb_video = np.zeros(YCbCr_video.shape)
    
    
    for k in range(frames):
        rgb_video[k,:,:,:]  = ycbcr2rgb(YCbCr_video[k,:,:,:])

            
    return rgb_video.astype(np.uint8)

def quantize(data,factor):
    return np.sign(data)*(abs(data)//factor)

def dequantize(coeffs, step_size, delta=0.5):
    # delta: a user-selectable parameter. 0 <= delta < 1.
    return np.sign(coeffs)*(abs(coeffs) + delta)*step_size


def thresh_dwt(dwt, f):
    # f: the fraction f largest wavelet coeffs (to save)
    # parts borrowed from hw9
    m = np.sort(abs(dwt.ravel()))[::-1]
    idx = int(len(m) * f) # the fraction f largest wavelet coeff.
    thr = m[idx] # threshhold
    return dwt * (abs(dwt) > thr)

def stackDWT(LL, coeffs):
    LH, HL, HH = coeffs
    return np.vstack((np.hstack((LL, LH)), np.hstack((HL, HH))))


def dwt2(im, level=1, wavelet='db4'):
    coeffs = pywt.wavedec2(im, wavelet=wavelet, mode='per', level=level)
    Wim, rest = coeffs[0], coeffs[1:]
    for levels in rest:
        Wim = stackDWT(Wim, levels)
    return Wim

def get_multiple_shape(shape, block_size):
    # finds the shape where each dimension is a multiple of block_size
    horiz_blocks = shape[0]//block_size
    if horiz_blocks*block_size != shape[0]:
        # we add a block to this dim so blocks fit w/out overlap
        horiz_blocks += 1

    vert_blocks = shape[1]//block_size
    if vert_blocks*block_size != shape[1]:
        vert_blocks += 1

    new_shape = (horiz_blocks*block_size, vert_blocks*block_size)
    return new_shape

def map2multiple(image, block_size):
    # map the original image so dims are a multiple of block_size.
    horiz, vert = get_multiple_shape(image.shape, block_size)
    new_image = np.zeros((horiz, vert,3))
    new_image[:image.shape[0],:image.shape[1],:] = image
    return new_image

def unstack_coeffs(Wim):
        L1, L2  = np.hsplit(Wim, 2) 
        LL, HL = np.vsplit(L1, 2)
        LH, HH = np.vsplit(L2, 2)
        return LL, [LH, HL, HH]
    
def idwt2(Wim, levels=1, wavelet='db4'):
    coeffs = img2coeffs(Wim, levels=levels)
    return pywt.waverec2(coeffs, wavelet=wavelet, mode='per')


def img2coeffs(Wim, levels=4):
    LL, c = unstack_coeffs(Wim)
    coeffs = [c]
    for i in range(levels-1):
        LL, c = unstack_coeffs(LL)
        coeffs.insert(0,c)
    coeffs.insert(0, LL)
    return coeffs

def wavelet_level(data,wvlt):
    wv = pywt.Wavelet(wvlt)
    return pywt.dwt_max_level(data_len=np.min(data.shape), filter_len=wv.dec_len)


def compress_block(frame,wavelet = 'db4',threshhold = .15):


    ycc_frame = rgb2YCbCr(frame)
    ycc_breakdown = [ycc_frame[:,:,0],imresize(ycc_frame[:,:,1],.5),imresize(ycc_frame[:,:,2],.5)]

    wavelet = 'db4'

    ycc_compressed = []
    levels = []
    for channel in ycc_breakdown :
        level = wavelet_level(channel,wavelet)
        stepsize = 2**level
        
        
        dwt = dwt2(channel,level=level,wavelet=wavelet)
        thresholded_dwt= thresh_dwt(dwt, f=threshhold)
        quatized_dwt = quantize(dwt, stepsize)
        levels.append(level)
        ycc_compressed.append(quatized_dwt)

        
    return ycc_compressed,levels

def decompress_block(ycc_compressed,levels,og_shape,wavelet = 'db4'):
    ycc_recovered =[]

    for i,channel in enumerate(ycc_compressed):
    
    
    
        level = levels[i]
        dequantized_dqt =dequantize(channel, 2**level)

        compressed_channel = idwt2(dequantized_dqt , levels=level,wavelet = wavelet)
        
        
        ycc_recovered.append(compressed_channel)


        
    rec_ycc = np.zeros(og_shape)
    rec_ycc[:,:,0] = ycc_recovered[0]
    rec_ycc[:,:,1] = imresize(ycc_recovered[1],2.0)
    rec_ycc[:,:,2] = imresize(ycc_recovered[2],2.0)

    return rec_ycc 


if __name__ == '__main__':
    main()
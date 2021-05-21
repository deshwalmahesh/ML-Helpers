'''
Note: According to the Github code flow, Sequence of operations used in ImageDataGenerator class for Augmentation is as follows:
apply_transform() -> standardize(x)


Where the Sequence for apply_transform() is as follows:
apply_affine_transform() -> flip_horizontal -> flip_vertical -> brightness_change

The sequence of apply_affine_transform() is as follows:
Rotation -> Width shift -> Heigh shift -> Shear -> Zoom in x direction -> Zoom in y direction



The Sequence of standardize() function is as follows:

preprocessing_function -> rescale -> samplewise_center -> samplewise_std_normalization -> 
  -> featurewise_center -> featurewise_std_normalization -> zca_whitening 


Given the conditions only a few can be applied In Order according to priority after overriding the others such as:

if zca_whitening: featurewise_center == True AND featurewise_std_normalization == False
if featurewise_std_normalization: featurewise_center == True
if samplewise_std_normalization: samplewise_center == True
'''

import numpy as np
import cv2
import skimage.filters as filters
from os import listdir
from os.path import isfile, join
import random
from PIL import ImageEnhance
from PIL import Image as pil_image
from tensorflow.keras.preprocessing.image import apply_affine_transform, apply_brightness_shift, img_to_array, array_to_img


def add_blur(img:[str,np.ndarray],kernel_size:int=5,kind:[str,int]='motion_h')->np.ndarray:
    '''
    Method to add different type of blurs to an image
    args:
        img: Path or the numpy array of image
        kernel_size: Size of the kernel to convolve. Directly dependent on the strength of the blur
        kind: Type of blurring to use. Can be any from ['horizontal_motion','motion_v','average','gauss','median']
    '''
    assert (kernel_size % 2 != 0), "kernel_size should be a positive odd number >= 3 " # required for most so declaring it common for all
    
    if isinstance(img,str):
        img = cv2.imread(img)
    
    blurs = ['motion_h','motion_v','average','gauss']
    if isinstance(kind,int):
        kind = blurs[kind]
        
    if kind == 'motion_h':
        kernel_h = np.zeros((kernel_size, kernel_size))  # horizontal kernel
        kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
        kernel_h /= kernel_size 
        return cv2.filter2D(img, -1, kernel_h) 
 
    elif kind == 'motion_v':
        kernel_v = np.zeros((kernel_size, kernel_size)) # vertical kernel
        kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
        kernel_v /= kernel_size  # Normalize. 
        return cv2.filter2D(img, -1, kernel_v)
    
    elif kind == 'average': return cv2.blur(img,(kernel_size,kernel_size)) # Works like PIL BoxBlur
   
    elif kind == 'gauss': return cv2.GaussianBlur(img, (kernel_size,kernel_size),0)  



class AddNoise():
    ''''
    Class to addd Noise. Full code is present at https://github.com/deshwalmahesh/OCR-Improvement-UNet-Binarization/blob/main/Binarize.py
    This is for Diagram -> Siamese Task only

    Same results with https://github.com/scikit-image/scikit-image/blob/main/skimage/util/noise.py when used with numpy.float32 images
    '''
    def poisson(self,img:np.ndarray,fact:float=2)->np.ndarray:
        assert True , "Needs implementation"


    def gauss(self,img:np.ndarray,mean:float=0,std:float=3)->np.ndarray: # Additive Noise
        '''
        Add Gaussian Noise to the Image
        args:
            img: Numpy array of image
            mean: Mean of the Gaussian distributio 
            std: Standard Deviation of the distribution
        '''
        noise = np.random.normal(mean, std, img.shape)
        return img + noise

    
    def speckle(self,img:np.ndarray,mean:float=0,std:float=2)->np.ndarray: # Multiplicative Noise
        '''
        Add speckle Noise to the Image. It is Gaussian but Multiplicative in nature
        args:
            img: Numpy array of image
            mean: Mean of the Gaussian distributio 
            std: Standard Deviation of the distribution
        '''
        noise = np.random.normal(mean, std, img.shape)
        return img + img * noise


def apply_random_contrast(img:np.ndarray,factor:float)->np.ndarray:
    '''
    Change the Contrast of an image
    args:
        img: numpy array
        factor: factor to which the contrast has to be changed
    '''
    assert factor != 0 , "factor should Not be equal to 0 else the image will be all black"
    img = array_to_img(img)
    enhancer = ImageEnhance.Contrast(img)
    return img_to_array(enhancer.enhance(factor))


def apply_random_sharpening(img:np.ndarray,factor:float)->np.ndarray:
    '''
    apply random Sharpening effect to an image
    args:
        img: numpy array
        factor: factor to which the sharpening has to be applied. Can be a negative number too
    '''
    img = array_to_img(img)
    enhancer = ImageEnhance.Sharpness(img)
    return img_to_array(enhancer.enhance(factor))


def apply_random_saturation(img:np.ndarray,factor:float)->np.ndarray:
    '''
    apply random Saturation effect to an image
    args:
        img: numpy array
        factor: factor to which theSaturation has to be applied. Can be a Negative number too
    '''
    img = array_to_img(img)
    enhancer = ImageEnhance.Color(img)
    return img_to_array(enhancer.enhance(factor))


def flip_image(x:np.ndarray,method:str)->np.ndarray:
    '''
    Flip the Image Horizontally or Vertically
    args:
        x: Numpy image array 
        method: which axis to rotate. 'h' for horizontal or 'v' for vertical
    '''
    if method == 'h':
        axis = 1
    else: axis = 0

    x = x.swapaxes(axis, 0)
    x = x[::-1, ...]
    return x.swapaxes(0, axis)

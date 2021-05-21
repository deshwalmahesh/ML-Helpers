# Basic Imports
import numpy as np
from skimage.util import random_noise
import os
from os import listdir
from os.path import isfile, join
import cv2
import sys
import argparse
from functools import partial
from multiprocessing import Pool
import logging
import time
from PIL import Image
import imutils
import random


# Command line argument handling
arg_parser = argparse.ArgumentParser(allow_abbrev=True,  description='Add random noise to images',)

arg_parser.add_argument('-d_inp','--DIR_INP',required=True,
                       help='Directory from where images will be loaded')

arg_parser.add_argument('-d_out','--DIR_OUT',required=True,
                       help='Directory where noisy images will be stored')

arg_parser.add_argument('-cnt','--COUNT',type=int,default=4000,
                       help='How many images to process')

arg_parser.add_argument('-lt','--log_type',type=str,default='display',
                       help='Error logging method. Write in a file or display. file/print')

args = vars(arg_parser.parse_args())

# Get Argument values
INP_DIR         = args['DIR_INP'] 
OUT_DIR         = args['DIR_OUT']
cnt             = args['COUNT']
log_type        = args['log_type']

# Logging error file
if log_type == 'file':
    print('\nNOTE:Error logs are stored in the app.log file\n')
    logging.basicConfig(filename='app.log', filemode='a',format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt='%m/%d/%Y %I:%M:%S %p') # logger config

# Choices of noise, blurs & contrast variation
available_noises = ["gauss", "random", "s&p", "poisson", "speckle"]
available_blurs  = ["vertical motion blur", "horizontal motion blur", "average blur", "gauss blur", "median blur", "bilateral blur"]
available_conts  = ["cont_bright"]


def find_biggest_contour(image):

    # Copy to prevent modification
    image = image.copy()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours are found then we return simply return nothing
    if(len(contours)==0):
        return -1,-1,-1

    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    # Empty image mask with black background
    mask = np.zeros(image.shape, np.uint8)
    # Applying the largest contour on the empty image of zeros
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return mask



def noisy(noise_typ,image):
    '''
    Noise Operations ---------------------------------------------------------
        image       :  The matrix on which operations will be performed
        noise_typ   :  The specific operation to perform
    '''
   
    # Additive Noise
    if noise_typ == "gauss": 
        row,col,ch= image.shape # H, W, C
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy_img = image + gauss
        return noisy_img
   
    # Random Noise
    if noise_typ == "random":
        row,col,ch= image.shape # H, W, C
        uniform = np.random.uniform(0,1,(row,col,ch))
        uniform = uniform.reshape(row,col,ch)
        noise_img = image + uniform
        return noise_img

    # Ceil & floor on pixels
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5          # % of pixels to perform salt effect
        amount = 0.02        # % of pixels to effect
        out = np.copy(image)

        # Salt mode 
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0

        return out
    
    # Multiplicative Noise
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy_img = np.random.poisson(image * vals) / float(vals)
        return noisy_img

    # Multiplicative Noise
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy_img = image + image * gauss
        return noisy_img
            
def blur(blur_typ,image):
    '''
    # Blur Operations ---------------------------------------------------------
        image       :  The matrix on which operations will be performed
        blur_typ    :  The specific blur operation to perform
    '''

    if blur_typ=="vertical motion blur":
        
        kernel_size = 5
        kernel_v = np.zeros((kernel_size, kernel_size)) 
        kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
        kernel_v /= kernel_size 
        noisy_img = cv2.filter2D(image, -1, kernel_v) 
        return noisy_img
    
    elif blur_typ=="horizontal motion blur":

        kernel_size = 5
        kernel_h = np.zeros((kernel_size, kernel_size)) 
        kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
        kernel_h /= kernel_size 
        noisy_img = cv2.filter2D(image, -1, kernel_h) 
        return noisy_img
    
    elif blur_typ=="average blur":
        noisy_img = cv2.blur(image,(5,5))
        return noisy_img

    elif blur_typ=="gauss blur":
        noisy_img = cv2.GaussianBlur(image,(5,5),0)
        return noisy_img

    elif blur_typ=="median blur":
        tmp = np.float32(image)
        noisy_img = cv2.medianBlur(tmp,5)
        return noisy_img   
    
    elif blur_typ=="bilateral blur":
        tmp = np.float32(image)
        noisy_img = cv2.bilateralFilter(tmp,5,75,75)
        return noisy_img
    
def saturation(sat_typ,image):
    '''
    # Saturation Operation ----------------------------------------------------
        image       :  The matrix on which operations will be performed
        sat_typ     :  We only have single saturation operation currently
    '''

    if sat_typ =="cont_bright":
        aplha = 1.2
        bias = 0.3
        noisy_img = aplha*image+bias
        return noisy_img



def randomTransformations(image):
    '''
    # Transformation Operation ----------------------------------------------------
        image       :  The matrix on which operations will be performed

    '''

    # Rotation
    if(np.random.randint(0,2)):
        rows,cols,ch  = image.shape
        check = 1
        angle = np.random.randint(10,25)
        v = np.random.randint(0,2)
        if(v == 1):
            angle *=-1
        image = imutils.rotate_bound(image, angle)
        
        uint_img = (image*255).astype('uint8')
        g = cv2.cvtColor(uint_img, cv2.COLOR_BGR2GRAY)
       

        m1 = find_biggest_contour(g) 
        m2 = cv2.bitwise_not(m1, mask = None)
        bg = cv2.cvtColor(m2,cv2.COLOR_GRAY2RGB)
        image = image + bg

    # Width Shift
    if(np.random.randint(0,2)):
        rows,cols,ch  = image.shape
        check = 1
        M = np.float32([[1,0,5],[0,1,0]])
        image = cv2.warpAffine(image,M,(cols,rows))

    # Height Shift
    if(np.random.randint(0,2)):
        rows,cols,ch  = image.shape
        check = 1
        M = np.float32([[1,0,0],[0,1,5]])
        image = cv2.warpAffine(image,M,(cols,rows))

    # Image Scaling
    if(np.random.randint(0,2)):
        rows,cols,ch  = image.shape
        scale_x = random.uniform(0.5,1)
        scale_y = random.uniform(0.5,1)
        image = cv2.resize(image,None,fx = scale_x, fy = scale_y,interpolation=cv2.INTER_CUBIC)
        return image
    
    # Shearing
    if(np.random.randint(0,2)):  
        rows,cols,ch  = image.shape
        M = np.float32([[1, 0, 0], [0.2, 1, 0]])
        M[0,2] = -M[0,1] * cols/2
        M[1,2] = -M[1,0] * rows/2
        image = cv2.warpAffine(image, M, (cols, rows))

    return image


# To get a count for already existing images in noise folder to avoid name clash
indices = [int(f.split('.')[0]) for f in listdir(OUT_DIR) if isfile(join(OUT_DIR, f))] 
total_images_already = len(indices)

# Naming index
start = 0
if len(indices):
    start = max(indices)+1 # set strating index

print(f'''Already *{total_images_already}* noisy images present in the directory -{OUT_DIR}-''')
onlyfiles = [f for f in listdir(INP_DIR) if isfile(join(INP_DIR, f))]

start_time = time.time()

# Limit the number of files to process
ctr = 0

for file_name in onlyfiles:

    # Reading Image
    file_name = INP_DIR+"/"+file_name
    image = cv2.imread(file_name)
    image = image/255
    
    # Random selection of operations
    noise_select = np.random.randint(0,2)
    blur_select  = np.random.randint(0,2)
    sat_select   = np.random.randint(0,2)
    rand_select  = np.random.randint(0,2)

    noise_img = np.copy(image)

    if(noise_select):
        noise_type = available_noises[np.random.randint(0,len(available_noises))]
        noise_img  = noisy(noise_type,noise_img)

    if(blur_select):
        blur_type  = available_blurs[np.random.randint(0,len(available_blurs))]
        noise_img  = blur(blur_type,noise_img)

    if(sat_select):
        noise_img = saturation("cont_bright",noise_img)
    
    if(rand_select):
        noise_img = randomTransformations(noise_img)

    #cv2.imshow("IMG", noise_img)
    #cv2.waitKey()
    
    save_file_name = OUT_DIR+"/"+str(start)+".png"
    
    noise_img = noise_img*255

    cv2.imwrite(save_file_name,noise_img)

    if(ctr == cnt):
        break
    start+=1
    ctr+=1


sys.exit(f'Processing Completed in *{round(time.time()-start_time,3)}* secs. Exiting script....') # end of program




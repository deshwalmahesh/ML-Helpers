import numpy as np
import cv2
import skimage.filters as filters
from os import listdir
from os.path import isfile, join
import random


def has_blur(image_path:str,thresh:float)->bool:
    '''
    Use Laplacian Variance to find if an image has blur or not. It is very critical to find the threshold and is vert data specific
    args:
        image_path: path of the image
        thresh: threshold to find the find. If the Laplacian Variance has value Lower than that threshold, it has blur
    '''
    gray = cv2.imread(image_path,0)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < thresh:
        return True
    return False


def image_colorfulness(image_path:str,thresh:float,alpha:float=0.5,beta:float=0.3)->bool:
    '''
    Find the image colorfulness. Idea given at PyimageSearch. Uses a threshold to find the colorfulness. threshold is critical and data dependent
    We'll use it as a very basic spam classifier.
    NOTE: It'll fail if the incoming question image is not from a B&W book and has colorful texts and diagrams
    args:
        image_path: Path of the image
        thresh: Threshold of the colurfulness. We'll assume that images crossing this threshold are spams
        alpha: floating number in computing the value of  yb  as  (alpha * (R + G) - B)
        beta: number for computing the colorfulness as  (stdRoot + (beta * meanRoot))
    '''
    image = cv2.imread(image_path)
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G) # difference between Red and Green Channel
    yb = np.absolute(alpha * (R + G) - B) # yb = 0.5 * (R + G) - B
    (rbMean, rbStd) = (np.mean(rg), np.std(rg)) # mean and std
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    colorfulness = stdRoot + (beta * meanRoot)
    if colorfulness > thresh:
        return True
    return False


def add_blur(img:[str,np.ndarray],kernel_size:int=23,kind:[str,int]='motion_h')->np.ndarray:
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
    
    blurs = ['motion_h','motion_v','average','gauss','median']
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
    
    elif kind == 'median': return cv2.medianBlur(img,kernel_size) 


class AddNoise():
    def pepper(self,img):
        assert img.shape != 2, 'Image should be Grayscale for Pepper Noise'
        row, col = img.shape
        number_of_pixels = random.randint(333 , 10000) 
        for i in range(number_of_pixels): 
            y_coord=random.randint(0, row - 1) 
            x_coord=random.randint(0, col - 1) 
            img[y_coord][x_coord] = 0 # # Color that pixel to black
        return img 

    
    def salt(self,img):
        assert img.shape != 2, 'Image should be Grayscale for Salt Noise'
        row, col = img.shape
        number_of_pixels = random.randint(333 , 10000)   # Randomly pick some pixels in the image for coloring them white 
        for i in range(number_of_pixels): 
            y_coord=random.randint(0, row - 1) # Pick a random y coordinate 
            x_coord=random.randint(0, col - 1) # Pick a random x coordinate 
            img[y_coord][x_coord] = 255  # Color that pixel to white 
        return img
    
    
    def gauss(self,img):
        gauss = np.random.normal(0,1,img.size)
        gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
        return cv2.add(img,gauss)
    
    
    def speckle(self,img): # Multiplicative Noise
        gauss = np.random.normal(0,1,img.size)
        gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
        return img + img * gauss
    
    
    def add(self,img:[str,np.ndarray],noise_type:[str,int])->np.ndarray:
        '''
        Add Noise to the given input Image
        args:
            image: Input Image or path
            noise_typ: Type of Noise to add. Can be from one of ['random','gauss','salt','pepper','salt_pepper','speckle','poisson'] or the index of this list
            s_p_val: Number of pixels to change to randomly to white or black in Salt & Pepper (s&p) 
        '''
        noises = ['random','gauss','salt','pepper','salt_pepper','speckle','poisson']
        
        if isinstance(image,str):
            img = cv2.imread(image)

        if isinstance(noise_type,int):
            assert(noise_type < len(noises)), "Pass a valid number or string for noise_type"
            noise_type = noises[noise_type]
            
        assert noise_type in noises, "noise_type must be a valid noise type given in docstring"
            
        if noise_type == 'gauss':
            return self.gauss(img)
        
        elif noise_type == "salt": # Salt Noise. convert random pixels to white
            return self.salt(img)

        elif noise_type == 'pepper': # make random pixels as black
            return selt.pepper(img)

        elif noise_type == 'salt_pepper': # do both salt and pepper
            return self.pepper(self.salt(img))

        elif noise_type == "poisson":
            assert False, "Need implementation in OpenCV"
            return self.poisson(img)
        
        elif noise_type == "speckle":
            return self.speckle(img)

        
def has_blur_fft(image:[np.ndarray,str], size=60, thresh=10,):
    '''
    Detect Blurs based of Fast Fourier Transformation
    args:
        image: Image path or array
        size: Radius around the centerpoint of the image for which we will zero out the FFT shift
        thresh: Threshold for determining if the image is blurry or not
    '''
    if isinstance(image,str): # If it is a string, Open the image using function
            image = cv2.imread(image) # read image
     
    if len(image.shape) !=2: # If it is not Grayscale Image then convert
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0)) #center of image
    
    fft = np.fft.fft2(image) # Find 2D FFT of the Image
    fftShift = np.fft.fftshift(fft) # Shift the 0 frequency elements (Top left elements) to the center
    
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0 # make the center elemet as Zero which are low frequency components
    fftShift = np.fft.ifftshift(fftShift) # perform first inverse FFT so that the centeer goes to Top left again 
    recon = np.fft.ifft2(fftShift) # Reconstruct the image by applying 2D Inverse FFT
    
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return mean <= thresh # If mean is less than equal to threshold, return True (has_blur)

        

def has_blur_wavelet(img, threshold):
    '''
    Detect Blurs using Wavelet Transform method. Copied from:
    https://github.com/pedrofrodenas/blur-Detection-Haar-Wavelet/blob/master/blur_wavelet.py
    '''
    
    # Convert image to grayscale
    Y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    M, N = Y.shape
    
    # Crop input image to be 3 divisible by 2
    Y = Y[0:int(M/16)*16, 0:int(N/16)*16]
    
    # Step 1, compute Haar wavelet of input image
    LL1,(LH1,HL1,HH1)= pywt.dwt2(Y, 'haar')
    # Another application of 2D haar to LL1
    LL2,(LH2,HL2,HH2)= pywt.dwt2(LL1, 'haar') 
    # Another application of 2D haar to LL2
    LL3,(LH3,HL3,HH3)= pywt.dwt2(LL2, 'haar')
    
    # Construct the edge map in each scale Step 2
    E1 = np.sqrt(np.power(LH1, 2)+np.power(HL1, 2)+np.power(HH1, 2))
    E2 = np.sqrt(np.power(LH2, 2)+np.power(HL2, 2)+np.power(HH2, 2))
    E3 = np.sqrt(np.power(LH3, 2)+np.power(HL3, 2)+np.power(HH3, 2))
    
    M1, N1 = E1.shape

    # Sliding window size level 1
    sizeM1 = 8
    sizeN1 = 8
    
    # Sliding windows size level 2
    sizeM2 = int(sizeM1/2)
    sizeN2 = int(sizeN1/2)
    
    # Sliding windows size level 3
    sizeM3 = int(sizeM2/2)
    sizeN3 = int(sizeN2/2)
    
    # Number of edge maps, related to sliding windows size
    N_iter = int((M1/sizeM1)*(N1/sizeN1))
    
    Emax1 = np.zeros((N_iter))
    Emax2 = np.zeros((N_iter))
    Emax3 = np.zeros((N_iter))
    
    
    count = 0
    
    # Sliding windows index of level 1
    x1 = 0
    y1 = 0
    # Sliding windows index of level 2
    x2 = 0
    y2 = 0
    # Sliding windows index of level 3
    x3 = 0
    y3 = 0
    
    # Sliding windows limit on horizontal dimension
    Y_limit = N1-sizeN1
    
    while count < N_iter:
        # Get the maximum value of slicing windows over edge maps 
        # in each level
        Emax1[count] = np.max(E1[x1:x1+sizeM1,y1:y1+sizeN1])
        Emax2[count] = np.max(E2[x2:x2+sizeM2,y2:y2+sizeN2])
        Emax3[count] = np.max(E3[x3:x3+sizeM3,y3:y3+sizeN3])
        
        # if sliding windows ends horizontal direction
        # move along vertical direction and resets horizontal
        # direction
        if y1 == Y_limit:
            x1 = x1 + sizeM1
            y1 = 0
            
            x2 = x2 + sizeM2
            y2 = 0
            
            x3 = x3 + sizeM3
            y3 = 0
            
            count += 1
        
        # windows moves along horizontal dimension
        else:
                
            y1 = y1 + sizeN1
            y2 = y2 + sizeN2
            y3 = y3 + sizeN3
            count += 1
    
    # Step 3
    EdgePoint1 = Emax1 > threshold;
    EdgePoint2 = Emax2 > threshold;
    EdgePoint3 = Emax3 > threshold;
    
    # Rule 1 Edge Pojnts
    EdgePoint = EdgePoint1 + EdgePoint2 + EdgePoint3
    
    n_edges = EdgePoint.shape[0]
    
    # Rule 2 Dirak-Structure or Astep-Structure
    DAstructure = (Emax1[EdgePoint] > Emax2[EdgePoint]) * (Emax2[EdgePoint] > Emax3[EdgePoint]);
    
    # Rule 3 Roof-Structure or Gstep-Structure
    
    RGstructure = np.zeros((n_edges))

    for i in range(n_edges):
    
        if EdgePoint[i] == 1:
        
            if Emax1[i] < Emax2[i] and Emax2[i] < Emax3[i]:
            
                RGstructure[i] = 1
                
    # Rule 4 Roof-Structure
    
    RSstructure = np.zeros((n_edges))

    for i in range(n_edges):
    
        if EdgePoint[i] == 1:
        
            if Emax2[i] > Emax1[i] and Emax2[i] > Emax3[i]:
            
                RSstructure[i] = 1

    # Rule 5 Edge more likely to be in a blurred image 

    BlurC = np.zeros((n_edges))

    for i in range(n_edges):
    
        if RGstructure[i] == 1 or RSstructure[i] == 1:
        
            if Emax1[i] < threshold:
            
                BlurC[i] = 1                        
        
    # Step 6
    Per = np.sum(DAstructure)/np.sum(EdgePoint)
    
    # Step 7
    if (np.sum(RGstructure) + np.sum(RSstructure)) == 0:
        
        BlurExtent = 100
    else:
        BlurExtent = np.sum(BlurC) / (np.sum(RGstructure) + np.sum(RSstructure))
    
    return Per, BlurExtent
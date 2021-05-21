import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import cv2
from PIL import Image
import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import math
import cv2



class PreProcess():
    '''
    Class to pre process an image 
    '''
    def load_image(self,PATH):
        return cv2.imread(PATH)

    # get grayscale image
    def get_grayscale(self,image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    # smoothing/filters
    def apply_filter(self,image,kind='median',kernel=np.ones((5,5),np.uint8),**kwargs):
        if kind == 'median':
            return cv2.medianBlur(image,**kwargs)
        
        elif kind == 'gaussian':
            return cv2.GaussianBlur(image,**kwargs)
        
        elif kind == 'blur':
            return cv2.blur(image,**kwargs)
        
        elif kind == 'bilateral':
            return cv2.bilateralFilter(image,**kwargs)
        
        elif kind == 'manual':
            return cv2.filter2D(image, -1, kernel)
        
        
    #thresholding
    def thresholding(self,image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    
    #morphological operations
    def morphology(self,image,kernel=None,iterations=1,kind='dilation'):
        if not kernel:
            kernel = np.ones((5,5),np.uint8)
            
        if kind == 'dilation':
            return cv2.dilate(image, kernel, iterations=iterations)
        
        elif kind == 'erosion':
            return cv2.erode(image,kernel,iterations = iterations)
        
        elif kind == 'opening': #Opening is erosion followed by dilation. Useful in removing noise
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        elif kind == 'closing': # Closing is Dilation followed by Erosion. Useful in closing 
            #small holes inside the foreground objects, or small black points on the object
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        elif kind == 'morph_grad': # difference between dilation and erosion
            return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        
        elif kind == 'top_hat': # difference between input image and Opening
            return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        
        elif kind == 'black_hat': # difference between the closing of the input image and input image
            return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        

    #canny edge detection
    def canny(self,image,t1=100,t2=200):
        return cv2.Canny(image,t1,t2)
    
    
    #skew correction
    def deskew(self,image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    
    #template matching
    def match_template(self,image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 
    
    


def correct_skew(image, delta=1, limit=5):
    '''
    Deskew an image
    '''

    def determine_score(arr, angle):
        '''
        Use Project Profile Method
        '''
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated

    

def auto_brightandcontrast(input_img, channel, clip_percent=1):
    '''
    Method to brighten and adjust the contrast of an image dynamically
    CHECK: https://stackoverflow.com/questions/56388949/i-want-to-increase-brightness-and-contrast-of-images-in-dynamic-way-so-that-the
    '''
    histSize=180
    alpha=0
    beta=0
    minGray=0
    maxGray=0
    accumulator=[]

    if(clip_percent==0):
        #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hist)
        return input_img

    else:
        hist = cv2.calcHist([input_img],[channel],None,[256],[0, 256])
        accumulator.insert(0,hist[0])    

        for i in range(1,histSize):
            accumulator.insert(i,accumulator[i-1]+hist[i])

        maxx=accumulator[histSize-1]
        minGray=0

        clip_percent=clip_percent*(maxx/100.0)
        clip_percent=clip_percent/2.0

        while(accumulator[minGray]<clip_percent[0]):
            minGray=minGray+1

        maxGray=histSize-1
        while(accumulator[maxGray]>=(maxx-clip_percent[0])):
            maxGray=maxGray-1

        inputRange=maxGray-minGray

        alpha=(histSize-1)/inputRange
        beta=-minGray*alpha

        out_img=input_img.copy()

        cv2.convertScaleAbs(input_img,out_img,alpha,beta)

        return out_img
    
    
    
def perfect_binary(img_path,gauss_filter=(95,95),unsharp_radius=1.5,unsharp_amount=1.5):
    '''
    Make a perfect binary image with blackness removed
    args:
        img_path: {str} path of the image
        gauss_gilter: (tuple) filter size
        unsharp_radius: {float} value to use in skimage.filters.unsharp_mask (radius) argument
        unsharp_amount: {float} value to use in skimage.filters.unsharp_mask (amount) argument
    out:
        smooth,division,sharp,thresh: (tuple) 4 different types of cv2 images array
    '''
    # read the image
    img = cv2.imread(img_path)

    # convert to gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # blur
    smooth = cv2.GaussianBlur(gray, gauss_filter, 0)

    # divide gray by morphology image
    division = cv2.divide(gray, smooth, scale=255)

    # sharpen using unsharp masking
    sharp = filters.unsharp_mask(division,radius=unsharp_radius,amount=unsharp_amount,multichannel=False,
                                 preserve_range=False)
    sharp = (255*sharp).clip(0,255).astype(np.uint8)

    # threshold
    thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_OTSU )[1] 

    # save results
    return smooth,division,sharp,thresh
    

    
def CalcBlockMeanVariance(Img,blockSide=21): # blockSide - the parameter (set greater for larger font on image) 
    '''
    https://stackoverflow.com/questions/22122309/opencv-adaptive-threshold-ocr/22127181#22127181
    '''
    I=np.float32(Img)/255.0
    Res=np.zeros( shape=(int(Img.shape[0]/blockSide),int(Img.shape[1]/blockSide)),dtype=np.float)

    for i in range(0,Img.shape[0]-blockSide,blockSide):           
        for j in range(0,Img.shape[1]-blockSide,blockSide):        
            patch=I[i:i+blockSide+1,j:j+blockSide+1]
            m,s=cv.meanStdDev(patch)
            if(s[0]>0.001): # Thresholding parameter (set smaller for lower contrast image)
                Res[int(i/blockSide),int(j/blockSide)]=m[0]
            else:            
                Res[int(i/blockSide),int(j/blockSide)]=0

    smallImg=cv.resize(I,(Res.shape[1],Res.shape[0] ) )    
    _,inpaintmask=cv.threshold(Res,0.02,1.0,cv.THRESH_BINARY);    
    smallImg=np.uint8(smallImg*255)    

    inpaintmask=np.uint8(inpaintmask)
    inpainted=cv.inpaint(smallImg, inpaintmask, 5, cv.INPAINT_TELEA)    
    Res=cv.resize(inpainted,(Img.shape[1],Img.shape[0] ) )
    Res=np.float32(Res)/255    
    return Res

#-----------------------------------------------------------------------------------------------------
# 
#-----------------------------------------------------------------------------------------------------

#     cv.namedWindow("Img")
#     cv.namedWindow("Edges")
#     Img=cv.imread("F:\\ImagesForTest\\BookPage.JPG",0)
#     res=CalcBlockMeanVariance(Img)
#     res=1.0-res
#     Img=np.float32(Img)/255
#     res=Img+res
#     cv.imshow("Img",Img);
#     _,res=cv.threshold(res,0.85,1,cv.THRESH_BINARY);
#     res=cv.resize(res,( int(res.shape[1]/2),int(res.shape[0]/2) ))



'''
# uses SIFT/SURF method
def deskew():
    im_out = cv2.warpPerspective(skewed_image, np.linalg.inv(M), (orig_image.shape[1], orig_image.shape[0]))
    plt.imshow(im_out, 'gray')
    plt.show()

orig_image = cv2.imread(r'image.png', 0)
skewed_image = cv2.imread(r'imageSkewed.png', 0)

surf = cv2.xfeatures2d.SURF_create(400)
kp1, des1 = surf.detectAndCompute(orig_image, None)
kp2, des2 = surf.detectAndCompute(skewed_image, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good
                          ]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good
                          ]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # see https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html for details
    ss = M[0, 1]
    sc = M[0, 0]
    scaleRecovered = math.sqrt(ss * ss + sc * sc)
    thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
    print("Calculated scale difference: %.2f\nCalculated rotation difference: %.2f" % (scaleRecovered, thetaRecovered))

    deskew()

else:
    print("Not  enough  matches are found   -   %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None
'''

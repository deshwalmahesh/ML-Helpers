import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
from cv2 import resize as cv2_resize
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 32


def plot_image_from_char(char,font_path,width=100,height=100):
    '''
    Function to convert an image from a given character
    Input args:-
        char: any character in any language
        font_path: full path of your .tty or any other format font file
        width: Desired width of the output image
        height: desired Height of the output image
    Output:-
        image:- an image of specified WIDTH and HEIGHT
    '''

    image_init = Image.new('RGB', (width, height))
    draw_obj = ImageDraw.Draw(image_init)
    custom_font = ImageFont.truetype(font_path,90)
    w, h = draw_obj.textsize(char, font=custom_font)
    draw_obj.text(((width - w) / 2,(height - h) / 3), char, font=custom_font)

    return image_init


def resize_image(df_in,curr_w,curr_h,res_w,res_h,lib='cv2'):
    '''
    Resizes all the images in a Dataframe where the columns represent the pixels of GRAYSCALE image
    and returns the resized DataFrame
    Input:
        df_if: Dataframe whose columns represents the pixels of image
        curr_w,curr_h: Current Width and Height or dimensions of image
        res_w,res_h: Desired Width and Height of the output image
        lib: library to be used for resizing. 'skimage' for 'scikit-image' and default='cv2' 
                for Open CV
    Output:
        resized:
            Dataframe like df_in but with changed Width and Height
    '''
    
    resized = {} # empty dictonary which stores the values of pixels for each image
    if lib=='skimage':
        for i in range(df_in.shape[0]): # iterate through all the images in dataframe
            image = resize(df_in.loc[df_in.index[i]].values.reshape(curr_w,curr_h),(res_w,res_h,1))
            # apply resize transformations on per-row basis 
            resized[df_in.index[i]] = image.reshape(-1)  # reshape accordingly
        resized = pd.DataFrame(resized).T 
        # resizing swaps the rows to columns so Transpose sets to default
    
    else: 
        for i in range(df_in.shape[0]):
            image = cv2_resize(df_in.loc[df_in.index[i]].values.reshape(curr_w,curr_h),(res_w,res_h))
            resized[df_in.index[i]] = image.reshape(-1)
        resized = pd.DataFrame(resized).T 
    
    return resized


def plot_comp(filepath,pixel_df,i,char):
    '''
    Function that plots the comparision between the given Character using the matplotlib 
    and the second one using the pixel of images given for the same in .parquet file

    input: 
        i: index of the image in the dataframes given
        fontpath: path of the font file
        pixel_df: dataframe that contains pixel values of columns
        char: character that you want to plot 
    '''

    fontpath = filepath  # we will set the file path in arguments
    
    print("Char used is: %s" %char)
    fig = plt.figure(figsize=(10,3)) 
    # set the figure size's dimensions to (width=10,height=3). Out 'N' subplots will 
    # acquire the areas this
    # area accordingly each one having same area
    
    ax1 = plt.subplot(121) # first subplot of the 1 row, 2 columns subplots
    ax1.set_title('Image of the Char using Pixels') # set title of the first subplot
    ax1.imshow(pixel_df.iloc[[i],:].values.reshape(137,236),cmap='gray', vmin=0,vmax=255)
    # show the image which is as index *i* of the input DataFrame. As we have a continuous 
    # values of pixels, we have to convert it to the dimensions of the given input already 
    # given to us. So we use (137,236) specified by one who gave us the data. cmap='gray' 
    # depicts Black & White 
    
    ax2 = plt.subplot(122) # second subplot of the 1 row, 2 columns
    font_prop = fm.FontProperties(fname=fontpath) # get the properties of the fonts
    ax2.text(0.15,0.35,char, fontproperties=font_prop,size=75) # plot the text as an image
    ax2.set_title('Image of the Grapheme from Fonts')
    

class CustomDataGenerator(ImageDataGenerator):
    '''
    NOTE: Use this function in your own code where you have batch_size defined as BATCH_SIZE
          else, the batch_size will be 32 as default.
    This class extends the ImageDataGenerator but as the parent class only map 1 class label 
    to each image For example it can only map if a picture of car is black or white but we 
    are trying to map it to N classes so that it can override the default flow() and provide 
    a mapping of a car to color,model,company etc. Specially useful if you have different 
    losses for each class so you have to pass a dict of y_labels
    
    This code's credit goes to - https://github.com/keras-team/keras/issues/12639
    '''
    

    def flow(self,x,y=None,batch_size=BATCH_SIZE,shuffle=True,sample_weight=None,seed=None,
            save_to_dir=None,save_prefix='',save_format='png',subset=None): 
        '''
        Function takes data & label arrays, generates batches of augmented data 
        (#official keras Documents)
        Input:
            x: Flow method looks for Rank-4 numpy array. i.e (number_of_images,width,height,
            channels)
            y: dictonary which maps each picture to its ONE-HOT ENCODES respective classes 
            such as if Image1 is associated to 3 classes in a way ->[0,1,2] and Image2 is 
            associated as [3,4,5] so the y will be as y={'y1':to_categorical([0,3]),
            'y2':to_categorical([1,4])...and so on} 
            others: default settings of parameters in the original flow() method
        Output:
            Just like the default flow(), it'll generate an instance of image array x  but 
            instead of a single y-label/class mapping it'll produce a a dictonary as 
            label_dict that contains mapping of all the classes for that image
        '''

        labels_array = None # all the labels array will be concatenated in this single array
        key_lengths = {} 
        # define a dict which maps the 'key' (y1,y2 etc) to lengths of corresponding 
        # label_array
        ordered_labels = [] # store the ordering in which the labels Y were passed
        for key, label_value in y.items():
            if labels_array is None:
                labels_array = label_value 
                # for the first time loop, it's empty, so insert first element
            else:
                labels_array = np.concatenate((labels_array, label_value), axis=1) 
                # concat each array of y_labels 
                
            key_lengths[key] = label_value.shape[1] 
            # key lengths will be different for different range of classes in each class 
            # due to_categorical ONE-HOT encodings. Ex- some have 2 classes (red,yellow) 
            # but other can have 4 (Audi,BMW,Ferrari,Toyota) so we have to keep track 
            # due to inner working of super().flow()
            ordered_labels.append(key)


        for x_out, y_out in super().flow(x, labels_array, batch_size=batch_size):
            label_dict = {} # final dictonary that'll be yielded
            i = 0 # keeps count of the ordering of the labels and their lengths
            for label in ordered_labels:
                target_length = key_lengths[label]
                label_dict[label] = y_out[:, i: i + target_length] 
                # Extract to-from the range of length of labels values. That is why we had 
                # ordered_labels and key_lengths It'll extract the elements ordering vise 
                # else there will be conflict
                i += target_length

            yield x_out, label_dict
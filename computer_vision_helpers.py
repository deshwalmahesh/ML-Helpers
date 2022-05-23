def overlay_mask(image:(np.ndarray, Image.Image), mask:(np.ndarray, Image.Image), return_comparison: bool = False)-> Image.Image:
    '''
    Overlay Mask over Image. Smoothes the image boundry
    image: RGB Numpy array
    mask: Grayscale B/W mask
    return_comparison: whether to return side by side images for original and new
    '''
    if isinstance(image, Image.Image): image = np.array(image)
    if isinstance(mask, Image.Image): mask = np.array(mask)

    mask = mask/ 255.

    # obtain predicted foreground
    if len(image.shape) == 2: # If grayscale, add empty dimension
        image = image[:, :, None]

    if image.shape[2] == 1: # If grayscale with empty dimension, repeat to give the image a RGB look
        image = np.repeat(image, 3, axis=2)

    elif image.shape[2] == 4: # if RGBA, Pick first 3 Dimensions
        image = image[:, :, 0:3]

    if len(mask.shape) == 2:
        mask = np.repeat(mask[:, :, None], 3, axis=2) / 255
        
    foreground = image * mask + np.full(image.shape, 255) * (1 - mask) # From the paper ModNet, Colab Demo

    if return_comparison: # combine image, foreground, and alpha into one line
        combined = np.concatenate((image, foreground), axis=1)
        return Image.fromarray(np.uint8(combined))

    return foreground.astype(np.uint8)


def resize_image(image:(np.ndarray, Image.Image, str,BytesIO), new_width_height:(int,tuple) = 1920, convert_RGB:bool = True)-> Image.Image:
  '''
  Resize and return Given Image
  args:
    path: Image object
    new_width_height = Reshaped image's width and height. # If integer is given, it'll keep the aspect ratio as it is by shrinking the Bigger dimension (width or height) to the max of new_width_height  and then shring the smaller dimension accordingly 
    save_image = Whether to save the image or not
    convert_RGB: Whether to Convert the RGBA image to RGB (by default backgroud is white)
  '''
  if isinstance(image, np.ndarray): image = Image.fromarray(image)
  elif isinstance(image, (str,BytesIO)): image = Image.open(image)
  
  w, h = image.size
  fixed_size = new_width_height if isinstance(new_width_height, int) else False

  if fixed_size:
    if h > w:
      fixed_height = fixed_size
      height_percent = (fixed_height / float(h))
      width_size = int((float(w) * float(height_percent)))
      image = image.resize((width_size, fixed_height), Image.NEAREST)

    else:
      fixed_width = fixed_size
      width_percent = (fixed_width / float(w))
      height_size = int((float(h) * float(width_percent)))
      image = image.resize((fixed_width, height_size), Image.NEAREST) # Try Image.ANTIALIAS inplace of Image.NEAREST

  else:
    image = image.resize(new_width_height)

  if image.mode == "RGBA" and convert_RGB:
    new = Image.new("RGBA", image.size, "WHITE") # Create a white rgba background
    new.paste(image, (0, 0), image) # Paste the image on the background.
    image = new.convert('RGB')

  return image


def overlay_with_mask_processing(orig_image:[np.ndarray, Image.Image], mask:[np.ndarray, Image.Image], convert_to_RGB:bool = True)->Image.Image:
    '''
    Overlay image using some post-processing tricks on Mask to get smoother boundaries
    args:
        orig_image: Actual Image numpy array
        mask: Binary mask
        convert_to_RGB: Whether to convert the RGBA to RGB with WHITE background
        return_array: Whether to return Image.Image object or numpy array
    '''
    if isinstance(orig_image, Image.Image): orig_image = np.array(orig_image)

    if isinstance(mask, Image.Image):
        if mask.mode != "L": mask = mask.convert("L")
        mask = np.array(mask)

    if len(mask.shape) != 2 or mask.shape[-1] != 1:
        assert True, "Mask must be Binary with 1 channel"

    # apply morphology to remove isolated extraneous noise || use borderconstant of black since foreground touches the edges
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # anti-alias the mask -- blur then stretch blur alpha channel
    mask = cv2.GaussianBlur(mask, (3,3), sigmaX = 2, sigmaY = 2, borderType = cv2.BORDER_DEFAULT)

    # linear stretch so that 127.5 goes to 0, but 255 stays 255
    mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)

    new_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2RGBA)
    new_image[:,:,3] = mask
    
    new_image = Image.fromarray(new_image)

    if convert_to_RGB:
        new = Image.new("RGBA", new_image.size, "WHITE") # Create a white rgba background
        new.paste(new_image, (0, 0), new_image) # Paste the image on the background.
        new_image = new.convert('RGB')

    return new_image


def superimpose_background(foreground, background, alpha, overlay:bool = False):
    '''
    Change the background of any image given it's binary mask
    args:
        foreground: RGB image which has to be superimposed
        backgroud: Background of same shape
        alpha: B&W Mask image 
        overlay: Overlay background first
    '''
    if overlay:
        foreground = overlay_mask(foreground, alpha)
          
    foreground = (alpha * foreground) # multiplying black pixels i.e 0 with image makes the background black for the image and keeps the foreground object as it is (x by white == 1)
    
    background = (1.0 - alpha) * background # now this makes the pixels in the background black in the shape of foreground object
    return (foreground + background).astype(np.uint8) # trick? Simple maths. black pixels == 0 in background gets added to white pixels pixels in the foregroud and vice versa -> no effect except blending


def open_image_with_exif(path):
    '''
    When there is an exif tag present in the image, the loaded image gets rotated.

    Solution:
    https://stackoverflow.com/questions/4228530/pil-thumbnail-is-rotating-my-image
    https://github.com/python-pillow/Pillow/issues/4703
    '''
    try :
        image=Image.open(path)
        for orientation in ExifTags.TAGS.keys() : 
            if ExifTags.TAGS[orientation]=='Orientation' : break 
        exif=dict(image._getexif().items())

        if   exif[orientation] == 3 : 
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6 : 
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8 : 
            image=image.rotate(90, expand=True)
        
        return image

    except:
        traceback.print_exc()

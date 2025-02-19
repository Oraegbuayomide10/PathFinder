import numpy as np
from PIL import Image
import os
import json
from tqdm import tqdm
import glob
from typing import List



def cvtColor(image):
    """
        Converts the image to RGB format if the ordering has been changed as cv2 was used. 
        cv2 is known to change the ordering of channels, to resolve this we use this function
    
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 



def preprocess_input(image):
    """
        Normalises the input image using the mean and standard deviation values of the pretrained model
    
    """
    image -= np.array([123.675, 116.28, 103.53], np.float32)
    image /= np.array([58.395, 57.12, 57.375], np.float32)

    return image




def resize_image(image, size):
    """
    Description
        The resize_image function resizes an input image to fit within a specified size,
          while maintaining the original aspect ratio. The resized image is then centered 
          within a new image of the target size, with padding filled in a specified color 
          (default: gray (128,128,128)).


    Parameters
        image:A PIL.Image object representing the image to be resized.
        size:A tuple (width, height) specifying the target dimensions of the new image
    """
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh



def load_image_array(image_path):
    jpg = Image.open(image_path)
    return np.array(jpg)












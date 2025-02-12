import numpy as np
from PIL import Image
import os
import json
from tqdm import tqdm

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


    # deepglobe, spacenet and WHU mean and std
    # image -= np.array([93.842, 94.793, 84.236], np.float32)
    # image /= np.array([35.962, 30.410, 28.513], np.float32)

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


def compute_img_mean_std(images_path):

    channel_means = np.zeros(3)  # Blue, Green, Red, NIR, SWIR
    channel_stds= np.zeros(3)  # Variance instead of std (std computed later)

    for img_path in tqdm(images_path):
        img = load_image_array(str(img_path))
        for i in range(3):  # Iterate over the 5 channels
            channel_means[i] += img[:, :, i].mean()
            channel_stds[i] += (img[:, :, i]).std()  # Accumulate squared values for variance

    # Finalize mean and standard deviation calculations
    channel_means /= len(images_path)
    channel_stds /= len(images_path)

    return channel_means, channel_stds



def saves_computed_stats(mean_channels, std_channels, dir):
    blue_mean, green_mean, red_mean = mean_channels
    blue_std, green_std, red_std = std_channels
    # dictionary
    channel_stats = {
        "means": {
            "blue_mean": blue_mean,
            "green_mean": green_mean,
            "red_mean": red_mean
        },
        "stds": {
            "blue_std": blue_std,
            "green_std": green_std,
            "red_std": red_std
        }
    }

    # Save to a JSON file
    output_file = r"channel_statistics.json"
    output_path = os.path.join(dir, output_file)
    with open(output_path, "w") as f:
        json.dump(channel_stats, f, indent=4)
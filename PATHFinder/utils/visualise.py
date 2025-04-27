import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
import numpy as np



def visualize_batch_predictions(all_predictions, all_labels, save_dir):

    all_predictions, all_labels = extend_preds(all_predictions, all_labels)

    for index in range(4):
        pred_image = all_predictions[index]
        label = all_labels[index] 
        array = pred_image.astype(np.uint8)
        image = Image.fromarray(array * 255)
        combine_images_matplotlib(image, label, save_dir + f'/prediction_{index+1}.png')



def combine_images_matplotlib(img1, img2, save_dir):
    """
        Fix two images side by side and save to `save_dir`
    """
    # Create a subplot with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 

    # Display images side by side
    axes[0].imshow(img1)
    axes[0].axis('off')  # Hide axes
    axes[0].set_title('Predicted')
    axes[1].imshow(img2)
    axes[1].axis('off')  # Hide axes
    axes[1].set_title('Original')

    # Save the output image
    plt.subplots_adjust(wspace=0.01, hspace=0) 
    plt.savefig(save_dir, bbox_inches='tight', pad_inches=0)


def extend_preds(batches_predictions, batches_labels):
    all_preds, all_labels = [], [] 

    # for batches_predictions   
    for i in range(len(batches_predictions)):
        all_preds.append(batches_predictions[i])
           

    # for batches_labels   
    for i in range(len(batches_labels)):
        all_labels.append(batches_labels[i])

    return all_preds, all_labels
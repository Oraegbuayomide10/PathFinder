import gdown
import os
from typing import Literal
import subprocess
import argparse

def Download_Weights(
                     model: Literal['PATHFinder', 'SegFormer'] = 'PATHFinder'

                    ):

    """
    Downloads the pretrained model weights from Google Drive and saves it to the a location (weights folder).

    Parameters:
        model (str): The name of the pretrained model's weight to download.
        output_directory (str): Path where the model's weight would be saved.
    """

    # get root dir of git-repo
    root_dir = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()

    # download model weight for PATHFinder
    if model.lower() == 'pathfinder':
        os.makedirs(os.path.join(root_dir, 'weights'), exist_ok=True)
        gdown.download(
            url = 'https://drive.google.com/uc?id=10i5jFbHffoh0FWs61DduBKZV6DqmmwiY',
            output = os.path.join(root_dir, 'weights', 'PATHFinder.pth')
        )

    elif model.lower() == 'mit_b5':
        os.makedirs(os.path.join(root_dir, 'weights'), exist_ok=True)
        gdown.download(
            url = 'https://drive.google.com/uc?id=1gvq5WJmVtoG0qYLLBrP-GyNVJEuXjgRx',
            output = os.path.join(root_dir, 'weights', 'mit_b5.pth')
        )

    else:
       raise ValueError('The specified model is not supported. Choose either "PATHFinder" or "SegFormer"')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download pretrained model weights.')
    parser.add_argument(
        '--model',
        type=str,
        choices = ['pathfinder', 'mit_b5'],
        default = 'pathfinder',
        help='The name of the pretrained model\'s weight to download (default: PATHFinder).'
    )

    args = parser.parse_args()

    Download_Weights(model=args.model.lower())



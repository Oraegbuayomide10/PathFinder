import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from segformer import SegFormer
from loss_optim_LR_utils import (get_lr_scheduler, set_optimizer_lr,
                                 weights_init)


from eval import LossHistory, EvalCallback
from dataset_utils import SegmentationDataset, seg_dataset_collate
from train_val import fit_one_epoch
from utils.utils import download_weights, show_config




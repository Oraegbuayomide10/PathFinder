import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from loss_optim_LR_utils import Dice_loss, Focal_Loss, ConLoss, f_score, CE_Loss, MIoU

from torch.amp.grad_scaler import GradScaler as GradScaler
from loss_optim_LR_utils import set_optimizer_lr, get_lr

def train_one_epoch(model, dataloader, optimizer,
                     device, cls_weights, num_classes, scaler, 
                     current_epoch, 
                     dice_loss=True,  focal_loss=False, 
                     lad=0.4, use_f16=True,
                     c3_weight = 0.6,
                     c1_weight = 0.4,
                     
                     ):
    
    '''
    Description
        This function performs one epoch of training for a given model.
        It iterates through the provided dataloader,
        computes the loss for each batch, and updates the model weights using the specified optimizer. 
        The function supports multiple loss types (e.g., Dice Loss, Focal Loss) 
        and includes label adjustment (LAD) as an optional feature.

    Parameters
    model (torch.nn.Module):
    The model to be trained. This should be a PyTorch module with defined forward and backward propagation logic.

    dataloader (torch.utils.data.DataLoader):
    The dataloader that provides batches of training data during the epoch.

    optimizer (torch.optim.Optimizer):
    The optimizer used to update the model's weights based on the computed gradients.

    device (str or torch.device):
    The device on which computations will be performed. Typically "cpu" or "cuda".

    cls_weights (np.array):
    Class weights used to adjust the contribution of each class to the loss function.

    num_classes (int):
    The number of classes in the dataset, used for multi-class loss computation.

    scaler(An instance scaler of GradScaler)
    Helps perform the steps of gradient scaling conveniently.

    dice_loss (bool, optional; default=False):
    If True, uses the Dice Loss in addition to other specified loss functions.

    focal_loss (bool, optional; default=False):
    If True, incorporates Focal Loss for handling class imbalance.

    lad (float, optional; default=0.4):
    Label Adjustment Decay (LAD) value to smooth out labels during training. A higher value indicates stronger smoothing.

    use_f16 (bool, optional; default=True):
    if True, applies mixed-precision (FP16) training, which can speed up training and reduce memory usage
        
    '''

    # set model to train mode
    model.train()
    scaler = GradScaler()
    total_loss = 0
    total_f_score = 0
    total_MIoU = 0

    for iteration, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):

        # do not track gradients for the weights
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)

        imgs, pngs, con_1, con_3, labels = batch 
        imgs, pngs, con_1, con_3, labels, weights = imgs.to(device), pngs.to(device), con_1.to(device), con_3.to(device), labels.to(device), weights.to(device)


        # zero out all gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, r1, r3 = model(imgs)

        # MIoU loss for plot
        MIoU_loss = MIoU(outputs, pngs, num_classes=2).item()

        # Main loss (CE or Focal)
        if focal_loss:
            loss = Focal_Loss(outputs, pngs, cls_weights=weights, num_classes=num_classes)
        else:
            loss = CE_Loss(outputs, pngs, cls_weights=weights, num_classes=num_classes)

        # Dice Loss if specified
        if dice_loss:
            dice = Dice_loss(outputs, labels)
            loss = loss + dice

        # Contrastive losses
        c1_loss = ConLoss(r1, con_1)
        c3_loss = ConLoss(r3, con_3)

        #Total loss with lad factor
        loss = loss + lad * (0.7 * c3_loss + 0.3 * c1_loss) # my own conloss  


        # Backpropagation
        loss.backward()
        optimizer.step()


        total_loss += loss.item()
        total_MIoU += MIoU_loss

    avg_loss = total_loss / (iteration + 1)
    avg_MIoU = total_MIoU / (iteration + 1)
    print(f"Training Loss: {avg_loss:.4f},  MIoU: {avg_MIoU:.4f}, LR: {get_lr(optimizer)}")
    return avg_loss, avg_MIoU







def validate_one_epoch(model, dataloader, device, cls_weights,
                        num_classes, dice_loss=True, focal_loss=False, lad=0.2):

    '''
   Description
        The validate_one_epoch function is responsible for evaluating the performance
        of the trained model over one complete epoch of validation data. 
        It calculates various loss components, including optional Dice loss and 
        Focal loss, and tracks the model's performance in terms of key metrics.

    Parameters:
        model: The model to be evaluated during the validation.
        dataloader: A DataLoader object that provides the validation dataset in batches.
        device: The device (CPU or GPU) where the model and data will be loaded for evaluation.
        weights: The weights to be applied during loss calculation (e.g., for handling class imbalance).
        num_classes: The number of classes in the dataset.
        dice_loss (optional, default: False): If True, the Dice loss is included in the final loss calculation.
        focal_loss (optional, default: False): If True, the Focal loss is included in the final loss calculation.
        lad (optional, default: 0.2): A factor used for the loss adjustment, combining losses such as ConLoss or Dice loss.
        
    '''


    model.eval()
    val_loss = 0
    val_f_score = 0
    val_MIoU = 0

    # compute no-gradients for the class weights
    with torch.no_grad():
        weights = torch.from_numpy(cls_weights)

        for iteration, batch in enumerate(tqdm(dataloader, desc="Validation", leave=False)):
            imgs, pngs, con_1, con_3, labels = batch
            imgs, pngs, con_1, con_3, labels, weights = imgs.to(device), pngs.to(device), con_1.to(device), con_3.to(device), labels.to(device), weights.to(device)
            
            # Forward pass
            outputs, r1, r3 = model(imgs)

            # MIoU loss for plot
            MIoU_loss = MIoU(outputs, pngs, num_classes=2).item()


            # Main loss (CE or Focal)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, cls_weights=weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, cls_weights=weights, num_classes=num_classes)

            # Dice Loss if specified
            if dice_loss:
                dice = Dice_loss(outputs, labels)
                loss += dice


            # Contrastive losses
            c1_loss = ConLoss(r1, con_1)
            c3_loss = ConLoss(r3, con_3)

            # Total loss with lad factor
            loss = loss + lad * (0.7 * c3_loss + 0.3 * c1_loss) # my own conloss  


            val_loss += loss.item()
            val_MIoU += MIoU_loss

    avg_val_loss = val_loss / (iteration + 1)
    avg_val_MIoU = val_MIoU/ (iteration + 1)
    print(f"Validation Loss: {avg_val_loss:.4f}, MIoU: {avg_val_MIoU:.4f}")
    return avg_val_loss, avg_val_MIoU




def train_and_validate(model, train_loader, val_loader,
                        cls_weights,
                        num_epochs,
                        total_epochs,
                        scaler,
                        device, optimizer, num_classes,
                        current_epoch, 
                        save_log_dir,
                        c3_weight,
                        c1_weight,
                        dice_loss=True, focal_loss=False,
                        save_period=5, lad=0.4, use_f16=True,):
    

    '''
    Description
        The train_and_validate function is responsible for both training and evaluating the model.
        It calculates various loss components, including optional Dice loss and 
        Focal loss, and tracks the model's performance in terms of key metrics.

    Parameters:
        model: The neural network model to be trained and validated. This could be any deep learning model (e.g., a classification or segmentation model).
        train_loader:The DataLoader object that provides the training dataset. It yields batches of training data to the model during the training process.
        val_loader:The DataLoader object that provides the validation dataset. It yields batches of validation data for evaluating the model’s performance after each epoch.
        cls_weights:A tensor or array of class weights used during the loss calculation to handle class imbalance. Each class can have a different weight, influencing the loss computation during training.
        num_epochs:The number of epochs for which the model will be trained. An epoch refers to one complete pass through the entire training dataset.
        scaler:A GradScaler object from PyTorch's automatic mixed precision (AMP) toolkit. It is used for scaling gradients in FP16 (16-bit floating point) training to prevent underflows and improve numerical stability.
        device:The device on which the model and data will be loaded and processed. It is typically either a CPU ('cpu') or a CUDA-enabled GPU ('cuda').
        optimizer:The optimizer used for updating the model parameters during training. Examples include torch.optim.Adam, torch.optim.SGD, etc.
        num_classes:The number of output classes in the dataset. For example, in segmentation tasks, this represents the number of pixel categories the model predicts.
        dice_loss (optional, default: True):A boolean flag indicating whether to include Dice loss in the training process. Dice loss is commonly used for segmentation tasks, especially to handle class imbalances in binary or multi-class segmentation.
        focal_loss (optional, default: False):A boolean flag that determines whether to use Focal loss. Focal loss is used to address class imbalance by focusing more on hard-to-classify examples.
        save_period (optional, default: 5):The interval (in terms of epochs) at which the model is saved during training. For instance, if save_period=5, the model will be saved every 5 epochs.
        saved_model_name (optional, default: None):The name or path to the file where the model should be saved after training. If None, the model may not be saved, or the function may decide on a default save location.
            
    '''
    
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {current_epoch+1}/{total_epochs}")

        # Training phase
        train_loss, train_MIoU = train_one_epoch (
            
                                                    model=model, 
                                                    dataloader=train_loader, 
                                                    optimizer=optimizer,
                                                    device=device,
                                                    cls_weights = cls_weights,
                                                    num_classes = num_classes,
                                                    scaler = scaler,
                                                    current_epoch = current_epoch,
                                                    dice_loss=dice_loss, 
                                                    focal_loss=focal_loss, 
                                                    lad=lad, use_f16=use_f16,
                                                    c3_weight = c3_weight,
                                                    c1_weight = c1_weight,
                                                        
                                                    
                                    )

        # Validation phase
        val_loss, val_MIoU = validate_one_epoch(model=model, 
                                                    dataloader=val_loader, 
                                                    device = device,
                                                    cls_weights=cls_weights,
                                                    num_classes=num_classes,
                                                    focal_loss=focal_loss,
                                                    dice_loss=dice_loss,
                                                    lad=lad,
        
                                                    )
            


    return model, train_loss, val_loss, train_MIoU, val_MIoU       
        

    


def print_configurations(**kwargs):
    """
    The show_config function is a utility to display configuration settings in a clean, tabular format. It takes keyword arguments as input and formats them into a structured table with labeled columns for "keys" and "values." This is particularly useful for logging, debugging, or documenting configurations in machine learning or software development workflows
    Parameters:
        kwargs (dict):
        Arbitrary keyword arguments representing configuration keys and their respective values.
        Example: learning_rate=0.001, batch_size=32.
    """
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


import os
import datetime
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from segformer import SegFormer
from loss_optim_LR_utils import get_lr_scheduler, set_optimizer_lr, weights_init
from segformer import SegFormer

from eval_utils import LossHistory
from dataset_utils import SegmentationDataset, seg_dataset_collate
from train_val_utils import train_and_validate
from typing import Literal
from train_val_utils import print_configurations
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import glob
from data_utils import compute_img_mean_std, saves_computed_stats
# from utils.utils import download_weights, show_configc


class CFG:
    seed = 42
    random_state = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
seed_everything(CFG.seed)


def Train_validate_model(
                    images_directory: str,
                    labels_directory: str,
                    device='cuda',
                    use_f16=True,
                    num_classes=2,
                    phi: Literal["b0", "b1", "b2", "b3", "b4", "b5"] = "b0",
                    pretrained=True,
                    image_input_shape=[512, 512],
        
                    train_percent = 0.92,
                    unfreeze_epoch = 50, # Unfreeze epoch could also be zero if freeze_train is false, this would mean that the model was unfrozen throughout the training and all layers were trained immediately so the training starts from zero
                    total_epochs=100,
                    freeze_batch_size = 16,
                    unfreeze_batch_size =8,
                    freeze_train = True,
                    initial_lr = 1e-5,
                    optimizer_type: Literal['adamW', 'adam'] = "adamW",
                    momentum = 0.9,
                    weight_decay = 1e-2,
                    lr_decay_type = "cos",
                    save_period: int = 5,
                    save_log_dir:str = None,
                    dice_loss = True,
                    focal_loss = False,
                    cls_weights: np.array = np.array([1, 3], dtype=np.float32),
                    model_name = 'model_b2',
                    save_best_model = True,
                    plot_train_val_loss = True,
                    lad=0.4,
                    c3_weight = 0.6, 
                    c1_weight = 0.4,
                    early_stopping_epochs = 10,
                    no_training = True
                    

                    
                    ):
    

    if not pretrained:
        raise ValueError("You are not training with pretrained_model")
    
    else:

        #--------------message

        # loop through all the dirs in the images_directory list
        # all_images =[f for f in os.listdir(dir) if f.endswith('.jpg') or f.endswith('.tif')]

        deepglobe_all = glob.glob(os.path.join(r'C:\WORKS\Master_Thesis\Train_Datasets\DeepGlobe', '*.jpg')) 
        spacenet_k = glob.glob(os.path.join(r'C:\WORKS\Master_Thesis\Train_Datasets\Spacenet\Khartoum', '*', '*.tif'), recursive=True)
        spacenet_p = glob.glob(os.path.join(r'C:\WORKS\Master_Thesis\Train_Datasets\Spacenet\Paris', '*', '*.tif'), recursive=True)
        spacenet_s = glob.glob(os.path.join(r'C:\WORKS\Master_Thesis\Train_Datasets\Spacenet\Shanghai', '*', '*.tif'), recursive=True)
        spacenet_v = glob.glob(os.path.join(r'C:\WORKS\Master_Thesis\Train_Datasets\Spacenet\Vegas', '*', '*.tif'), recursive=True)
        whu_all= glob.glob(os.path.join(r'C:\WORKS\Master_Thesis\Train_Datasets\WHU', '*.jpg'))[:12994]

        # pick images for validation
        deepglobe_val = deepglobe_all[:500]
        spacenet_k_val = spacenet_k[:125]
        spacenet_p_val = spacenet_p[:125]
        spacenet_s_val = spacenet_s[:125]
        spacenet_v_val = spacenet_v[:125]
        whu_val = whu_all[:1000]

        # pick images for train
        deepglobe_train = deepglobe_all[500:]
        spacenet_k_train = spacenet_k[125:]
        spacenet_p_train = spacenet_p[125:]
        spacenet_s_train = spacenet_s[125:]
        spacenet_v_train = spacenet_v[125:]
        whu_train = whu_all[1000:]

        # sum all train and validation images into a single list
        train_images = deepglobe_train + spacenet_k_train +  spacenet_p_train + spacenet_s_train +  spacenet_v_train + whu_train
        val_images = deepglobe_val + spacenet_k_val +  spacenet_p_val + spacenet_s_val +  spacenet_v_val + whu_val



        #-----------------------------finetuning on UN dataset

        # caf_all = glob.glob(os.path.join(r'C:\WORKS\Master_Thesis\UN_Dataset\CAF_images_masks', '*.tiff'))
        # somalia_all = glob.glob(os.path.join(r'C:\WORKS\Master_Thesis\UN_Dataset\somalia_images_masks', '*.tiff'))

        # random.shuffle(caf_all)
        # random.shuffle(somalia_all)

        # #pick validation images
        # caf_val = caf_all[:100]
        # somalia_val = somalia_all[:60]

        # # pick images for train 
        # caf_train = caf_all[100:]
        # somalia_train = somalia_all[60:]


        # # add all images together
        # train_images = caf_train + somalia_train
        # # train_images = train_images[:60]
        # val_images = caf_val + somalia_val


        all_images = train_images + val_images
        

        # shuffle the images in the list
        random.shuffle(train_images)

    

        if len(all_images) == 0:
            raise ValueError("No images were detected in the image_directory '{image_directory}'.\
                            Make sure images exists within the directory and the images ends with the extension '.jpg'")
        
        #  # -------------- extract train and val images
        # train_images = random.sample(all_images, k=int(train_percent * len(all_images)))
        # val_images = [image for image in all_images if image not in train_images]

        # export the names of the val_images
        with open(os.path.join(save_log_dir, 'val_images.txt'), 'w') as f:
            for item in val_images:
                f.write("%s\n" % item)


        if no_training:
            raise ValueError("You set no_training parameter, so training won't occur")
        
        #------------- message on number of training epochs to ensure enough training
        recommended_steps = 1.5e-4 if optimizer_type == 'adamW' else 0.5e-4
        total_steps = len(train_images) // total_epochs * unfreeze_batch_size
        if total_steps <= recommended_steps:
            if len(train_images) // unfreeze_batch_size == 0 or len(val_images) // unfreeze_batch_size == 0:
                raise ValueError(f'The dataset is too small to train, please expand the dataset.')
            recommend_epochs = ((recommended_steps * unfreeze_batch_size) // len(train_images)) + 1
            print(f"\n\033[1;33;44m[Warning] When using the {optimizer_type} optimizer, it is recommended to set the total training steps to at least {recommended_steps}")
            print(f"\033[1;33;44m[Warning] The total amount of training data for this run is {len(train_images)}, Unfreeze_batch_size is {unfreeze_batch_size}, training for {unfreeze_epoch} epochs, and the calculated total training steps are {total_steps}")
            print(f"\033[1;33;44m[Warning] Since the total training steps are {total_steps}, which is less than the recommended total steps of {recommended_steps}, it is advised to set the total epochs to {recommend_epochs}.\033")
                        



        #--------------- create folder to save model's log
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_log_dir, model_name) # create directory name
        if not os.path.exists(log_dir): # check if it exists
            os.makedirs(log_dir) # if not create it


        #--------------------- loss and f-score history variables
        loss_dict = {'epochs': [],
                              'train_loss': [],
                              'validation_loss': []
        } # dictionary to save train and eval loss history

        MIoU_dict = {'epochs': [],
                              'train_MIoU_score': [],
                              'validation_MIoU_score': []

        } # dictionary to save train and eval f-score history

        #----------------if save model is true 
        if save_best_model:
            # dictionary to save model's weight to then check
            best_loss = float('inf')

            #----------- create folder to save model weights if it does not exists
            weights_dir = os.path.join(log_dir, 'weights')
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir)

            #------------- best weight of model folder
            best_dir =os.path.join(log_dir, 'weights', 'best')
            if not os.path.exists(best_dir):
                os.makedirs(best_dir)


        #--------------if plotting is set to True
        if plot_train_val_loss:
            #--------------- create plot folder if it does not exists
            plots_dir = os.path.join(log_dir, 'plots')
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)



        # early_stopping_countetr
        early_stopping_counter = 0
        last_val_loss = float('inf')


        #--------------learning rate
        min_lr = initial_lr * 0.01

        #---------------- define model
        model = SegFormer(num_classes=num_classes, phi=phi, pretrained=pretrained)
        device= 'cuda'

        # only to be used when loading checkpoint
        checkpoint = torch.load(r"C:\WORKS\Master_Thesis\Codes\logs\PathFinder_B5_finetune\weights\best\best.pth", weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])


        model = model.to(device)



        #----------------f16 
        if use_f16:
            from torch.amp.grad_scaler import GradScaler
            scaler = GradScaler(device=device)
        else:
            scaler = None


        #---------- settings to implement if backbone of model is to be frozen based on training parameters
        if freeze_train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        # ----------- define batch size based on if backbone of model is frozen or not
        batch_size = freeze_batch_size if freeze_train else unfreeze_batch_size


        #------------ define minimum learning rate and initial learning rate using implemented batch size and recommended size
        recommended_batch_size = 16
        lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamW'] else 5e-2
        lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamW'] else 5e-4
        init_lr_fit     = min(max(batch_size / recommended_batch_size * initial_lr, lr_limit_min), lr_limit_max)
        min_lr_fit      = min(max(batch_size / recommended_batch_size * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)


        init_lr_fit = 1e-2
        min_lr_fit = 1e-4
        
        #------------ initialise optimizer's paramaters with learning rate, weight decay and momentum
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'adamW' : optim.AdamW(model.parameters(), init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        # #--setting optimizer for loading checkpoint ---only to be used when loading checkpoint
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f'{init_lr_fit=}')



        #------------ set learning rate scheduler function using learning rate decay defined in training parameters
        lr_scheduler_func = get_lr_scheduler(lr_decay_type= lr_decay_type,
                                             lr=init_lr_fit,
                                              min_lr=min_lr_fit,
                                              total_iters=total_epochs)
        
     
        
        #------------updating message (as the batch size may has changed from unfreeze_batch_size to freeze_batch
        epoch_step_train = len(train_images) // batch_size
        epoch_step_val = len(val_images) // batch_size
        if epoch_step_train == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training. Please expand the dataset.") 
        
        
       

        #------------ Define dataset and dataloader
        train_dataset = SegmentationDataset(images_list= train_images,
                                        input_shape=image_input_shape,
                                        num_classes=num_classes,
                                        train=True,
                                        images_directory=images_directory,
                                        labels_directory=labels_directory
                                        )

        val_dataset = SegmentationDataset(images_list= val_images,
                                        input_shape=image_input_shape,
                                        num_classes=num_classes,
                                        train=False,
                                        images_directory=images_directory,
                                        labels_directory=labels_directory
                                        )

        train_dataloader = DataLoader(dataset=train_dataset,
           batch_size=batch_size,
           shuffle=True,
           collate_fn=seg_dataset_collate,
           num_workers=4,
           pin_memory=True
           )

        val_dataloader = DataLoader(dataset=val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=seg_dataset_collate,
                num_workers=4,
                pin_memory=True
                )
        

        

        #------train and validate
        for epoch in range(0, total_epochs):

            print(f"\n\033[1;33;44m[Note:]{early_stopping_counter=}\033[0m")

            if early_stopping_counter >= early_stopping_epochs + 1 and epoch > unfreeze_epoch:
                print(f"\n\033[1;33;44m[Note:] Early stopping initiated, Training on Unfreeze stopped\033[0m")
                print(f"\n\033[1;33;44m[Note:] Training Stopped!!!\033[0m")
                break



            # check if early_stopping_counter has reached the early_stopping_epoch and training is still in freeze stage
            # then move the training epoch to unfreeze by setting unfreeze epoch to current epoch  
            if early_stopping_counter == early_stopping_epochs and epoch < unfreeze_epoch and freeze_train:
                unfreeze_epoch = epoch
                print(f"\n\033[1;33;44m[Note:] Early stopping initiated, Training moving to Unfreeze\033[0m")


            if epoch == 0 and freeze_train: # if the training has just started and freeze_train was set to True this means that this is the freezed layer first epoch training
                print(f"\n\033[1;33;44m[Note:] Training on frozen backbone layer about to start\033[0m")

                print_configurations(
                                num_of_train_images = len(train_images),
                                num_of_val_images = len(val_images)
                                )
            
            # elif epoch == 0 and not freeze_train:  # if the training has just started and freeze_train was set to False 
            #     print(f"\n\033[1;33;44m[Note:] Training on the whole backbone layers about to start...\033[0m")
            #     print('')
            
            
            
            # if epoch has gotten to the epoch to unfreeze backbone layers and freezing the backbone layer was set to True in the training parameters
            # then change the batch size to the unfreeze batch size, update the learning rate, and unfreeze the backbone
            if epoch >= unfreeze_epoch and freeze_train or not freeze_train:

                if epoch == unfreeze_epoch and freeze_train or epoch==0 and not freeze_train:
                    early_stopping_counter = 0
                    print(f"\n\033[1;33;44m[Note:] Training on freezed layer is completed!!!\033[0m")
                    print(f"\n\033[1;33;44m[Note:] Training on the whole backbone layers about to start....\033[0m")
                    # training about to start on unfrozen backbone layer
                    print_configurations(    
                                epoch_to_unfreeze_model_layers = unfreeze_epoch,
                                total_training_epochs = total_epochs,
                                freeze_batch_size = freeze_batch_size,
                                unfreeze_batch_size = unfreeze_batch_size
                                )
                    
               

        
                    # update batch size
                    batch_size = unfreeze_batch_size

                    # # for finetuning - do not delete this line of code - this worked
                    # init_lr_fit = 1e-5
                    # min_lr_fit = 1e-6

                    # update learning rates due to batch size
                    recommended_batch_size = 16
                    lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamW'] else 5e-2
                    lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamW'] else 5e-4
                    init_lr_fit     = min(max(batch_size / recommended_batch_size * initial_lr, lr_limit_min), lr_limit_max)
                    min_lr_fit      = min(max(batch_size / recommended_batch_size * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                    init_lr_fit = 1e-5
                    min_lr_fit = 1e-7
                    
                    if not freeze_train:
                        optimizer = {
                            'adam'  : optim.Adam(model.parameters(), init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
                            'adamW' : optim.AdamW(model.parameters(), init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
                            'sgd'   : optim.SGD(model.parameters(), init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
                        }[optimizer_type]

                    
                    
                    # reset learning rate scheduler function
                    lr_scheduler_func = get_lr_scheduler(lr_decay_type= lr_decay_type,
                                                lr=init_lr_fit, # for finetuning
                                                min_lr=min_lr_fit, # for finetuning
                                                total_iters=total_epochs)
                    
                    # unfreeze backbone layers of model
                    for param in model.backbone.parameters():
                        param.requires_grad = True


                    #------------updating message (as the batch size may has changed from freeze_batch_size to unfreeze_batch_size
                    epoch_step_train = len(train_images) // batch_size
                    epoch_step_val = len(val_images) // batch_size
                    if epoch_step_train == 0 or epoch_step_val == 0:
                        raise ValueError("The dataset is too small to continue training. Please expand the dataset.") 
                    

                    #---------- update dataloader with new batch size
                    train_dataloader = DataLoader(dataset=train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    collate_fn=seg_dataset_collate,
                                                    num_workers=4,
                                                    pin_memory=True
                                                )

                    val_dataloader = DataLoader(dataset=val_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        collate_fn=seg_dataset_collate,
                                                        num_workers=4,
                                                        pin_memory=True
                                                )
                
            # set optimizer learning rate
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)


            # train and validate one epoch
            model, train_loss, val_loss, train_MIoU, val_MIoU= train_and_validate(model=model,
                                train_loader=train_dataloader,
                                val_loader=val_dataloader,
                                cls_weights=cls_weights,
                                num_epochs=1,
                                scaler=scaler,
                                device=device,
                                optimizer=optimizer,
                                save_log_dir=save_log_dir,
                                save_period=save_period, 
                                dice_loss=dice_loss, 
                                focal_loss=focal_loss,
                                current_epoch=epoch,
                                num_classes=num_classes,
                                total_epochs=total_epochs,
                                use_f16=use_f16,
                                lad=lad,
                                c3_weight = c3_weight,
                                c1_weight = c1_weight,
                                )
            


            # increase early stopping counter by 1
            early_stopping_counter +=1


            # save last model
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }

            torch.save(checkpoint, f"{best_dir}/last.pth")
            print('\n\033[1;33;44m[Note:]Last Model saved \033[0m')
            print(f'\n\033[1;33;44m[Note:]EarlyStoppingCounter is: {early_stopping_counter} \033[0m')


            # save best model
            if save_best_model:
                if val_loss < best_loss: # if there is a reduction in validation loss
                    best_loss = val_loss

                    # reset early_stopping_counter
                    early_stopping_counter = 0

                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                    }
                    torch.save(checkpoint, f"{best_dir}/best.pth")
                    print('\n\033[1;33;44m[Note:]New best weights found and saved\033[0m')



                        



            #------------losses
            loss_dict['epochs'].append(epoch + 1)
            loss_dict['train_loss'].append(train_loss)
            loss_dict['validation_loss'].append(val_loss)

            #------------ f-score
            MIoU_dict['epochs'].append(epoch + 1)
            MIoU_dict['train_MIoU_score'].append(train_MIoU) 
            MIoU_dict['validation_MIoU_score'].append(val_MIoU) 

        


                        



            # plot train and validation loss and f_score for each save period specified in the function
            if plot_train_val_loss:
                if (epoch + 1) % save_period == 0:
                    # convert dict to dataframe for plotting
                    loss_df = pd.DataFrame(loss_dict) 
                    MIoU_df = pd.DataFrame(MIoU_dict)


                    # for loss
                    fig_loss = px.line(loss_df, x="epochs", y=loss_df.columns,
                                title='Train and Validation Loss Plot', width=1000, height=500)
                    fig_loss.update_layout(xaxis_title='epoch', yaxis_title='loss')
                    save_path = os.path.join(plots_dir, f'loss_plot.html') 
                    fig_loss.write_html(save_path)

                    # for f-score
                    fig_f_score = px.line(MIoU_df, x="epochs", y=MIoU_df.columns,
                                title='Train and Validation MIoU Plot', width=1000, height=500)
                    fig_f_score.update_layout(xaxis_title='epoch', yaxis_title='MIoU')
                    save_path = os.path.join(plots_dir, f'MIoU.html') 
                    fig_f_score.write_html(save_path)



            
# dataloader

import os

import cv2
import numpy as np
import torch
from PIL import Image
from data_utils import preprocess_input, cvtColor
from torch.utils.data.dataset import Dataset
from numba import jit

class SegmentationDataset(Dataset):
    def __init__(self, images_list, input_shape, num_classes, train, images_directory, labels_directory):
        super(SegmentationDataset, self).__init__()
        self.images_list   = images_list
        self.length             = len(images_list)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.images_directory       = images_directory
        self.labels_directory       = labels_directory

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_name = self.images_list[index]
        splitted_name = image_name.split('_')
        mask_name = splitted_name[0] + '_mask.png' 
       

        #-------------------------------#
        #   Read the image from the file
        #-------------------------------#
        jpg         = Image.open(os.path.join(self.images_directory, image_name))
        png         = Image.open(os.path.join(self.labels_directory, mask_name))


        #-------------------------------#
        #   Data Augmentation
        #-------------------------------#
        jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)


        #-------------------------------#
        #   Data Processing (transposing, normalisation for image, conversion to numpy for label)
        #-------------------------------#
        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        png         = np.array(png)
        png[png >= self.num_classes] = self.num_classes-1


        #-------------------------------------------------------#
        # one hot encoding..it is not necessary just testing things
        #-------------------------------------------------------#
        seg_labels  = np.eye(self.num_classes)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes))

        return jpg, png, self.get_con_1(png), self.get_con_3(png), seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    #@jit(nopython=True)
    def get_con_1(self,png):
        img = np.where(png > 0, 1, 0)
        shp = img.shape

        img_pad = np.zeros([shp[0] + 4, shp[0] + 4])
        img_pad[2:-2, 2:-2] = img
        dir_array0 = np.zeros([shp[0], shp[1], 9])

        for i in range(shp[0]):
            for j in range(shp[1]):
                if img[i, j] == 0:
                    continue
                dir_array0[i, j, 0] = img_pad[i, j]
                dir_array0[i, j, 1] = img_pad[i, j + 2]
                dir_array0[i, j, 2] = img_pad[i, j + 4]
                dir_array0[i, j, 3] = img_pad[i + 2, j]
                dir_array0[i, j, 4] = img_pad[i + 2, j + 2]
                dir_array0[i, j, 5] = img_pad[i + 2, j + 4]
                dir_array0[i, j, 6] = img_pad[i + 4, j]
                dir_array0[i, j, 7] = img_pad[i + 4, j + 2]
                dir_array0[i, j, 8] = img_pad[i + 4, j + 4]
        return dir_array0

    #@jit(nopython=True)
    def get_con_3(self,png):
        img = np.where(png > 0, 1, 0)
        shp = img.shape

        img_pad = np.zeros([shp[0] + 8, shp[0] + 8])
        img_pad[4:-4, 4:-4] = img
        dir_array0 = np.zeros([shp[0], shp[1], 9])

        for i in range(shp[0]):
            for j in range(shp[1]):
                if img[i, j] == 0:
                    continue
                dir_array0[i, j, 0] = img_pad[i, j]
                dir_array0[i, j, 1] = img_pad[i, j + 4]
                dir_array0[i, j, 2] = img_pad[i, j + 8]
                dir_array0[i, j, 0] = img_pad[i + 4, j]
                dir_array0[i, j, 1] = img_pad[i + 4, j + 4]
                dir_array0[i, j, 2] = img_pad[i + 4, j + 8]
                dir_array0[i, j, 0] = img_pad[i + 8, j]
                dir_array0[i, j, 1] = img_pad[i + 8, j + 4]
                dir_array0[i, j, 2] = img_pad[i + 8, j + 8]
        return dir_array0

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image   = cvtColor(image)
        label   = Image.fromarray(np.array(label))
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            iw, ih  = image.size
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data      = np.array(image, np.uint8)
        #------------------------------------------#
        #   高斯模糊
        #------------------------------------------#
        blur = self.rand() < 0.25
        if blur: 
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        #------------------------------------------#
        #   旋转
        #------------------------------------------#
        rotate = self.rand() < 0.25
        if rotate: 
            center      = (w // 2, h // 2)
            rotation    = np.random.randint(-10, 11)
            M           = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data  = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128,128,128))
            label       = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label


def seg_dataset_collate(batch):
    images      = []
    con_1 = []
    con_3 = []
    pngs        = []
    seg_labels  = []
    for img, png,c1,c3, labels in batch:
        images.append(img)
        pngs.append(png)
        con_1.append(c1)
        con_3.append(c3)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    con_1 = torch.from_numpy(np.array(con_1))#.long()
    con_3 = torch.from_numpy(np.array(con_3))#.long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs,con_1,con_3, seg_labels
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import os
import albumentations as A
import cv2
import random


# Define parameters

DATA = 'raw_data'
OUTPUT = 'output'
random.seed(2021)
count = 3

def load_data(directory)
# Read in images and masks

imgs = tiff.imread(os.path.join(DATA,'train-volume.tif'))
imgs = imgs.transpose(1,2,0)
imgs = np.squeeze(np.dsplit(imgs, 30))

masks = tiff.imread(os.path.join(DATA,'train-labels.tif'))
masks = masks.transpose(1,2,0)
masks = np.squeeze(np.dsplit(masks, 30))
masks = masks / 255
# //MH maybe turn mask into integer


def get_aug(p=1.0):
    return A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, 
                         border_mode=cv2.BORDER_REFLECT),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.3),
        A.OneOf([
            A.HueSaturationValue(10,15,10),
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(),            
        ], p=0.3),
    ], p=p)

def aug_image(imgs, masks):
    random_number = random.randint(0,29)
    image = imgs[random_number]
    mask = masks[random_number]
    tfms = get_aug()
    augmented = tfms(image=image, mask=mask)
    image, mask = augmented['image'],augmented['mask']
    return image, mask

input_images, target_masks = zip(*[aug_image(imgs, masks) for i in range(0, count)])
input_images = np.asarray(input_images)
target_masks = np.asarray(target_masks)

for x in [input_images, target_masks]:
    print(x.shape)
    print(x.min(), x.max())

for i in range(3):
    plt.imshow(input_images[i])
    plt.show()
    plt.clf()

    plt.imshow(target_masks[i])
    plt.show()
    plt.clf()

#print(image.shape)
#print(mask.shape)
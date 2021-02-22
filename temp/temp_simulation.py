import numpy as np
import tifffile as tiff
import os
import albumentations as A
import cv2
import random


def oned_to_threed(image):
    image = np.expand_dims(image, 0)
    image = image.repeat(3, axis=0)
    return image

def get_aug_train(p=1.0):
    return A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, 
                         border_mode=cv2.BORDER_REFLECT),
        # Update image size to 572 once model structure from original paper is adopted
        A.PadIfNeeded(min_height=576, min_width=576, p=1),
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

def get_aug_test(p=1.0):
    return A.Compose([
        A.PadIfNeeded(min_height=576, min_width=576, p=1),
    ], p=p)

def aug_image(imgs, masks, train):
    # We have thirty images for training data and we randomly pick one
    # for augmentation
    random_number = random.randint(0,29)
    image = imgs[random_number]
    mask = masks[random_number]
    if train:
        tfms = get_aug_train()
    else:
        tfms = get_aug_test()
    augmented = tfms(image=image, mask=mask)
    image, mask = augmented['image'],augmented['mask']
    return image, mask

def reshape_images(imgs_train, masks, train, count):
    """
    Reshape training images into array format required
    by model.
    """
    input_images, target_masks = zip(*[aug_image(imgs_train, masks, train) for i in range(0, count)])
    input_images = np.asarray(input_images)
    target_masks = np.asarray(target_masks)
    # add channel for number of target categories. In this case 1.
    target_masks = np.expand_dims(target_masks, 1)
    return input_images, target_masks
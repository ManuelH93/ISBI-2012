import numpy as np
import tifffile as tiff
import os
import albumentations as A
import cv2
import random

def load_data(directory):
    """
    Read in train images, test images and masks.
    """

    imgs_train = tiff.imread(os.path.join(directory, 'train-volume.tif'))
    imgs_train = imgs_train.transpose(1,2,0)
    imgs_train = np.squeeze(np.dsplit(imgs_train, 30))

    masks = tiff.imread(os.path.join(directory,'train-labels.tif'))
    masks = masks.transpose(1,2,0)
    masks = np.squeeze(np.dsplit(masks, 30))
    masks = masks / 255

    imgs_test = tiff.imread(os.path.join(directory, 'test-volume.tif'))
    imgs_test = imgs_test.transpose(1,2,0)
    imgs_test = np.squeeze(np.dsplit(imgs_test, 30))

    return imgs_train, masks, imgs_test
    
def get_aug(p=1.0):
    return A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, 
                         border_mode=cv2.BORDER_REFLECT),
        A.PadIfNeeded(min_height=572, min_width=572, p=1),
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
    # We have thirty images for training data
    random_number = random.randint(0,29)
    image = imgs[random_number]
    mask = masks[random_number]
    tfms = get_aug()
    augmented = tfms(image=image, mask=mask)
    image, mask = augmented['image'],augmented['mask']
    return image, mask

def reshape_images(imgs_train, masks, count):
    """
    Reshape training images into array format required
    by model.
    """
    input_images, target_masks = zip(*[aug_image(imgs_train, masks) for i in range(0, count)])
    input_images = np.asarray(input_images)
    # add colour channel (in this case greyscale, in case of RGB thee would be 3 channels).
    input_images = np.expand_dims(input_images, 3)
    target_masks = np.asarray(target_masks)
    # add channel for number of target categories. In this case 1.
    target_masks = np.expand_dims(target_masks, 1)

    return input_images, target_masks
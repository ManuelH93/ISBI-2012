import numpy as np
import tifffile as tiff
import os
import albumentations as A
import cv2
import random
import copy

def oned_to_twod(image):
    image = np.expand_dims(image, 0)
    membrane = copy.deepcopy(image)
    # Replace 1s with 0s and 0s with 1s
    indices_one = image == 1
    indices_zero = image == 0
    image[indices_one] = 0
    image[indices_zero] = 1
    mask = np.concatenate((image, membrane), axis=0)
    return mask

def twod_to_oned(image):
    membrane = image[1]
    return membrane

def get_aug(p=1.0):
    return A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, 
                         border_mode=cv2.BORDER_REFLECT),
        A.PadIfNeeded(min_height=572, min_width=572, p=1),
        A.OneOf([
            A.OpticalDistortion(p=0.5),
            A.GridDistortion(p=0.2),
            A.IAAPiecewiseAffine(p=0.5),
        ], p=0.3),
        A.OneOf([
            A.HueSaturationValue(10,15,10),
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(),            
        ], p=0.3),
    ], p=p)

def center_crop(image):
    target_size = 388
    array_size = 572
    delta = array_size - target_size
    delta = delta // 2
    return image[delta:array_size-delta, delta:array_size-delta]

def load_data(directory):
    """
    Read in train images, test images and masks.
    """

    imgs_train = tiff.imread(os.path.join(directory, 'train-volume.tif'))
    imgs_train = imgs_train.transpose(1,2,0)
    imgs_train = np.squeeze(np.dsplit(imgs_train, 30))
    # Add colour dimension for RGB channel
    imgs_train = np.expand_dims(imgs_train,1)
    imgs_train = imgs_train.repeat(3, axis=1).transpose([0, 2, 3, 1]).astype(np.uint8)

    masks_train = tiff.imread(os.path.join(directory,'train-labels.tif'))
    masks_train = masks_train.transpose(1,2,0)
    masks_train = np.squeeze(np.dsplit(masks_train, 30))
    masks_train = masks_train / 255
    # Replace 1s with 0s and 0s with 1s
    indices_one = masks_train == 1
    indices_zero = masks_train == 0
    masks_train[indices_one] = 0
    masks_train[indices_zero] = 1

    imgs_test = tiff.imread(os.path.join(directory, 'test-volume.tif'))
    imgs_test = imgs_test.transpose(1,2,0)
    imgs_test = np.squeeze(np.dsplit(imgs_test, 30))
    # Add colour dimension for RGB channel
    imgs_test = np.expand_dims(imgs_test,1)
    imgs_test = imgs_test.repeat(3, axis=1).transpose([0, 2, 3, 1]).astype(np.uint8)

    # Create dummy masks for code to work
    masks_test = np.zeros((30, 512, 512))

    return imgs_train, masks_train, imgs_test, masks_test
    
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
        # Update image size to 572 once model structure from original paper is adopted
        A.PadIfNeeded(min_height=576, min_width=576, p=1),
    ], p=p)

def aug_image(imgs, masks, train, i):
    # We have thirty images for training data and we randomly pick one
    # for augmentation. In test mode we don't want a random value.
    random_number = random.randint(0,29)
    if train:
        number = random_number
    else:
        number = i
    image = imgs[number]
    mask = masks[number]
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
    input_images, target_masks = zip(*[aug_image(imgs_train, masks, train, i) for i in range(0, count)])
    input_images = np.asarray(input_images)
    target_masks = np.asarray(target_masks)
    # add channel for number of target categories. In this case 1.
    target_masks = np.expand_dims(target_masks, 1)
    return input_images, target_masks
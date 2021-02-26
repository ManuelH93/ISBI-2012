import numpy as np
import albumentations as A
import cv2
import copy
import torch

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

def get_aug_train(p=1.0):
    return A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=360, p=0.9, 
                         border_mode=cv2.BORDER_REFLECT),
        A.PadIfNeeded(min_height=572, min_width=572, p=1),
        A.OneOf([
            A.OpticalDistortion(p=0.7),
            A.GridDistortion(p=0.3),
        ], p=1.0),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(),            
        ], p=0.7),
    ], p=p)

def get_aug_test(p=1.0):
    return A.Compose([
        A.PadIfNeeded(min_height=696, min_width=696, p=1),
    ], p=p)

def center_crop(image):
    target_size = 388
    array_size = 572
    delta = array_size - target_size
    delta = delta // 2
    return image[delta:array_size-delta, delta:array_size-delta]

def dim_plus_two(image):
    image = np.expand_dims(image, 0)
    image = np.expand_dims(image, 0)
    return image

def crop(images):
    all_crops = []
    for image in images:
        image = np.squeeze(image)
        crop_1 = dim_plus_two(image[0:572, 0:572])
        crop_2 = dim_plus_two(image[0:572, 124:696])
        crop_3 = dim_plus_two(image[124:696, 0:572])
        crop_4 = dim_plus_two(image[124:696, 124:696])
        image_crops = np.concatenate((crop_1, crop_2, crop_3, crop_4))
        all_crops.append(image_crops)
    all_crops = np.concatenate(all_crops)  
    all_crops = torch.from_numpy(all_crops.astype(np.float32, copy=False))  
    return all_crops
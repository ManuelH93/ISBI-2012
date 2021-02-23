import numpy as np
import albumentations as A
import cv2


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
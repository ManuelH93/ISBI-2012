import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import os
import albumentations as A
import cv2

# Define parameters

DATA = 'raw_data'
OUTPUT = 'output'

# Read in images and masks

imgs = tiff.imread(os.path.join(DATA,'train-volume.tif'))
imgs = imgs.transpose(1,2,0)
imgs = np.squeeze(np.dsplit(imgs, 30))

masks = tiff.imread(os.path.join(DATA,'train-labels.tif'))
masks = masks.transpose(1,2,0)
masks = np.squeeze(np.dsplit(masks, 30))
masks = masks / 255

print(imgs.shape)
print(masks.shape)

plt.imshow(imgs[0])
plt.show()
plt.clf()


plt.imshow(masks[0])
plt.show()
plt.clf()

image = imgs[0]
mask = masks[0]

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

tfms = get_aug()
augmented = tfms(image=image, mask=mask)
image, mask = augmented['image'],augmented['mask']

plt.imshow(image)
plt.show()
plt.clf()

plt.imshow(mask)
plt.show()
plt.clf()

print(image.shape)
print(mask.shape)






x, y = zip(*[generate_img_and_mask(height, width) for i in range(0, count)])

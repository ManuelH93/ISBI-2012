# Install packages

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import cv2
import os
import zipfile

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

# Plot images and masks

fig=plt.figure(figsize=(24, 20))
for ind, (image, mask) in enumerate(zip(imgs, masks)):
    fig.add_subplot(6, 5, 1+ind)
    plt.axis('off')
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.imshow(mask, cmap="hot", alpha=0.5)
plt.savefig(os.path.join(OUTPUT,'images.png'))
plt.clf()

# Resize images

def resize(image):
    image = np.pad(image,[[30,30],[30,30]], mode = 'reflect')
    return image

imgs_r = []
masks_r = []
for (image, mask) in zip(imgs, masks):
    image = resize(image)
    imgs_r.append(image)
    mask = resize(mask)
    masks_r.append(mask)

# Plot resized images

fig=plt.figure(figsize=(24, 20))
for ind, (image, mask) in enumerate(zip(imgs_r, masks_r)):
    fig.add_subplot(6, 5, 1+ind)
    plt.axis('off')
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.imshow(mask, cmap="hot", alpha=0.5)
plt.savefig(os.path.join(OUTPUT,'resized.png'))
plt.clf()

# Save resized images and masks and print image stats

x_tot,x2_tot = [],[]
with zipfile.ZipFile(os.path.join(OUTPUT,'train.zip'), 'w') as img_out,\
 zipfile.ZipFile(os.path.join(OUTPUT,'mask.zip'), 'w') as mask_out:
    for i,(image,mask) in enumerate(zip(imgs_r,masks_r)):
        x_tot.append((image/255.0).reshape(-1).mean(0))
        x2_tot.append(((image/255.0)**2).reshape(-1).mean(0))
        
        image = cv2.imencode('.png',image)[1]
        img_out.writestr(f'image_{i+1}.png', image)
        mask = cv2.imencode('.png',mask * 255)[1]
        mask_out.writestr(f'image_{i+1}.png', mask)

#image stats
img_avr =  np.array(x_tot).mean(0)
img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
print('mean:',img_avr, ', std:', img_std)
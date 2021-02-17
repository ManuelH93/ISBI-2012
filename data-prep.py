# Install packages

import numpy as np
import tifffile as tiff
import cv2
import os
import zipfile

# Define parameters

DATA = 'raw_data'
OUTPUT = 'processed_data'

# Read in images and masks

train_imgs = tiff.imread(os.path.join(DATA,'train-volume.tif'))
train_imgs = train_imgs.transpose(1,2,0)
train_imgs = np.squeeze(np.dsplit(train_imgs, 30))

test_imgs = tiff.imread(os.path.join(DATA,'test-volume.tif'))
test_imgs = test_imgs.transpose(1,2,0)
test_imgs = np.squeeze(np.dsplit(test_imgs, 30))

masks = tiff.imread(os.path.join(DATA,'train-labels.tif'))
masks = masks.transpose(1,2,0)
masks = np.squeeze(np.dsplit(masks, 30))
masks = masks

# Save train_images and masks and print image stats
x_tot,x2_tot = [],[]
with zipfile.ZipFile(os.path.join(OUTPUT, 'train.zip'), 'w') as img_out,\
 zipfile.ZipFile(os.path.join(OUTPUT, 'masks.zip'), 'w') as mask_out:
    for i,(image,mask) in enumerate(zip(train_imgs,masks)):
        x_tot.append((image/255.0).reshape(-1).mean(0))
        x2_tot.append(((image/255.0)**2).reshape(-1).mean(0))
        
        image = cv2.imencode('.png',image)[1]
        img_out.writestr(f'image_{i+1}.png', image)
        mask = cv2.imencode('.png',mask)[1]
        mask_out.writestr(f'image_{i+1}.png', mask)

# Save test_images
with zipfile.ZipFile(os.path.join(OUTPUT, 'test.zip'), 'w') as img_out:
    for i, image in enumerate(test_imgs):        
        image = cv2.imencode('.png',image)[1]
        img_out.writestr(f'image_{i+1}.png', image)

#image stats
img_avr =  np.array(x_tot).mean(0)
img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
print('mean:',img_avr, ', std:', img_std)
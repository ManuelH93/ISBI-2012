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

imgs = tiff.imread(os.path.join(DATA,'train-volume.tif'))
imgs = imgs.transpose(1,2,0)
imgs = np.squeeze(np.dsplit(imgs, 30))

masks = tiff.imread(os.path.join(DATA,'train-labels.tif'))
masks = masks.transpose(1,2,0)
masks = np.squeeze(np.dsplit(masks, 30))
masks = masks

# Save resized images and masks and print image stats

x_tot,x2_tot = [],[]
with zipfile.ZipFile(os.path.join(OUTPUT, 'train.zip'), 'w') as img_out,\
 zipfile.ZipFile(os.path.join(OUTPUT, 'masks.zip'), 'w') as mask_out:
    for i,(image,mask) in enumerate(zip(imgs,masks)):
        x_tot.append((image/255.0).reshape(-1).mean(0))
        x2_tot.append(((image/255.0)**2).reshape(-1).mean(0))
        
        image = cv2.imencode('.png',image)[1]
        img_out.writestr(f'image_{i+1}.png', image)
        mask = cv2.imencode('.png',mask)[1]
        mask_out.writestr(f'image_{i+1}.png', mask)

#image stats
img_avr =  np.array(x_tot).mean(0)
img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
print('mean:',img_avr, ', std:', img_std)
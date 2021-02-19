import os
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import time
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torchsummary import summary
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F
#import imgaug


from loss import dice_loss
import simulation
import pytorch_unet

###########################################################
# Define parameters
###########################################################

DATA = 'processed_data'
TRAIN = 'train'
MASKS = 'masks'
TEST = 'test'
OUTPUT = 'output'
SEED = 2002
random.seed(SEED)
torch.manual_seed(SEED)
#imgaug.random.seed(SEED)

ids = np.array([f'image_{i}.png' for i in range(1,31)])
random.shuffle(ids)
split = int(0.8 * len(ids))

###########################################################
# Define dataset
###########################################################

class ISBI_Dataset(Dataset):

    def __init__(self, train = True, tfms=None):
        self.fnames = ids[:split] if train else ids[split:]
        self.tfms = tfms
            
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        print(fname)
        img = cv2.imread(os.path.join(DATA,TRAIN,fname), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(DATA,MASKS,fname),cv2.IMREAD_GRAYSCALE)

        if self.tfms is not None:
            augmented = self.tfms(image=img,mask=mask)
            img,mask = augmented['image'],augmented['mask']

        img = img/255.0
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img.astype(np.float32, copy=False))

        mask = mask/255.0
        mask = simulation.center_crop(mask)
        mask = simulation.oned_to_twod(mask)
        mask = torch.from_numpy(mask.astype(np.float32, copy=False))
        
        return img, mask

###########################################################
# Test if dataset load works
###########################################################

train_set = ISBI_Dataset(train=True, tfms=simulation.get_aug_train())
val_set = ISBI_Dataset(train=False, tfms=simulation.get_aug_train())

image_datasets = {
    'train': train_set, 'val': val_set
}

batch_size = 3

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
}

dataset_sizes = {
    x: len(image_datasets[x]) for x in image_datasets.keys()
}

print(dataset_sizes)

# Get a batch of training data
inputs, masks = next(iter(dataloaders['train']))
print(inputs.dtype)
print(masks.dtype)

print(inputs.shape, masks.shape)
for x in [inputs.numpy(), masks.numpy()]:
    print(x.shape)
    print(x.min(), x.max(), x.mean(), x.std())

print('understand input image')
image_l_1 = inputs.numpy()[0,0,:,:]
print(image_l_1.shape)
print(image_l_1[170:180, 150:160])

print('understand mask')

mask_l_1 = masks.numpy()[0,0,:,:]
mask_l_2 = masks.numpy()[0,1,:,:]
print(mask_l_1.shape)
print(mask_l_1[9:19, 330:343])
print(mask_l_2[9:19, 330:343])

# Convert tensors back to arrays

imgs = inputs.numpy()
masks = masks.numpy()
masks = [simulation.twod_to_oned(mask) for mask in masks]

plt.imshow(np.squeeze(imgs[0]), cmap='gray')
plt.show()
plt.clf()
plt.imshow(masks[0], cmap='gray')
plt.show()
plt.clf()
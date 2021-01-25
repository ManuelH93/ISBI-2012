######################################################################
# Import modules
######################################################################

import torch
import torch.nn as nn
import random
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

######################################################################
# Set parameters
######################################################################

SEED = 2021
OUTPUT = 'output'
TRAIN = os.path.join(OUTPUT,'train')
MASKS = os.path.join(OUTPUT,'mask')
mean = np.array([0])
std = np.array([1])

######################################################################
# Set random seeds
######################################################################

random.seed(SEED)

######################################################################
# Define Dataset
######################################################################

def img2tensor(img,dtype:np.dtype=np.float32):
    img = torch.from_numpy(img.astype(dtype, copy=False))
    return img

def crop(image):
    target_size = 388
    tensor_size = 572
    delta = tensor_size - target_size
    delta = delta // 2
    return image[delta:tensor_size-delta, delta:tensor_size-delta]

class IsbiDataset(torch.utils.data.Dataset):
    def __init__(self, ids):
        self.fnames = [fname for fname in os.listdir(TRAIN) if fname.split('.')[0] in ids]
            
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(TRAIN,fname), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(MASKS,fname),cv2.IMREAD_GRAYSCALE)
        img, mask = img2tensor((img/255.0 - mean)/std),img2tensor(mask/255.0)
        img = torch.unsqueeze(img,0)
        mask = crop(mask)
        return img, mask

ids = np.array([f'image_{i}' for i in range(1,31)])
random.shuffle(ids)
cut_off = int(len(ids)*0.8)
train = ids[:cut_off]
val = ids[cut_off:]

# Test if dataset is loading images as expected

ds = IsbiDataset(train)
dl = torch.utils.data.DataLoader(ds,batch_size=2,shuffle=False)
imgs,masks = next(iter(dl))
#print(type(imgs))
#print(imgs.shape)
#print(type(masks))
#print(masks.shape)

fig=plt.figure(figsize=(24, 20))
for ind, (image, mask) in enumerate(zip(imgs, masks)):
    fig.add_subplot(5, 5, 1+ind)
    plt.axis('off')
    # Remove channel dimension
    image = torch.squeeze(image)
    # Crop image to fit mask
    image = crop(image)
    plt.imshow(image*255, cmap='gray', vmin=0, vmax=255)
    plt.imshow(mask, cmap="hot", alpha=0.5)
#plt.savefig(os.path.join(OUTPUT,'train_dataset.png'))
plt.show()
plt.clf()
    
#del ds,dl,imgs,masks
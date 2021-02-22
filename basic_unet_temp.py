import os,sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import temp_helper
import temp_simulation
import random
import torch
import cv2

###########################################################
# Define parameters
###########################################################

DATA = 'processed_data'
TRAIN = 'train'
MASKS = 'masks'
TEST = 'test'
OUTPUT = 'output'
SEED = 2001
random.seed(SEED)
torch.manual_seed(SEED)

ids = np.array([f'image_{i}.png' for i in range(1,31)])
random.shuffle(ids)
split = int(0.8 * len(ids))

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

class ISBI_Dataset(Dataset):

    def __init__(self, train = True, tfms=None):
        self.fnames = ids[:split] if train else ids[split:]
        self.tfms = tfms
            
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(DATA,TRAIN,fname), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(DATA,MASKS,fname),cv2.IMREAD_GRAYSCALE)

        if self.tfms is not None:
            augmented = self.tfms(image=img,mask=mask)
            img,mask = augmented['image'],augmented['mask']

        img = img/255.0
        img = temp_simulation.oned_to_threed(img)
        img = torch.from_numpy(img.astype(np.float32, copy=False))

        mask = mask/255.0
        mask = np.expand_dims(mask, 0)
        
        return img, mask

###########################################################
# Test if dataset load works
###########################################################

#ds = ISBI_Dataset(tfms = temp_simulation.get_aug_train())
#dl = DataLoader(ds,batch_size=4)
#imgs,masks = next(iter(dl))
#print(imgs.shape, masks.shape)
#print(imgs.dtype, masks.dtype)
#for x in [imgs.numpy(), masks.numpy()]:
#    print(x.min(), x.max(), x.mean(), x.std())

# Convert tensors back to arrays
#imgs = imgs.numpy()
#imgs = np.squeeze(imgs)
#imgs = imgs.transpose([0, 2, 3, 1])
#masks = masks.numpy()
#masks= np.squeeze(masks)

#for image, mask in zip(imgs,masks):
#    plt.imshow(image)
#    plt.show()
#    plt.clf()
#    plt.imshow(mask, cmap='gray')
#    plt.show()
#    plt.clf()

train_set = ISBI_Dataset(train=True, tfms=temp_simulation.get_aug_train())
val_set = ISBI_Dataset(train=False, tfms=temp_simulation.get_aug_train())

image_datasets = {
    'train': train_set, 'val': val_set
}

batch_size = 5

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
}

dataset_sizes = {
    x: len(image_datasets[x]) for x in image_datasets.keys()
}

print(dataset_sizes)


import torchvision.utils

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    
    return inp

# Get a batch of training data
inputs, masks = next(iter(dataloaders['train']))

print(inputs.shape, masks.shape)
for x in [inputs.numpy(), masks.numpy()]:
    print(x.min(), x.max(), x.mean(), x.std())

plt.imshow(reverse_transform(inputs[3]))
#plt.show()
plt.clf()


from torchsummary import summary
import torch
import torch.nn as nn
import temp_pytorch_unet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = temp_pytorch_unet.UNet(1)
model = model.to(device)

summary(model, input_size=(3, 576, 576))


from collections import defaultdict
import torch.nn.functional as F
from temp_loss import dice_loss

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))    

def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(OUTPUT, 'bst_unet.model'))
    return model


import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_class = 1

model = temp_pytorch_unet.UNet(num_class).to(device)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=3000)

# prediction

import math

class ISBI_Dataset_test(Dataset):

    def __init__(self, tfms=None):
        self.fnames = np.array([f'image_{i}.png' for i in range(1,31)])
        self.tfms = tfms
            
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(DATA,TEST,fname), cv2.IMREAD_GRAYSCALE)

        if self.tfms is not None:
            augmented = self.tfms(image=img)
            img = augmented['image']

        img = img/255.0
        img = temp_simulation.oned_to_threed(img)
        img = torch.from_numpy(img.astype(np.float32, copy=False))     
        return img

model.eval()   # Set model to evaluate mode

test_dataset = ISBI_Dataset_test(tfms=temp_simulation.get_aug_test())
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)

inputs = next(iter(test_loader))
inputs = inputs.to(device)

pred = model(inputs)

pred = pred.data.cpu().numpy()
print(pred.shape)

# Change channel-order and make 3 channels for matplot
input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

# Map each channel (i.e. class) to each color
pred_rgb = [temp_helper.masks_to_colorimg(x) for x in pred]

for i, mask in enumerate(pred_rgb):
    plt.imshow(mask)
    plt.savefig(os.path.join(OUTPUT, f'mask_{i}.png'))
    plt.clf()
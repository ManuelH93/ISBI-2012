import os,sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import helper
import simulation
import random
import cv2
import torch

###########################################################
# Define parameters
###########################################################

DATA = 'processed_data'
TRAIN = 'train'
MASKS = 'masks'
OUTPUT = 'output'
random.seed(2021)

###########################################################
# Define dataset
###########################################################

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

class ISBI_Dataset(Dataset):

    def __init__(self, train = True, tfms=None):
        ids = np.array([f'image_{i}.png' for i in range(1,31)])
        random.shuffle(ids)
        split = int(0.8 * len(ids))
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
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img.astype(np.float32, copy=False))

        mask = mask/255.0
        mask = simulation.oned_to_twod(mask)
        mask = torch.from_numpy(mask.astype(np.int64, copy=False))
        
        return img, mask

###########################################################
# Test if dataset load works
###########################################################

ds = ISBI_Dataset(tfms = simulation.get_aug())
dl = DataLoader(ds,batch_size=4)
imgs,masks = next(iter(dl))
print(imgs.shape)
print(masks.shape)

# Convert tensors back to arrays

imgs = imgs.numpy()
imgs = np.squeeze(imgs)
masks = masks.numpy()
masks = [simulation.twod_to_oned(mask) for mask in masks]

for image, mask in zip(imgs,masks):
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.clf()
    plt.imshow(mask, cmap='gray')
    plt.show()
    plt.clf()


train_set = ISBI_Dataset(2000, imgs_train, masks_train, train=True, transform = trans)
val_set = ISBI_Dataset(200, imgs_train, masks_train, train=True, transform = trans)

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
import torch.nn as nn
import pytorch_unet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = pytorch_unet.UNet(1)
model = model.to(device)

summary(model, input_size=(3, 576, 576))


from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss

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

    # for figure
    epochs = []
    train_loss = []
    val_loss = []

    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        epochs.append(epoch+1)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
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
                        scheduler.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
           
            # collect statistics for figure
            if phase == 'train':
                train_loss.append(metrics['loss']/epoch_samples)
            else:
                val_loss.append(metrics['loss']/epoch_samples)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # Save loss figure
    plt.plot(epochs, train_loss, color='g', label = 'train')
    plt.plot(epochs, val_loss, color='orange', label = 'test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses')
    plt.legend(loc="upper left")
    #plt.show()
    plt.savefig(os.path.join(OUTPUT, 'losses.png'))
    plt.clf()

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

model = pytorch_unet.UNet(num_class).to(device)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=40)

# prediction

import math

model.eval()   # Set model to evaluate mode

test_dataset = ISBI_Dataset(3, imgs_test, masks_test, train=False, transform = trans)
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)

inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.to(device)

pred = model(inputs)

pred = pred.data.cpu().numpy()
print(pred.shape)

# Change channel-order and make 3 channels for matplot
input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

# Map each channel (i.e. class) to each color
target_masks_rgb = [helper.masks_to_colorimg(x) for x in labels.cpu().numpy()]
pred_rgb = [helper.masks_to_colorimg(x) for x in pred]

helper.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])
#plt.show()
plt.savefig(os.path.join(OUTPUT, 'prediction.png'))
plt.clf()
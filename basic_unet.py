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
random.seed(2021)

###########################################################
# Define dataset
###########################################################

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
        mask = simulation.center_crop(mask)
        mask = simulation.oned_to_twod(mask)
        mask = torch.from_numpy(mask.astype(np.float32, copy=False))
        
        return img, mask

###########################################################
# Test if dataset load works
###########################################################

#ds = ISBI_Dataset(tfms = simulation.get_aug_train())
#dl = DataLoader(ds,batch_size=4)
#imgs,masks = next(iter(dl))
#print(imgs.shape)
#print(masks.shape)

# Convert tensors back to arrays

#imgs = imgs.numpy()
#imgs = np.squeeze(imgs)
#masks = masks.numpy()
#masks = [simulation.twod_to_oned(mask) for mask in masks]

#for image, mask in zip(imgs,masks):
#    plt.imshow(image, cmap='gray')
#    plt.show()
#    plt.clf()
#    plt.imshow(mask, cmap='gray')
#    plt.show()
#    plt.clf()

###########################################################
# Load test and validation dataset
###########################################################

train_set = ISBI_Dataset(train=True, tfms=simulation.get_aug_train())
val_set = ISBI_Dataset(train=False, tfms=simulation.get_aug_train())

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

###########################################################
# Load U-net
###########################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = pytorch_unet.UNet()
model = model.to(device)

summary(model, input_size=(1, 572, 572))

###########################################################
# Define loss calculation
###########################################################

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

###########################################################
# Define training
###########################################################

def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    early_stopping = False

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
                epochs_no_improve = 0
            elif phase == 'val' and epoch_loss >= best_loss:
                epochs_no_improve += 1
                if epochs_no_improve == 5:
                    print('Early stopping!')
                    early_stopping = True

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        if early_stopping == True:
            break
        else:
            continue
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

###########################################################
# Run model
###########################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = pytorch_unet.UNet().to(device)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1500, gamma=0.1)

model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=3000)

###########################################################
# Predict
###########################################################

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
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img.astype(np.float32, copy=False))        
        return img

model.eval()   # Set model to evaluate mode

test_dataset = ISBI_Dataset_test(tfms=simulation.get_aug_test())
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)

inputs = next(iter(test_loader))
inputs = inputs.to(device)
preds = model(inputs)

preds = preds.data.cpu().numpy()
print(preds.shape)

# Convert tensors back to arrays

preds = [simulation.twod_to_oned(pred) for pred in preds]

for i,pred in enumerate(preds):
    plt.imshow(pred, cmap='gray')
    plt.savefig(os.path.join(OUTPUT, f'prediction{i+1}.png'))
    #plt.show()
    plt.clf()
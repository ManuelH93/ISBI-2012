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
SEED = 'some'
ids = np.array([f'image_{i}.png' for i in range(1,31)])

###########################################################
# Set seed
###########################################################

def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_some(seed):
    random.seed(seed)
    torch.manual_seed(seed)

if SEED == 'all':
    print("[ Seed setting : slow and reproducible ]")
    seed_all(2001)
else:
    print("[ Seed setting : fast and random ]")
    seed_some(2001)

###########################################################
# Define dataset
###########################################################

class ISBI_Dataset(Dataset):

    def __init__(self, train = True, tfms=None):
        self.fnames = ids
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
#print(imgs.shape, masks.shape)
#print(imgs.dtype, masks.dtype)
#for x in [imgs.numpy(), masks.numpy()]:
#    print(x.min(), x.max(), x.mean(), x.std())

# Convert tensors back to arrays

#imgs = imgs.numpy()
#masks = masks.numpy()
#masks = [mask[1] for mask in masks]

#for image, mask in zip(imgs,masks):
#    plt.imshow(np.squeeze(image), cmap='gray')
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

batch_size = 1

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
          
                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
           
            # collect statistics for figure and take lr step
            if phase == 'train':
                train_loss.append(metrics['loss']/epoch_samples)
                scheduler.step()
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
                if epochs_no_improve == 500:
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
    plt.savefig(os.path.join(OUTPUT, 'losses.png'))
    #plt.show()
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
optimizer_ft = optim.SGD(model.parameters(), lr=1e-2, momentum = 0.99)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1000, gamma=0.1)

model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=1000)

###########################################################
# Predict
###########################################################

class ISBI_Dataset_test(Dataset):

    def __init__(self, tfms=None):
        self.fnames = np.array([f'image_{i}.png' for i in range(1,4)])
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
        return img

model.eval()   # Set model to evaluate mode

test_dataset = ISBI_Dataset_test(tfms=simulation.get_aug_test())
# Important to keep batch size equalt to one, as each image gets
# split into several tiles and is then put back together
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

masks = []
for inputs in test_loader:
    inputs = inputs.to(device).numpy()
    inputs = simulation.crop(inputs)
    preds = model(inputs)
    preds = preds.data.cpu()
    # Create class probabilities
    preds = preds.softmax(dim = 1).numpy()
    # Keep probabilities for membrane only
    preds = [pred[1] for pred in preds]
    # Create membrane and background based on probabilities
    for prediction in preds:
        prediction = simulation.probs_to_mask(prediction)
    mask = simulation.stitch(preds)
    masks.append(mask)

masks = np.array(masks, dtype=np.float32)
tiff.imsave(os.path.join(OUTPUT,'submission.tif'), masks)

for i, mask in enumerate(masks):
    plt.imshow(mask, cmap='gray')
    plt.savefig(os.path.join(OUTPUT, f'mask_{i}.png'))
    #plt.show()
    plt.clf()
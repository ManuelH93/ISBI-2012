######################################################################
# Import modules
######################################################################

import torch
import torch.nn as nn
from torch.autograd import Variable
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
TEST = os.path.join(OUTPUT,'test')
mean = np.array([0])
std = np.array([1])

######################################################################
# Set random seeds
######################################################################

random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
else:
    torch.manual_seed(SEED)

######################################################################
# Define Dataset
######################################################################

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
        img = (img/255.0 - mean)/std
        img = torch.from_numpy(img.astype(np.float32, copy=False))
        # Need to add channel dimension
        img = torch.unsqueeze(img,0)
        mask = cv2.imread(os.path.join(MASKS,fname),cv2.IMREAD_GRAYSCALE)
        mask = mask/255.0
        mask = crop(mask)
        mask = torch.from_numpy(mask.astype(np.int64, copy=False))
        
        return img, mask

######################################################################
# Prepare the dataset
######################################################################

def prepare_data():

    # Determine train and test images

    #ids = np.array([f'image_{i}' for i in range(1,31)])
    #random.shuffle(ids)
    # //MH update train/val split
    #cut_off = int(len(ids)*0.8)
    #train = ids[:cut_off]
    #val = ids[cut_off:]

    train = np.array(['image_2'])
    val = np.array(['image_16'])

    # Test if dataset is loading images as expected
    #ds = IsbiDataset(train)
    #dl = torch.utils.data.DataLoader(ds,batch_size=25,shuffle=False)
    #imgs,masks = next(iter(dl))
    #fig=plt.figure(figsize=(24, 20))
    #for ind, (image, mask) in enumerate(zip(imgs, masks)):
    #    fig.add_subplot(5, 5, 1+ind)
    #    plt.axis('off')
    #    # Remove channel dimension
    #    image = torch.squeeze(image)
    #    # Crop image to fit mask
    #    image = crop(image)
    #    plt.imshow(image*255, cmap='gray', vmin=0, vmax=255)
    #    plt.imshow(mask, cmap="hot", alpha=0.5)
    #plt.savefig(os.path.join(OUTPUT,'train_dataset.png'))
    #plt.clf()
        
    #del ds,dl,imgs,masks

    # Create a dataloader for train and validation set

    ds_t = IsbiDataset(train)
    ds_v = IsbiDataset(val)

    # create a data loader for train and test sets
    train_dl = torch.utils.data.DataLoader(ds_t, batch_size=1, shuffle=True)
    val_dl = torch.utils.data.DataLoader(ds_v, batch_size=1, shuffle=False)

    return train_dl, val_dl

######################################################################
# Define the model
######################################################################

def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size = 3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size = 3),
        nn.ReLU(inplace=True)
    )
    return conv

def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1,64)
        self.down_conv_2 = double_conv(64,128)
        self.down_conv_3 = double_conv(128,256)
        self.down_conv_4 = double_conv(256,512)
        self.down_conv_5 = double_conv(512,1024)
        
        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels = 1024,
            out_channels = 512,
            kernel_size = 2,
            stride = 2)
        
        self.up_conv_1 = double_conv(1024, 512)
        
        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels = 512,
            out_channels = 256,
            kernel_size = 2,
            stride = 2)
        
        self.up_conv_2 = double_conv(512, 256)
        
        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels = 256,
            out_channels = 128,
            kernel_size = 2,
            stride = 2)
        
        self.up_conv_3 = double_conv(256, 128)
        
        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels = 128,
            out_channels = 64,
            kernel_size = 2,
            stride = 2)
        
        self.up_conv_4 = double_conv(128, 64)
        
        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=2,
            kernel_size=1
        )
    
    def forward(self, image):
        # batch size, channel, hight, width
        # encoder
        x1 = self.down_conv_1(image) #
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2) #
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4) #
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6) #
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        
        # decoder
        x = self.up_trans_1(x9)
        y = crop_img(x7,x)
        x = self.up_conv_1(torch.cat([x, y], 1))
        
        x = self.up_trans_2(x)
        y = crop_img(x5,x)
        x = self.up_conv_2(torch.cat([x, y], 1))
        
        x = self.up_trans_3(x)
        y = crop_img(x3,x)
        x = self.up_conv_3(torch.cat([x, y], 1))
        
        x = self.up_trans_4(x)
        y = crop_img(x1,x)
        x = self.up_conv_4(torch.cat([x, y], 1))
        
        x = self.out(x)
        print(x.size())
        return(x)

######################################################################
# Train and evaluate the model
######################################################################

def get_loss(dl, model):
    criterion = torch.nn.functional.cross_entropy
    loss = 0
    for X, y in dl:
        if torch.cuda.is_available():
            X, y = Variable(X).cuda(), Variable(y).cuda()
        else:
            X, y = Variable(X), Variable(y)
        output = model(X)
        loss += criterion(output, y).item()
    loss = loss / len(dl)
    return loss

def train_model(train_dl, val_dl, model):
    # define the optimization
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.functional.cross_entropy
    
    iters = []
    train_losses = []
    val_losses = []

    it = 0
    min_loss = np.inf
    model.train()
    for epoch in range(1):
        print(f'epoch: {epoch}')
        for i, (X, y) in enumerate(train_dl):
            if torch.cuda.is_available():
                X = Variable(X).cuda()  # [N, 1, H, W]
                y = Variable(y).cuda()  # [N, H, W] with class indices (0, 1)
            else:
                X = Variable(X)  # [N, 1, H, W]
                y = Variable(y)
            output = model(X)  # [N, 2, H, W]
            loss = criterion(output, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            it += 1
            iters.append(it)
            train_losses.append(loss.item())
            model.eval()
            # Get's average loss on validation data
            val_loss = get_loss(val_dl, model)
            model.train()
            val_losses.append(val_loss)

            if val_loss < min_loss:
                torch.save(model.state_dict(), os.path.join(OUTPUT,'bst_unet.model'))
                min_loss = val_loss
            print(f'min_loss: {min_loss}')

    model.eval()
    val_loss = get_loss(val_dl, model)
    if val_loss < min_loss:
        torch.save(model.state_dict(), os.path.join(OUTPUT,'bst_unet.model'))
    return iters, train_losses, val_losses

######################################################################
# Make a prediction
######################################################################

def predict(fname, model):
    img = cv2.imread(os.path.join(TEST,fname), cv2.IMREAD_GRAYSCALE)
    img = (img/255.0 - mean)/std
    img = torch.from_numpy(img.astype(np.float32, copy=False))
    
    if torch.cuda.is_available():
        img = Variable(img).cuda()
    else:
        img = Variable(img)
    
    img = torch.unsqueeze(img,0)
    img = torch.unsqueeze(img,0)
    model.eval()
    output = model(img)

    return img, output

######################################################################
# Main
######################################################################

train_dl, val_dl = prepare_data()
print(len(train_dl.dataset), len(val_dl.dataset))
# define the network
model = UNet()
if torch.cuda.is_available():
    model.cuda()
# train the model
iters, train_losses, val_losses = train_model(train_dl, val_dl, model)
# evaluate the model
print(iters)
print(train_losses)
print(val_losses)
plt.plot(iters, train_losses)
plt.plot(iters, val_losses)
plt.savefig(os.path.join(OUTPUT,'evaluation.png'))
plt.clf()


#make a single prediction
image, mask_pred = predict('image_7.png', model)
print('marker1')
print(type(mask_pred))
print('marker2')
print(mask_pred.size())
print(mask_pred)
#image = crop(image)
#plt.imshow(mask_pred, cmap="hot", alpha=0.5)
#plt.imshow(image*255, cmap='gray', vmin=0, vmax=255)
#plt.savefig(os.path.join(OUTPUT,'prediction.png'))
#plt.clf()


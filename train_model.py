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
    return torch.from_numpy(img.astype(dtype, copy=False))

class IsbiDataset(torch.utils.data.Dataset):
    def __init__(self, ids):
        self.fnames = [fname for fname in os.listdir(TRAIN) if fname.split('.')[0] in ids]
            
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(TRAIN,fname), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(MASKS,fname),cv2.IMREAD_GRAYSCALE)
        return img2tensor((img/255.0 - mean)/std),img2tensor(mask/255.0)

######################################################################
# Prepare the dataset
######################################################################

def prepare_data():

    # Determine train and test images

    ids = np.array([f'image_{i}' for i in range(1,31)])
    random.shuffle(ids)
    cut_off = int(len(ids)*0.8)
    train = ids[:cut_off]
    val = ids[cut_off:]

    # Test if dataset is loading images as expected

    #ds = IsbiDataset(train)
    #dl = torch.utils.data.DataLoader(ds,batch_size=25,shuffle=False)
    #imgs,masks = next(iter(dl))
    #fig=plt.figure(figsize=(24, 20))
    #for ind, (image, mask) in enumerate(zip(imgs, masks)):
    #    fig.add_subplot(5, 5, 1+ind)
    #    plt.axis('off')
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
# Train the model
######################################################################

def train_model(train_dl, model):
    # define the optimization
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

######################################################################
# Evaluate the model
######################################################################

def evaluate_model(eval_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(eval_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

######################################################################
# Main
######################################################################

train_dl, eval_dl = prepare_data()
print(len(train_dl.dataset), len(eval_dl.dataset))
# define the network
model = UNet()
# train the model
train_model(train_dl, model)
# evaluate the model
#acc = evaluate_model(test_dl, model)
#print('Accuracy: %.3f' % acc)
# make a single prediction (expect class=1)
#row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
#yhat = predict(row, model)
#print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))
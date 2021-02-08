import os,sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import helper
import simulation

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

import torchvision.utils

from torchsummary import summary
import torch
import torch.nn as nn
import pytorch_unet

from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss

import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

import math


# Generate some random images
input_images, target_masks = simulation.generate_random_data(192, 192, count=4)

input_images = np.squeeze(input_images)
print(input_images.shape)

target_masks = np.squeeze(target_masks)
print(target_masks.shape)

#layer1 = target_masks[0]
#layer2 = target_masks[1]
#layer3 = target_masks[2]
#layer4 = target_masks[0]
#layer5 = target_masks[0]
#layer6 = target_masks[0]

#print('layer1')
#print(layer1.min(), layer1.max(), layer1.mean(), layer1.std())
#print(layer1[100:115,70:85])
#print('layer2')
#print(layer2.min(), layer2.max(), layer2.mean(), layer2.std())
#print(layer2[100:115,70:85])
#print('layer3')
#print(layer3.min(), layer3.max(), layer3.mean(), layer3.std())
#print(layer3[100:115,70:85])
#print('layer4')
#print(layer4.min(), layer4.max(), layer4.mean(), layer4.std())
#print(layer4[100:115,70:85])
#print('layer5')
#print(layer5.min(), layer5.max(), layer5.mean(), layer5.std())
#print(layer5[100:115,70:85])
#print('layer6')
#print(layer6.min(), layer6.max(), layer6.mean(), layer6.std())
#print(layer6[100:115,70:85])

#plt.imshow(target_masks)
#plt.show()
#plt.clf()

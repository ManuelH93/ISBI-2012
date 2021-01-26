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

OUTPUT = 'output'
MASKS = os.path.join(OUTPUT,'mask')


def img2tensor(img,dtype:np.dtype=np.float32):
    img = torch.from_numpy(img.astype(dtype, copy=False))
    return img

def crop(image):
    target_size = 388
    tensor_size = 572
    delta = tensor_size - target_size
    delta = delta // 2
    return image[delta:tensor_size-delta, delta:tensor_size-delta]



mask = cv2.imread(os.path.join(MASKS,'image_1.png'),cv2.IMREAD_GRAYSCALE)
print(type(mask))
#mask = img2tensor(mask/255)
mask = torch.from_numpy(np.array(mask/255, dtype=np.int64))
print(type(mask))
mask = crop(mask)
print(type(mask))

print('test')
del mask
mask = np.zeros([512,512], dtype=np.uint8)
print(type(mask))
mask = np.clip(mask, 0, 1)
print(type(mask))
#mask = Image.fromarray(mask)
#print(type(mask))
mask = torch.from_numpy(np.array(mask, dtype=np.int64))
print(type(mask))
print(torch.__version__)

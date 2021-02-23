import torch
import os
import pytorch_unet
from torch.utils.data import Dataset, DataLoader
import simulation
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy

MODEL = 'trained_model'
MODEL_VERSION = '2021.02.18'

DATA = 'processed_data'
TEST = 'test'
OUTPUT = 'output'

PRED_THRESHOLD_v1 = 0.5
PRED_THRESHOLD_v2 = 0.5

# Load data

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = pytorch_unet.UNet().to(device)

model.load_state_dict(torch.load(os.path.join(MODEL,MODEL_VERSION,'bst_unet.model'), map_location=torch.device(device)))

model.eval()   # Set model to evaluate mode

test_dataset = ISBI_Dataset_test(tfms=simulation.get_aug_test())
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)

inputs = next(iter(test_loader))
inputs = inputs.to(device)

pred = model(inputs)

preds = pred.data.cpu().numpy()
print(preds.shape)

preds_v1 = copy.deepcopy(preds)
preds_v2 = copy.deepcopy(preds)

#########################################################
# Use only class 1
#########################################################

preds_v1 = [simulation.twod_to_oned(pred) for pred in preds_v1]

for prediction in preds_v1:
    # Replace 1s with 0s and 0s with 1s
    indices_one = prediction >= PRED_THRESHOLD_v1
    indices_zero = prediction < PRED_THRESHOLD_v1
    prediction[indices_one] = 1
    prediction[indices_zero] = 0

for i, mask in enumerate(preds_v1):
    plt.imshow(mask, cmap='gray')
    plt.savefig(os.path.join(OUTPUT, f'mask_{i}_v1.png'))
    #plt.show()
    plt.clf()

#########################################################
# Use predictions for both classes
#########################################################

# //MH: try using softmax

for prediction in preds_v2:
    # Turn utils into 0s and 1s
    # For first class, 1 is background and 0 membrane.
    # We want to switch this around
    indices_zero = prediction[0] >= PRED_THRESHOLD_v2
    indices_one = prediction[0] < PRED_THRESHOLD_v2
    prediction[0][indices_one] = 1
    prediction[0][indices_zero] = 0
    
    indices_one = prediction[1] >= PRED_THRESHOLD_v2
    indices_zero = prediction[1] < PRED_THRESHOLD_v2
    prediction[1][indices_one] = 1
    prediction[1][indices_zero] = 0

masks = []
for prediction in preds_v2:
    mask = (prediction[0]+prediction[1])/2
    masks.append(mask)

for i, mask in enumerate(masks):
    plt.imshow(mask, cmap='gray')
    plt.savefig(os.path.join(OUTPUT, f'mask_{i}_v2.png'))
    #plt.show()
    plt.clf()

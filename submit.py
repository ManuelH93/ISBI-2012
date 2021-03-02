import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import copy
import tifffile as tiff

import pytorch_unet
import simulation

MODEL = 'trained_model'
MODEL_VERSION = '2021.02.24_and_model'

DATA = 'processed_data'
TEST = 'test'
OUTPUT = 'output'

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
        return img

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = pytorch_unet.UNet().to(device)

model.load_state_dict(torch.load(os.path.join(MODEL,MODEL_VERSION,'bst_unet.model'), map_location=torch.device(device)))

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
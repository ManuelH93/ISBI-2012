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
MODEL_VERSION = '2021.02.24'

DATA = 'processed_data'
TEST = 'test'
OUTPUT = 'output'

PRED_THRESHOLD_v1 = 0
PRED_THRESHOLD_v2 = 0

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

preds = pred.data.cpu()
print(preds.shape)

# Create class porbabilities
preds = preds.softmax(dim = 1).numpy()

# Keep probabilities for membrane only
preds = [pred[1] for pred in preds]

for prediction in preds:
    # Create membrane and background based on probabilities
    indices_one = prediction >= 0.5
    indices_zero = prediction < 0.5
    prediction[indices_one] = 1
    prediction[indices_zero] = 0

for i, mask in enumerate(preds):
    plt.imshow(mask, cmap='gray')
    plt.savefig(os.path.join(OUTPUT, f'mask_{i}.png'))
    plt.show()
    plt.clf()

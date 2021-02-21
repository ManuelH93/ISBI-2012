import torch
import os
import pytorch_unet
from torch.utils.data import Dataset, DataLoader
import simulation
import numpy as np
import matplotlib.pyplot as plt
import cv2


MODEL = 'trained_model'
MODEL_VERSION = '2021.02.18'

DATA = 'processed_data'
TEST = 'test'

PRED_THRESHOLD = 2

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
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        
inputs = next(iter(test_loader))
inputs = inputs.to(device)
preds = model(inputs)

preds = preds.data.cpu().numpy()
print(preds.shape)

# Convert tensors back to arrays

preds = [simulation.twod_to_oned(pred) for pred in preds]

for prediction in preds:
    # Replace 1s with 0s and 0s with 1s
    indices_one = prediction >= PRED_THRESHOLD
    indices_zero = prediction < PRED_THRESHOLD
    prediction[indices_one] = 1
    prediction[indices_zero] = 0

for i,pred in enumerate(preds):
    plt.imshow(pred, cmap='gray')
    #plt.savefig(os.path.join(OUTPUT, f'prediction{i+1}.png'))
    plt.show()
    plt.clf()

import torch
import os
import pytorch_unet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import simulation
import numpy as np
import helper
import random
import tifffile as tiff


MODEL = 'trained_model'

DATA = 'raw_data'
OUTPUT = 'output'
random.seed(2021)

# Load data
imgs_train, masks_train, imgs_test, masks_test = simulation.load_data(DATA)

class ISBI_Dataset(Dataset):
    def __init__(self, count, imgs_train, masks, train, transform=None):
        self.input_images, self.target_masks = simulation.reshape_images(imgs_train, masks, train, count=count)        
        self.transform = transform
    
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):        
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)
        
        return [image, mask]

trans = transforms.Compose([
    transforms.ToTensor(),
])

def crop(image):
    target_size = 512
    array_size = 576
    delta = array_size - target_size
    delta = delta // 2
    return image[delta:array_size-delta, delta:array_size-delta]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_class = 1

model = pytorch_unet.UNet(num_class).to(device)

model.load_state_dict(torch.load(os.path.join(MODEL,'bst_unet - 2021.01.10.model'), map_location=torch.device(device)))

model.eval()   # Set model to evaluate mode

test_dataset = ISBI_Dataset(30, imgs_test, masks_test, train=False, transform = trans)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False, num_workers=0)
        
inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)

pred = model(inputs)

pred = pred.data.cpu().numpy()
print(pred.shape)

# Map each channel (i.e. class) to each color
# Note we chose yellow as colour for cell membrane
# The following code relies on this colour being chosen.
pred_rgb = [helper.masks_to_colorimg(x) for x in pred]

predictions = []

for prediction in pred_rgb:
    # The whole mask either is white (RGB = (255,255,255))
    # or it is yellow (RGB = (255,255,0)).
    # We therefore only need the blue channel to identify the mask.
    prediction = prediction[:,:,2]
    # Replace 255s with 1
    indices_255 = prediction == 255
    prediction[indices_255] = 1
    # Crop image to size of train image (512*512)
    prediction = crop(prediction)
    predictions.append(prediction)

predictions = np.array(predictions, dtype=np.float32)
tiff.imsave(os.path.join(OUTPUT,'submission.tif'), predictions)

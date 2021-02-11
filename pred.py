import torch
import os
import pytorch_unet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import simulation
import numpy as np
import helper
import matplotlib.pyplot as plt
import random


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

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    
    return inp


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_class = 1

model = pytorch_unet.UNet(num_class).to(device)

model.load_state_dict(torch.load(os.path.join(MODEL,'bst_unet.model'),map_location=torch.device(device)))



model.eval()   # Set model to evaluate mode

test_dataset = ISBI_Dataset(3, imgs_test, masks_test, train=False, transform = trans)
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)
        
inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.to(device)

pred = model(inputs)

pred = pred.data.cpu().numpy()
print(pred.shape)

# Change channel-order and make 3 channels for matplot
input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

# Map each channel (i.e. class) to each color
target_masks_rgb = [helper.masks_to_colorimg(x) for x in labels.cpu().numpy()]
pred_rgb = [helper.masks_to_colorimg(x) for x in pred]

helper.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])
#plt.show()
plt.savefig(os.path.join(OUTPUT,'prediction.png'))
plt.clf()


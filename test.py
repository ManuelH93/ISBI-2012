######################################################################
# Import modules
######################################################################

import torch
import numpy as np
from skimage.morphology import binary_opening, disk, label
from PIL import Image

torch.manual_seed(2021)
r1 = -1
r2 = 1

pred = (r1 - r2) * torch.rand((1, 2, 2, 2)) + r2

print('mask1')
print(pred)

pred = torch.softmax(pred,dim=1)
pred = pred.data.cpu().numpy()

for masks in pred:
    channels, height, width = masks.shape
    colorimg = np.ones((height, width), dtype=np.float32) * 255
    for y in range(height):
        for x in range(width):
            if masks[1,y,x] >= 0.5:
                colorimg[y,x] = 1
            else:
                colorimg[y,x] = 0

print('colorimg')
print(colorimg)

#print('##########################')
#print('# GitHub')
#print('##########################')

#pred = (r1 - r2) * torch.rand((1, 6, 2, 2)) + r2

#print('mask1')
#print(pred)

#pred = torch.sigmoid(pred)
#pred = pred.data.cpu().numpy()

#for masks in pred:
#    colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56)])
#    channels, height, width = masks.shape
#    print(masks.shape)
#    colorimg = np.ones((height, width, 3), dtype=np.float32) * 255
#    print('masks')
#    print(masks)
#    print('colorimg')
#    print(colorimg)
    

#    for y in range(height):
#        for x in range(width):
#            print('masks[:,y,x]')
#            print(masks[:,y,x])
#            print('masks[:,y,x] > 0.5')
#            print(masks[:,y,x] > 0.5)
#            print('colors[masks[:,y,x] > 0.5]')
#            print(colors[masks[:,y,x] > 0.5])
#            selected_colors = colors[masks[:,y,x] > 0.5]

#            if len(selected_colors) > 0:
#                colorimg[y,x,:] = np.mean(selected_colors, axis=0)
#print('colorimg')
#print(colorimg)

# Change channel-order and make 3 channels for matplot
#input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

# Map each channel (i.e. class) to each color
#target_masks_rgb = [helper.masks_to_colorimg(x) for x in labels.cpu().numpy()]
#pred_rgb = [helper.masks_to_colorimg(x) for x in pred

##########################
# Kaggle
##########################

#mask = torch.rand((1, 2, 5, 5))
#print('mask1')
#print(mask)
#mask = mask[0, 0]
#print('mask2')
#print(mask)
#mask = torch.sigmoid(mask).data.cpu().numpy()
#print('mask3')
#print(mask)
#mask = binary_opening(mask > 0.5)
#print(mask)
#print('mask4')
#mask = Image.fromarray(mask.astype(np.uint8))
#print('mask5')
#print(mask)
#mask = np.array(mask).astype(np.bool)
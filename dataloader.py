import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = T.Compose([
T.Resize((256,256)),
T.ToTensor(),
T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

image_folder = r'C:\Users\YongTaek\Desktop\CNN_DATASET'

trainset = torchvision.datasets.ImageFolder(root = image_folder, transform = transform)
len(trainset)

trainset
train_loader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0)

dataiter = iter(train_loader)
dataiter
image, labels= dataiter.next()
image, labels

def imshow(img):
    img = img / 2+0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1,2,0)))
    print(np_img.shape)

import numpy as np

print(image.shape)
imshow(torchvision.utils.make_grid(image, nrow=4))
print(image.shape)
print( (torchvision.utils.make_grid(image)).shape )
print(''.join('%5s ' %labels[j] for j in range(8)))


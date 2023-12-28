from torch.utils.data import Dataset,DataLoader,random_split
import torch,glob,os
import torchvision.transforms as transforms
from torchvision.io import read_image
from PytorchDataset import LanesDataset
from torchvision import models
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from Network import Net
from training import fit
from torch import optim
import matplotlib.pyplot as plt

images_directory = r'C:\Users\monsi\Desktop\project\lane_detection\train_data\clips'
label_directory = r'C:\Users\monsi\Desktop\project\lane_detection\train_data\gray_labels'
transformations = transforms.Compose([transforms.GaussianBlur((5,5))])

dataset = LanesDataset(images_directory,label_directory,transformations)

train_dataset,test_dataset,valid_dataset = random_split(dataset,[2908,359,359])

train_loader = DataLoader(train_dataset,batch_size = 8,shuffle = True)
test_loader = DataLoader(test_dataset,batch_size = 8,shuffle = True)
valid_loader = DataLoader(valid_dataset,batch_size = 8,shuffle = True)

model = Net()
print(model)
cost_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.0005)
epochs = 30 
x = fit(epochs,model,train_loader,cost_fn,optimizer,device="cuda")

plt.plot(x,[epoch for epoch in range(1,epochs+1)])
plt.show()
plt.savefig(r'C:\Users\monsi\Desktop\project\lane_detection\image.jpg')
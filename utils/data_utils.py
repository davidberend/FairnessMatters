import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from img_set import Img_Dataset

def make_dataloader(samples,labels,img_size=(32,32), batch_size=256, transform_test=None, shuffle=False, num_workers=4):
    dataset = Img_Dataset(samples,labels = labels,img_size = img_size, transform=transform_test)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader
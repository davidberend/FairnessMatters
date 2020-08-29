import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from .img_set import Img_Dataset,Img_Dataset_Iter

def make_dataloader(samples,labels,img_size=(32,32), batch_size=256, transform_test=None, shuffle=False, num_workers=2):
    dataset = Img_Dataset(samples,labels = labels,img_size = img_size, transform=transform_test)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader

def make_dataloader_iter(samples,labels,img_size=(32,32), batch_size=256, transform_test=None, shuffle=False, num_workers=2):
    dataset = Img_Dataset_Iter(samples,labels = labels,img_size = img_size, transform=transform_test)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader

def process_data(path):
    X,y=[],[]
    f=open(path,'r')
    lines=f.readlines()
    classes=set()
    for line in lines:
        temp=line.strip()
        if temp is not None:
            X.append(temp.split('\t')[0])
            y.append(int(temp.split('\t')[2]))
            classes.add(temp.split('\t')[2])
    return X,y,len(classes)

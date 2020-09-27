import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import os
import numpy as np
import math
from .img_set import Img_Dataset,Img_Dataset_Iter

def get_min_max_sample(num_sample):
    max_sample = np.inf
    min_sample = 0
    for key in num_sample:
        max_sample = min(max_sample,np.quantile(list(num_sample[key].values()),0.8))
        min_sample = max(min_sample,np.quantile(list(num_sample[key].values()),0.2))
    return min_sample, max_sample

def update(select_size, threshold, num,ds_num):
    threshold -= num
    ds_num -= 1
    if ds_num!=0:
        select_size = math.ceil(threshold*1.0/ds_num)
    else:
        select_size = threshold
    return select_size, threshold, ds_num

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

def process_data_ood(path,gender=False):
    X,y=[],[]
    f=open(path,'r')
    lines=f.readlines()
    classes=set()
    genderMap={
        'male':0,
        'female':1
    }
    race=[]
    for line in lines:
        temp=line.strip()
        if temp is not None:
            if len(temp.split('\t'))<4:continue
            if not os.path.exists(temp.split('\t')[0]):continue
            X.append(temp.split('\t')[0])

            y.append(int(temp.split('\t')[2]))
            classes.add(temp.split('\t')[2])
            race.append(temp.split('\t')[1])
    return X,y,len(classes),race
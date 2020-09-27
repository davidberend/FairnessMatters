import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from ood.OOD_techniques.glod import retrieve_scores, ConvertToGlod, calc_gaussian_params
from models.Generalmodels import create_Resnet
import utils.data_utils

DarkBlue= '#051c2c'
LightBlue= '#3ba9f5'
LightGrey='#989898'
Blue= '#2140e6'
BrightBlue='#aae6f0'
Orange='#FF7828'
Green= '#009926'

def visualize_distribution(in_dist, out_dist, model_type, data_type, k):

    in_dist = in_dist
    out_dist = out_dist
    plt.hist(in_dist, bins=300, rwidth=0.8,  color=LightBlue,alpha=0.5)
    plt.hist(out_dist, bins=300, rwidth=0.8, color=Orange,alpha=0.5)
    plt.cla()

def convert(train_path, batch_size, weight_path, device, transform, version=50, num_classes=100, img_pixels= (224, 224)):
    # data
    train_set_X, train_set_y, _ = utils.data_utils.process_data(train_path)

    print('Using data:{}'.format(train_path))
    train_loader = utils.data_utils.make_dataloader_iter(train_set_X, train_set_y, img_size=img_pixels,
                                                batch_size=batch_size, transform_test=transform, shuffle=True)

    # Creat the original model
    print('Creating model using model :{}'.format(weight_path))
    model = create_Resnet(resnet_number=version, num_classes=num_classes,
                        gaussian_layer=False, weight_path=weight_path,
                        device=device)

    # Convert it to glod model
    print('Begin converting')
    model = ConvertToGlod(model, num_classes=num_classes)
    print('Begin calculating')
    covs, centers = calc_gaussian_params(model, train_loader, device, num_classes)
    print('Done Calculation')
    model.gaussian_layer.covs.data = covs
    model.gaussian_layer.centers.data = centers

    return model

def convert_and_get_scores(train_path, aug_loader, batch_size, weight_path, num_classes, glod_k = 100, transform=None):
    device='cuda:0'
    device = torch.device(device)
    
    # Load original model
    model = create_Resnet(resnet_number=50, num_classes=num_classes,
                      gaussian_layer=False, weight_path=weight_path,
                      device=device)

    # Convert to Glod
    glod_model = convert(train_path, batch_size, weight_path, device,transform)

    # Retrive scores
    aug_scores = retrieve_scores(glod_model, aug_loader, device, glod_k)

    return aug_scores

def split_data(scores, dataset,label_age,label_race, quantile=0.03):
    threshold=np.quantile(scores,quantile)
    diff = (scores-threshold)
    good_indices=np.where(diff>0)[0]
    splited_data = []
    for index in good_indices:
        splited_data.append([dataset[index],label_race[index],label_age[index]])
    return splited_data